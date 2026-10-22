# ============================================================
#  07_raft_fast.py
#  RAFT - Retrieval Augmented Fine-Tuning - 3 modèles
#  - Qwen2.5-7B  : pré-entraîné 3GPP → RAFT fine-tuning par-dessus
#  - LLaMA 3.1-8B: base → RAFT fine-tuning
#  - Mistral-7B  : base → RAFT fine-tuning
#  Dataset RAFT : raoulbia/3gpp-5g-nr-qa + contexte ChromaDB
#  Évaluation   : TeleQnA (ancien dataset)
#  Compatible CPU et GPU
#
#  CORRECTIFS (cf. cache_utils.py) :
#  - raft_dataset.json est maintenant validé par signature
#    (nombre de questions, MAX_TRAIN_SAMPLES, TOP_K, limites de
#    troncature des chunks). Le bug "400 demandées mais 300
#    chargées silencieusement depuis un vieux fichier" ne peut
#    plus se reproduire : un changement de ces paramètres
#    régénère le dataset.
#  - RAFT_DONE.json (renommé depuis RAFT_DONE.txt) contient une
#    signature qui CHAÎNE la signature du dataset RAFT : si le
#    dataset change, l'entraînement RAFT se réinvalide aussi.
#  - Le cache d'évaluation raft_{model}.json chaîne à son tour
#    la signature de l'entraînement RAFT : un ré-entraînement
#    invalide automatiquement l'éval qui en dépend.
#  - truncation_side="left" sur les tokenizers (entraînement ET
#    éval) : avec MAX_SEQ_LEN=256, la troncature par défaut à
#    DROITE coupait la fin du prompt — la question, les options
#    et "### Response:" — ce qui explique en grande partie la
#    chute de précision du RAFT (18-43%) par rapport au FT+RAG
#    (48-64%) qui lui utilise une longueur max bien plus grande.
#  - FORCE_REFRESH=1 régénère dataset + résultats d'éval.
#    FORCE_RETRAIN=1 force aussi un ré-entraînement RAFT complet.
# ============================================================

import os
import sys
import gc
import json
import yaml
import time
import torch
import random
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
from datasets import load_dataset, Dataset

# ── MONKEY-PATCH LlamaConfig ─────────────────────────────────
# LLaMA 3.1 utilise un format rope_scaling étendu (rope_type, low_freq_factor,
# high_freq_factor…) non supporté par certaines versions de transformers qui
# n'attendent que {type, factor}. On remplace la validation par une version
# tolérante qui normalise le dict avant de le vérifier.
try:
    from transformers.models.llama.configuration_llama import LlamaConfig as _LlamaConfig

    def _patched_rope_scaling_validation(self):
        if self.rope_scaling is None:
            return
        if not isinstance(self.rope_scaling, dict):
            raise ValueError("`rope_scaling` doit être un dict.")
        rs = self.rope_scaling
        # Normaliser rope_type -> type (format LLaMA 3.1)
        if "rope_type" in rs and "type" not in rs:
            rs["type"] = rs.pop("rope_type")
        # "llama3" n'existe pas dans transformers 4.41 → mapper vers "dynamic"
        rope_type = rs.get("type", "linear")
        if rope_type not in ("linear", "dynamic", "yarn", "longrope"):
            rope_type = "dynamic"
        # Conserver seulement les deux champs requis par cette version
        self.rope_scaling = {
            "type":   rope_type,
            "factor": float(rs.get("factor", 8.0))
        }

    _LlamaConfig._rope_scaling_validation = _patched_rope_scaling_validation
    print("STEP 1.5 : LlamaConfig rope_scaling patché ✅", flush=True)
except Exception as _e:
    print(f"STEP 1.5 : Patch LlamaConfig ignoré ({_e.__class__.__name__})", flush=True)
# ─────────────────────────────────────────────────────────────

print("STEP 1 : Imports OK", flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_utils import (
    compute_signature, write_marker, read_marker, is_marker_valid,
    migrate_legacy_marker, archive_stale_dir, load_json_cache, save_json_cache,
    FORCE_REFRESH, FORCE_RETRAIN
)

with open(os.environ.get("RAG_CONFIG", "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

CHROMA_DIR  = config["paths"]["chroma_db"]
COLL_NAME   = config["rag_baseline"]["collection_name"]
EMB_MODEL   = config["rag_baseline"]["embedding_model"]
TEST_FILE   = config["paths"]["test_teleqna"]
MODELS_DIR  = config["paths"]["models_dir"]
RESULTS_DIR = config["paths"]["results_dir"]

RAFT_DIR  = "./raft_models"
RAFT_DATA = "./raft_dataset.json"

# ── HYPER-PARAMÈTRES ────────────────────────────────────────
TOP_K             = 3
MAX_TRAIN_SAMPLES = 400
EVAL_BATCH_SIZE   = 2    # passer à 1 si OOM
CHUNK_LIMIT       = 120
EXPL_LIMIT        = 100
MAX_SEQ_LEN       = 256
EMB_BATCH_SIZE    = 64
GRAD_ACCUM_GPU    = 4
GRAD_ACCUM_CPU    = 8
RAFT_PROMPT_VERSION = "v2-left-trunc"
EVAL_MAX_NEW_TOKENS  = 15
# ────────────────────────────────────────────────────────────

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_4BIT = torch.cuda.is_available()

print(f"STEP 2 : Config OK — DEVICE={DEVICE.upper()}", flush=True)
if FORCE_REFRESH:
    print("⚠️  FORCE_REFRESH=1 — dataset RAFT et résultats d'éval seront recalculés", flush=True)
if FORCE_RETRAIN:
    print("⚠️  FORCE_RETRAIN=1 — les modèles RAFT seront ré-entraînés depuis zéro", flush=True)

os.makedirs(RAFT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ============================================================
#  MODÈLES
#  Qwen  -> RAFT par-dessus raoulbia/Qwen2.5-7B-3GPP-NR
#  LLaMA -> RAFT depuis base HuggingFace
#  Mistral -> RAFT depuis base HuggingFace
# ============================================================
HF_BASE_MODELS = {
    "qwen25_7b":  "raoulbia/Qwen2.5-7B-3GPP-NR",           # pas de dossier local → HuggingFace
    "llama31_8b": "./models/llama31_8b_finetuned",          # local ✅
    "mistral_7b": "./models/mistral_7b_finetuned"           # local ✅
}


# ============================================================
#  CHARGEMENT DATASET RAFT : raoulbia/3gpp-5g-nr-qa
# ============================================================
def _load_hf_questions(force_redownload: bool = False):
    print("\n📥 Chargement dataset RAFT : raoulbia/3gpp-5g-nr-qa", flush=True)
    if force_redownload:
        ds = load_dataset("raoulbia/3gpp-5g-nr-qa", download_mode="force_redownload")
    else:
        ds = load_dataset("raoulbia/3gpp-5g-nr-qa")
    splits = list(ds.keys())
    data   = ds["train"] if "train" in splits else ds[splits[0]]
    cols   = data.column_names
    print(f"   Colonnes : {cols} | Exemples : {len(data)}", flush=True)

    # Le dataset raoulbia/3gpp-5g-nr-qa a 3 colonnes : instruction, input, output
    # - instruction : la question
    # - input       : contexte additionnel (souvent vide)
    # - output      : la réponse (peut contenir <|im_end|> à nettoyer)

    questions = []
    for item in data:
        # Question depuis 'instruction' en priorité
        question = ""
        for col in ["instruction", "question", "Question", "prompt"]:
            if col in cols and item.get(col):
                question = str(item[col]).strip()
                break

        # Réponse depuis 'output' en priorité
        answer = ""
        for col in ["output", "answer", "Answer", "response", "label"]:
            if col in cols and item.get(col):
                answer = str(item[col]).strip()
                answer = answer.replace("<|im_end|>", "").strip()
                answer = answer.replace("<|endoftext|>", "").strip()
                break

        # Contexte/explication depuis 'input' (si non vide)
        explanation = ""
        for col in ["input", "explanation", "Explanation", "context", "Context"]:
            if col in cols and item.get(col):
                val = str(item[col]).strip()
                if val:
                    explanation = val[:EXPL_LIMIT]
                    break

        # Pas d'options dans ce dataset (format Q&A libre)
        options = {}
        option_keys = sorted([c for c in cols if c.lower().startswith("option")])
        if option_keys:
            options = {k: item[k] for k in option_keys if item.get(k)}

        if not question or not answer:
            continue

        questions.append({
            "question":    question,
            "options":     options,
            "answer":      answer,
            "explanation": explanation,
            "category":    str(item.get("category", ""))
        })

    print(f"   {len(questions)} questions chargées", flush=True)
    return questions


def load_hf_training_questions(max_samples=MAX_TRAIN_SAMPLES):
    """
    Charge raoulbia/3gpp-5g-nr-qa pour générer le dataset RAFT.
    Retente un téléchargement forcé si 0 question est trouvée
    (cache HF local corrompu/incomplet).
    """
    questions = _load_hf_questions(force_redownload=False)

    if len(questions) == 0:
        print("⚠️  0 question chargée avec le cache local — "
              "tentative de re-téléchargement forcé du dataset...", flush=True)
        questions = _load_hf_questions(force_redownload=True)

    if len(questions) > max_samples:
        random.seed(42)
        questions = random.sample(questions, max_samples)
        print(f"   Sous-échantillon : {max_samples} questions", flush=True)

    return questions


# ============================================================
#  CHARGEMENT QUESTIONS TELEQNA (évaluation)
# ============================================================
def load_teleqna_questions(filepath, max_questions=100):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = []
    for q_id, q_data in list(data.items())[:max_questions]:
        options    = {k: v for k, v in q_data.items() if k.startswith("option")}
        raw_answer = q_data.get("answer", None)
        answer     = raw_answer.split(":")[0].strip() if raw_answer else None
        questions.append({
            "id":       q_id,
            "question": q_data.get("question", ""),
            "options":  options,
            "answer":   answer,
            "category": q_data.get("category", "")
        })
    print(f"  {len(questions)} questions TeleQnA chargées", flush=True)
    return questions


# ============================================================
#  RETRIEVAL BATCH ChromaDB
# ============================================================
def batch_retrieve_contexts(questions_text, collection, emb_model, top_k=3):
    print(f"  Encoding {len(questions_text)} questions en batch...", flush=True)
    embeddings = emb_model.encode(
        questions_text,
        batch_size=EMB_BATCH_SIZE,
        device=DEVICE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    results  = collection.query(
        query_embeddings=embeddings.tolist(),
        n_results=top_k,
        include=["documents"]
    )
    contexts = []
    for chunks in results["documents"]:
        short_chunks = [c[:CHUNK_LIMIT] for c in chunks]
        contexts.append("\n---\n".join(short_chunks))
    return contexts


def retrieve_context_single(question_text, collection, emb_model, top_k=3):
    emb = emb_model.encode(
        [question_text],
        device=DEVICE,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    results = collection.query(
        query_embeddings=emb.tolist(),
        n_results=top_k,
        include=["documents"]
    )
    chunks = [c[:CHUNK_LIMIT] for c in results["documents"][0]]
    return "\n---\n".join(chunks)


# ============================================================
#  SIGNATURE DU DATASET RAFT (utilisée + chaînée plus bas)
# ============================================================
def get_raft_dataset_signature(num_questions):
    return compute_signature({
        "dataset":           "raoulbia/3gpp-5g-nr-qa",
        "num_questions":     num_questions,
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "top_k":             TOP_K,
        "chunk_limit":       CHUNK_LIMIT,
        "expl_limit":        EXPL_LIMIT
    })


# ============================================================
#  GÉNÉRATION DATASET RAFT
# ============================================================
def generate_raft_dataset(questions, collection, emb_model, output_file):
    signature = get_raft_dataset_signature(len(questions))

    cached = load_json_cache(output_file, signature, force=FORCE_REFRESH)
    if cached:
        samples = cached["samples"]
        print(f"  Dataset RAFT déjà généré (cache valide) : {output_file}", flush=True)
        print(f"  {len(samples)} exemples chargés", flush=True)
        return samples

    if os.path.exists(output_file):
        print(f"  ⚠️  raft_dataset.json existant mais périmé "
              f"(paramètres différents) — régénération", flush=True)

    print(f"\n  Génération dataset RAFT ({len(questions)} exemples)...", flush=True)

    questions_text = [q["question"] for q in questions]
    contexts       = batch_retrieve_contexts(questions_text, collection, emb_model, top_k=TOP_K)

    samples = []
    for q, context in zip(questions, contexts):
        options_block = ""
        if q["options"]:
            options_text  = "\n".join([f"{k}: {v}" for k, v in q["options"].items()])
            options_block = f"\n{options_text}"
        explanation_block = f" {q['explanation']}" if q["explanation"] else ""
        text = (
            f"### Instruction:\n3GPP 5G NR expert. Use context to answer.\n\n"
            f"Context: {context}\n\n"
            f"Q: {q['question']}{options_block}\n\n"
            f"### Response:\n{q['answer']}.{explanation_block}"
        )
        samples.append({
            "text":     text,
            "question": q["question"],
            "answer":   q["answer"],
            "category": q["category"]
        })

    save_json_cache(output_file, {"samples": samples}, signature)
    print(f"  ✅ {len(samples)} exemples RAFT générés → {output_file}", flush=True)
    return samples


# ============================================================
#  CHECKPOINT
# ============================================================
def find_last_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None
    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    last = os.path.join(output_dir, checkpoints[-1])
    print(f"  Checkpoint trouvé : {last}", flush=True)
    return last


# ============================================================
#  RAFT FINE-TUNING
#  Qwen  : RAFT par-dessus raoulbia/Qwen2.5-7B-3GPP-NR
#  LLaMA/Mistral : RAFT depuis base HuggingFace
# ============================================================
def raft_finetune(model_short, hf_base_name, samples, dataset_signature):
    output_dir   = os.path.join(RAFT_DIR, f"{model_short}_raft")
    done_file    = os.path.join(output_dir, "RAFT_DONE.json")
    legacy_file  = os.path.join(output_dir, "RAFT_DONE.txt")
    run_sig_file = os.path.join(output_dir, "RAFT_RUN_SIGNATURE.json")

    grad_accum = GRAD_ACCUM_GPU if USE_4BIT else GRAD_ACCUM_CPU

    # La signature chaîne celle du dataset RAFT : si le dataset change,
    # cet entraînement se réinvalide automatiquement.
    # NOTE : on utilise model_short (pas hf_base_name) car le CHEMIN du modèle
    # peut changer (ex: URL HuggingFace -> dossier local) sans que ce soit
    # une vraie config différente -> ça ne doit PAS invalider un entraînement.
    signature = compute_signature({
        "model":             model_short,
        "dataset_signature": dataset_signature,
        "num_samples":       len(samples),
        "lora_r":            8,
        "lora_alpha":        16,
        "lora_dropout":      0.05,
        "target_modules":    ["q_proj", "v_proj"],
        "epochs":            1,
        "lr":                2e-4,
        "max_seq_len":       MAX_SEQ_LEN,
        "grad_accum":        grad_accum,
        "device":            DEVICE
    })

    # Compat : un ancien marqueur .txt vaut "déjà fait" une fois, puis migre vers .json
    migrate_legacy_marker(legacy_file, done_file, signature, {
        "model": hf_base_name, "migrated_from_txt": True
    })

    if is_marker_valid(done_file, signature, force=FORCE_RETRAIN):
        print(f"\n  ✅ {model_short} déjà RAFT fine-tuné (config identique) !", flush=True)
        return output_dir

    if os.path.exists(output_dir):
        existing_sig = (read_marker(done_file) or read_marker(run_sig_file) or {}).get("signature")
        if FORCE_RETRAIN:
            archive_stale_dir(output_dir, reason="FORCE_RETRAIN demandé")
        elif read_marker(done_file) and existing_sig and existing_sig != signature:
            # Seul un entraînement TERMINÉ (RAFT_DONE.json présent) avec une
            # config réellement différente est archivé.
            archive_stale_dir(output_dir, reason="dataset RAFT ou hyperparamètres différents")
        elif existing_sig and existing_sig != signature:
            # Entraînement INTERROMPU (pas de RAFT_DONE.json) : on ne supprime
            # JAMAIS un checkpoint en cours, même si la signature a légèrement
            # changé (ex: chemin du modèle). On reprend simplement depuis là.
            print(f"  ℹ️  Signature légèrement différente mais entraînement non "
                  f"terminé — reprise du checkpoint existant sans archivage", flush=True)
        # si existing_sig == signature mais done_file absent -> entraînement
        # interrompu avec la MÊME config -> reprise normale depuis checkpoint.

    os.makedirs(output_dir, exist_ok=True)
    write_marker(run_sig_file, signature, {"status": "in_progress", "model": hf_base_name})

    last_checkpoint = find_last_checkpoint(output_dir)

    print(f"\n{'='*60}", flush=True)
    print(f"  RAFT Fine-tuning : {model_short} ({len(samples)} exemples)", flush=True)
    print(f"  Base model       : {hf_base_name}", flush=True)
    print(f"  Device           : {DEVICE.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    # Certains tokenizers (ex: raoulbia/Qwen2.5-7B) ont un tokenizer.json
    # incompatible avec la version installée de `tokenizers` → fallback slow
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_base_name, trust_remote_code=True)
    except Exception as e_fast:
        print(f"  ⚠️  Tokenizer rapide échoué ({e_fast.__class__.__name__})"
              f" — fallback use_fast=False", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            hf_base_name, trust_remote_code=True, use_fast=False
        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"   # FIX : garder la fin du prompt si troncature

    # rope_scaling déjà normalisé par le monkey-patch LlamaConfig au démarrage
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_base_name,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        optim = "adamw_8bit"
        fp16  = True
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hf_base_name,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        optim = "adamw_torch"
        fp16  = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Paramètres entraînables : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)", flush=True)

    dataset = Dataset.from_list(samples)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        fp16=fp16,
        optim=optim,
        logging_steps=5,
        save_steps=10,        # checkpoint toutes les 10 steps (~15 min max perdu si coupure)
        eval_strategy="no",
        save_total_limit=5,   # garde les 5 derniers checkpoints
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=0,
        group_by_length=True,
        resume_from_checkpoint=last_checkpoint
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        tokenizer=tokenizer,
        args=training_args,
        packing=True
    )

    print(f"\n  🔥 Début RAFT fine-tuning ({DEVICE.upper()})...", flush=True)
    start = time.time()
    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    elapsed = round((time.time() - start) / 60, 1)
    write_marker(done_file, signature, {
        "elapsed_min": elapsed,
        "model":       hf_base_name,
        "dataset":     "raoulbia/3gpp-5g-nr-qa",
        "samples":     len(samples),
        "device":      DEVICE
    })

    print(f"\n  ✅ RAFT {model_short} terminé en {elapsed} minutes !", flush=True)

    del model, tokenizer
    gc.collect()
    if USE_4BIT:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(2)
    print(f"  Mémoire libérée", flush=True)
    return output_dir


# ============================================================
#  ÉVALUATION RAFT
# ============================================================
def extract_option(response_text):
    response_lower = response_text.lower().strip()
    for i in range(1, 6):
        if f"option {i}" in response_lower:
            return f"option {i}"
    for i in range(1, 6):
        if response_lower.startswith(str(i)):
            return f"option {i}"
    return response_text


def compute_accuracy(results):
    correct, total = 0, 0
    for r in results:
        if r["true_answer"] and r["predicted_answer"]:
            total += 1
            true = r["true_answer"].lower().strip()
            pred = extract_option(r["predicted_answer"]).lower().strip()
            if true == pred or true in pred or pred in true:
                correct += 1
    return (correct / total if total > 0 else 0), correct, total


def evaluate_raft_model(model_short, hf_base_name, questions, collection, emb_model):
    cache_file  = os.path.join(RESULTS_DIR, f"raft_{model_short}.json")
    adapter_dir = os.path.join(RAFT_DIR, f"{model_short}_raft")

    done_marker = read_marker(os.path.join(adapter_dir, "RAFT_DONE.json"))
    if not done_marker:
        if os.path.exists(os.path.join(adapter_dir, "RAFT_DONE.txt")):
            done_marker = {"signature": "legacy-pre-migration"}
        else:
            print(f"  ⚠️  {model_short} RAFT non terminé, skip", flush=True)
            return None

    # La signature chaîne celle de l'entraînement RAFT : si le modèle est
    # ré-entraîné, ce cache d'éval se réinvalide automatiquement.
    eval_signature = compute_signature({
        "model":           model_short,
        "raft_signature":  done_marker.get("signature"),
        "top_k":           TOP_K,
        "max_seq_len":     MAX_SEQ_LEN,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "max_new_tokens":  EVAL_MAX_NEW_TOKENS,
        "prompt_version":  RAFT_PROMPT_VERSION,
        "num_questions":   len(questions)
    })

    cached = load_json_cache(cache_file, eval_signature, force=FORCE_REFRESH)
    if cached:
        acc, correct, total = compute_accuracy(cached["results"])
        print(f"\n  Cache RAFT valide pour {model_short}", flush=True)
        print(f"  {model_short} RAFT : {correct}/{total} = {acc:.1%}", flush=True)
        return cached
    elif os.path.exists(cache_file):
        print(f"\n  ⚠️  Cache RAFT périmé pour {model_short} (modèle ré-entraîné ou "
              f"paramètres d'éval changés) — réévaluation", flush=True)

    print(f"\n  Chargement modèle RAFT : {model_short}", flush=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    except Exception as e_fast:
        print(f"  ⚠️  Tokenizer rapide échoué ({e_fast.__class__.__name__})"
              f" — fallback use_fast=False", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_dir, trust_remote_code=True, use_fast=False
        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"   # FIX : la troncature à droite coupait la
                                          # question / les options / "### Response:"
                                          # quand Context + Q dépassaient MAX_SEQ_LEN.

    gc.collect()
    if USE_4BIT:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(3)

    # rope_scaling déjà normalisé par le monkey-patch LlamaConfig au démarrage
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            hf_base_name,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            hf_base_name,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    # Batch encode tous les contextes
    print(f"  Pré-calcul des contextes RAG...", flush=True)
    questions_text = [q["question"] for q in questions]
    contexts       = batch_retrieve_contexts(questions_text, collection, emb_model, top_k=TOP_K)

    # Construire prompts
    prompts = []
    for q, context in zip(questions, contexts):
        options_block = ""
        if q.get("options"):
            options_text  = "\n".join([f"{k}: {v}" for k, v in q["options"].items()])
            options_block = f"\n{options_text}\n\nRespond with ONLY the option number (e.g., \"option 1\")."
        prompt = (
            f"### Instruction:\n3GPP 5G NR expert. Use context to answer.\n\n"
            f"Context: {context}\n\n"
            f"Q: {q['question']}{options_block}\n\n"
            f"### Response:"
        )
        prompts.append(prompt)

    batch_size_eval = EVAL_BATCH_SIZE if USE_4BIT else 1
    print(f"  Inférence batch={batch_size_eval} sur {len(questions)} questions...", flush=True)

    model_results = []
    start_time    = time.time()

    for batch_start in tqdm(range(0, len(prompts), batch_size_eval), desc=f"RAFT {model_short}"):
        batch_prompts = prompts[batch_start: batch_start + batch_size_eval]
        batch_qs      = questions[batch_start: batch_start + batch_size_eval]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        input_len = inputs["input_ids"].shape[1]
        for q, out in zip(batch_qs, outputs):
            generated = out[input_len:]
            response  = tokenizer.decode(generated, skip_special_tokens=True).strip()
            model_results.append({
                "id":               q["id"],
                "question":         q["question"],
                "true_answer":      q["answer"],
                "predicted_answer": response,
                "extracted_option": extract_option(response),
                "category":         q["category"]
            })

    acc, correct, total = compute_accuracy(model_results)
    total_time = time.time() - start_time

    result_data = {
        "model":        model_short,
        "pipeline":     "raft",
        "accuracy":     acc,
        "correct":      correct,
        "total":        total,
        "total_time_s": round(total_time, 2),
        "results":      model_results
    }

    save_json_cache(cache_file, result_data, eval_signature)

    print(f"  ✅ {model_short} RAFT : {correct}/{total} = {acc:.1%} ({total_time:.0f}s)", flush=True)

    del model, base_model, tokenizer
    gc.collect()
    if USE_4BIT:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(2)
    return result_data


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    print("\n" + "="*60, flush=True)
    print("  3GPP RAFT - Retrieval Augmented Fine-Tuning", flush=True)
    print(f"  Train dataset : raoulbia/3gpp-5g-nr-qa", flush=True)
    print(f"  Eval dataset  : TeleQnA (ancien dataset)", flush=True)
    print(f"  Samples RAFT  : {MAX_TRAIN_SAMPLES} | SeqLen : {MAX_SEQ_LEN}", flush=True)
    print(f"  Device        : {DEVICE.upper()}", flush=True)
    print("="*60, flush=True)

    if DEVICE == "cpu":
        print("\n⚠️  Mode CPU — entraînement et inférence lents mais fonctionnels", flush=True)

    # ChromaDB
    print(f"\nConnexion ChromaDB : {CHROMA_DIR}", flush=True)
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLL_NAME)
    print(f"Collection : {collection.count()} chunks", flush=True)

    # Embedding model
    print(f"\nChargement embedding : {EMB_MODEL}", flush=True)
    emb_model = SentenceTransformer(EMB_MODEL, device=DEVICE)
    print(f"Embedding chargé sur {DEVICE.upper()}", flush=True)

    # ── ETAPE 1 : Dataset RAFT depuis HuggingFace ───────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  ÉTAPE 1 : Génération dataset RAFT", flush=True)
    print(f"{'='*60}", flush=True)

    hf_questions = load_hf_training_questions(max_samples=MAX_TRAIN_SAMPLES)
    raft_samples = generate_raft_dataset(hf_questions, collection, emb_model, RAFT_DATA)
    dataset_signature = get_raft_dataset_signature(len(hf_questions))
    avg_len = sum(len(s["text"]) for s in raft_samples) / len(raft_samples)
    print(f"  Longueur moyenne exemple : {avg_len:.0f} chars", flush=True)

    # ── ETAPE 2 : RAFT Fine-tuning des 3 modèles ────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  ÉTAPE 2 : RAFT Fine-tuning", flush=True)
    print(f"{'='*60}", flush=True)

    for model_short, hf_base_name in HF_BASE_MODELS.items():
        raft_finetune(model_short, hf_base_name, raft_samples, dataset_signature)

    # ── ETAPE 3 : Évaluation sur TeleQnA ────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  ÉTAPE 3 : Évaluation RAFT sur TeleQnA", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"\nChargement questions TeleQnA : {TEST_FILE}", flush=True)
    test_questions = load_teleqna_questions(TEST_FILE, max_questions=100)

    raft_results = {}
    for model_short, hf_base_name in HF_BASE_MODELS.items():
        data = evaluate_raft_model(
            model_short, hf_base_name,
            test_questions, collection, emb_model
        )
        if data:
            raft_results[model_short] = data

    # ── Tableau final ────────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print(f"  RÉSULTATS RAFT", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Modèle':<20} {'Accuracy':<12} {'Correct'}", flush=True)
    print(f"{'-'*50}", flush=True)

    summary = []
    for model_short in ["qwen25_7b", "llama31_8b", "mistral_7b"]:
        if model_short in raft_results:
            acc     = raft_results[model_short]["accuracy"]
            correct = raft_results[model_short]["correct"]
            total   = raft_results[model_short]["total"]
            print(f"{model_short:<20} {acc:<12.1%} {correct}/{total}", flush=True)
            summary.append({"model": model_short, "pipeline": "raft", "accuracy": acc,
                            "correct": correct, "total": total})
        else:
            print(f"{model_short:<20} N/A", flush=True)

    summary_file = os.path.join(RESULTS_DIR, "raft_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRésumé sauvegardé : {summary_file}", flush=True)
    print("Prochaine étape   : Rapport final", flush=True)
    print("="*70, flush=True)
