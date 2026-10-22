# ============================================================
#  07b_raft_rag.py
#  RAFT + RAG — Évaluation uniquement (pas de ré-entraînement)
#
#  Différence avec 07_raft_fast.py :
#  - RAFT seul     : entraîné avec contexte, inférence SANS RAG
#  - RAFT + RAG    : entraîné avec contexte, inférence AVEC RAG
#                    (contexte ChromaDB frais à chaque question)
#
#  Les modèles RAFT sont déjà entraînés → pas de re-training
#  On charge les adapters RAFT + on ajoute RAG à l'inférence
#
#  Modèles RAFT locaux :
#  - ./raft_models/qwen25_7b_raft/
#  - ./raft_models/llama31_8b_raft/
#  - ./raft_models/mistral_7b_raft/
# ============================================================

import os
import sys
import gc
import json
import yaml
import time
import torch
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ── MONKEY-PATCH LlamaConfig + LlamaAttention ────────────────
try:
    from transformers.models.llama.configuration_llama import LlamaConfig as _LlamaConfig

    def _patched_rope_scaling_validation(self):
        if self.rope_scaling is None:
            return
        if not isinstance(self.rope_scaling, dict):
            raise ValueError("`rope_scaling` doit être un dict.")
        rs = self.rope_scaling
        if "rope_type" in rs and "type" not in rs:
            rs["type"] = rs.pop("rope_type")

    _LlamaConfig._rope_scaling_validation = _patched_rope_scaling_validation
    print("STEP 1.5a : LlamaConfig rope_scaling patché ✅", flush=True)
except Exception as _e:
    print(f"STEP 1.5a : Patch LlamaConfig ignoré ({_e.__class__.__name__})", flush=True)

# Patch _init_rope pour accepter rope_type "llama3"
try:
    from transformers.models.llama import modeling_llama as _llama_modeling

    _original_init_rope = _llama_modeling.LlamaRotaryEmbedding.__init__ \
        if hasattr(_llama_modeling, "LlamaRotaryEmbedding") else None

    def _patched_init_rope(self):
        """Remplace rope_type llama3 par linear pour compatibilité."""
        if hasattr(self.config, "rope_scaling") and isinstance(
            self.config.rope_scaling, dict
        ):
            rs = self.config.rope_scaling
            if rs.get("type") == "llama3" or rs.get("rope_type") == "llama3":
                rs["type"] = "linear"
                rs.pop("rope_type", None)
        # Appeler l'init original
        if _original_init_rope:
            _original_init_rope(self)

    # Patch plus simple : neutraliser la validation dans LlamaAttention
    for _cls_name in ["LlamaAttention", "LlamaFlashAttention2", "LlamaSdpaAttention"]:
        _cls = getattr(_llama_modeling, _cls_name, None)
        if _cls is not None:
            _orig = _cls._init_rope
            def _make_patched(_orig_fn):
                def _patched(self):
                    try:
                        _orig_fn(self)
                    except ValueError as _ve:
                        if "Unknown RoPE scaling type" in str(_ve):
                            # Forcer linear
                            self.config.rope_scaling = {
                                "type": "linear",
                                "factor": float(
                                    self.config.rope_scaling.get("factor", 1.0)
                                )
                            }
                            _orig_fn(self)
                        else:
                            raise
                return _patched
            _cls._init_rope = _make_patched(_orig)

    print("STEP 1.5b : LlamaAttention _init_rope patché ✅", flush=True)
except Exception as _e2:
    print(f"STEP 1.5b : Patch _init_rope ignoré ({_e2.__class__.__name__}: {_e2})", flush=True)

print("STEP 1 : Imports OK", flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_utils import (
    compute_signature, write_marker, read_marker,
    load_json_cache, save_json_cache,
    FORCE_REFRESH
)

with open(os.environ.get("RAG_CONFIG", "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

CHROMA_DIR  = config["paths"]["chroma_db"]
COLL_NAME   = config["rag_baseline"]["collection_name"]
EMB_MODEL   = config["rag_baseline"]["embedding_model"]
TEST_FILE   = config["paths"]["test_teleqna"]
MODELS_DIR  = config["paths"]["models_dir"]
RESULTS_DIR = config["paths"]["results_dir"]
RAFT_DIR    = "./raft_models"

# ── HYPER-PARAMÈTRES ────────────────────────────────────────
TOP_K              = 3         # chunks RAG récupérés à l'inférence
CHUNK_LIMIT        = 300       # plus long qu'en training (on a de la place)
MAX_SEQ_LEN        = 512       # plus long qu'en RAFT seul (512 vs 256)
EVAL_BATCH_SIZE    = 1         # 1 pour sécurité mémoire
EVAL_MAX_NEW_TOKENS = 15
EMB_BATCH_SIZE     = 64
PIPELINE_VERSION   = "raft_rag_v1"
# ────────────────────────────────────────────────────────────

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_4BIT = torch.cuda.is_available()

print(f"STEP 2 : Config OK — DEVICE={DEVICE.upper()}", flush=True)

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── MODÈLES RAFT (déjà entraînés) ───────────────────────────
# Base pour charger les adapters LoRA RAFT
HF_BASE_MODELS = {
    "qwen25_7b":  "raoulbia/Qwen2.5-7B-3GPP-NR",
    "llama31_8b": "./models/llama31_8b_finetuned",
    "mistral_7b": "./models/mistral_7b_finetuned"
}


# ============================================================
#  UTILITAIRES
# ============================================================
def extract_option(response_text):
    r = response_text.lower().strip()
    for i in range(1, 6):
        if f"option {i}" in r:
            return f"option {i}"
    for i in range(1, 6):
        if r.startswith(str(i)):
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
def batch_retrieve_contexts(questions_text, collection, emb_model, top_k=TOP_K):
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


# ============================================================
#  CHARGEMENT MODÈLE RAFT (adapter LoRA par-dessus la base)
# ============================================================
def load_raft_model(model_short, hf_base_name):
    """
    Charge le modèle RAFT fine-tuné.
    - Qwen  : base HuggingFace + adapter LoRA RAFT
    - LLaMA/Mistral : modèle local complet + adapter LoRA RAFT
    """
    adapter_dir = os.path.join(RAFT_DIR, f"{model_short}_raft")
    done_file   = os.path.join(adapter_dir, "RAFT_DONE.json")
    legacy_done = os.path.join(adapter_dir, "RAFT_DONE.txt")

    if not os.path.exists(done_file) and not os.path.exists(legacy_done):
        print(f"  ⚠️  {model_short} RAFT non trouvé dans {adapter_dir}", flush=True)
        print(f"      Lancez d'abord : python 07_raft_fast.py", flush=True)
        return None, None, None

    print(f"\n  Chargement RAFT {model_short}...", flush=True)
    print(f"  Base    : {hf_base_name}", flush=True)
    print(f"  Adapter : {adapter_dir}", flush=True)

    # Tokenizer depuis l'adapter
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_dir, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_dir, trust_remote_code=True, use_fast=False)

    tokenizer.pad_token       = tokenizer.eos_token
    tokenizer.padding_side    = "left"
    tokenizer.truncation_side = "left"

    gc.collect()
    if USE_4BIT:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(2)

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
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            hf_base_name,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True
        )

    # Charger l'adapter RAFT par-dessus la base
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    print(f"  ✅ Modèle RAFT {model_short} chargé", flush=True)
    return model, tokenizer, base_model


# ============================================================
#  ÉVALUATION RAFT + RAG
# ============================================================
def evaluate_raft_rag(model_short, hf_base_name, questions, collection, emb_model):
    """
    Évalue le modèle RAFT avec RAG à l'inférence.
    Différence clé vs RAFT seul :
    - RAFT seul  : prompt sans contexte frais (ou contexte figé du training)
    - RAFT + RAG : contexte ChromaDB récupéré à chaque question en temps réel
    """
    cache_file  = os.path.join(RESULTS_DIR, f"raft_rag_{model_short}.json")
    adapter_dir = os.path.join(RAFT_DIR, f"{model_short}_raft")

    # Signature pour invalider le cache si les params changent
    done_marker  = read_marker(os.path.join(adapter_dir, "RAFT_DONE.json")) or {}
    eval_sig = compute_signature({
        "pipeline":        PIPELINE_VERSION,
        "model":           model_short,
        "raft_signature":  done_marker.get("signature", "legacy"),
        "top_k":           TOP_K,
        "chunk_limit":     CHUNK_LIMIT,
        "max_seq_len":     MAX_SEQ_LEN,
        "max_new_tokens":  EVAL_MAX_NEW_TOKENS,
        "num_questions":   len(questions)
    })

    # Cache valide ?
    cached = load_json_cache(cache_file, eval_sig, force=FORCE_REFRESH)
    if cached:
        acc, correct, total = compute_accuracy(cached["results"])
        print(f"\n  📂 Cache RAFT+RAG valide pour {model_short}", flush=True)
        print(f"  {model_short} RAFT+RAG : {correct}/{total} = {acc:.1%}", flush=True)
        return cached

    # Charger le modèle RAFT
    model, tokenizer, base_model = load_raft_model(model_short, hf_base_name)
    if model is None:
        return None

    # ── Récupérer les contextes RAG pour toutes les questions ────────
    print(f"\n  Pré-calcul contextes RAG (ChromaDB)...", flush=True)
    questions_text = [q["question"] for q in questions]
    contexts       = batch_retrieve_contexts(
        questions_text, collection, emb_model, top_k=TOP_K
    )

    # ── Construire les prompts RAFT+RAG ─────────────────────────────
    # Format identique à l'entraînement RAFT mais avec contexte FRAIS
    prompts = []
    for q, context in zip(questions, contexts):
        options_text  = "\n".join([f"{k}: {v}" for k, v in q["options"].items()])
        prompt = (
            f"### Instruction:\n"
            f"3GPP 5G NR expert. Use the context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Q: {q['question']}\n"
            f"{options_text}\n\n"
            f"Respond with ONLY the option number (e.g., \"option 1\").\n\n"
            f"### Response:"
        )
        prompts.append(prompt)

    # ── Inférence ────────────────────────────────────────────────────
    print(f"  Inférence RAFT+RAG sur {len(questions)} questions...", flush=True)
    model_results = []
    start_time    = time.time()

    for batch_start in tqdm(
        range(0, len(prompts), EVAL_BATCH_SIZE),
        desc=f"RAFT+RAG {model_short}"
    ):
        batch_prompts = prompts[batch_start: batch_start + EVAL_BATCH_SIZE]
        batch_qs      = questions[batch_start: batch_start + EVAL_BATCH_SIZE]

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
                temperature=1.0,   # neutre avec do_sample=False
                top_p=1.0,         # neutre avec do_sample=False
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
        "pipeline":     "raft_rag",
        "accuracy":     acc,
        "correct":      correct,
        "total":        total,
        "total_time_s": round(total_time, 2),
        "results":      model_results
    }

    save_json_cache(cache_file, result_data, eval_sig)
    print(f"  ✅ {model_short} RAFT+RAG : {correct}/{total} = {acc:.1%} ({total_time:.0f}s)", flush=True)

    # Libérer la mémoire avant le modèle suivant
    del model, base_model, tokenizer
    gc.collect()
    if USE_4BIT:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(2)
    print(f"  Mémoire libérée", flush=True)

    return result_data


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    print("\n" + "="*60, flush=True)
    print("  3GPP RAFT + RAG — Évaluation", flush=True)
    print(f"  Modèles RAFT  : déjà entraînés dans {RAFT_DIR}", flush=True)
    print(f"  RAG           : ChromaDB (contexte frais à l'inférence)", flush=True)
    print(f"  Eval dataset  : TeleQnA", flush=True)
    print(f"  Device        : {DEVICE.upper()}", flush=True)
    print("="*60, flush=True)

    # ── ChromaDB ────────────────────────────────────────────────────
    print(f"\nConnexion ChromaDB : {CHROMA_DIR}", flush=True)
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLL_NAME)
    print(f"Collection : {collection.count()} chunks", flush=True)

    # ── Embedding model ─────────────────────────────────────────────
    print(f"\nChargement embedding : {EMB_MODEL}", flush=True)
    emb_model = SentenceTransformer(EMB_MODEL, device=DEVICE)
    print(f"Embedding chargé sur {DEVICE.upper()}", flush=True)

    # ── Questions TeleQnA ───────────────────────────────────────────
    print(f"\nChargement questions TeleQnA : {TEST_FILE}", flush=True)
    questions = load_teleqna_questions(TEST_FILE, max_questions=100)

    # ── Évaluation RAFT+RAG des 3 modèles ───────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  ÉVALUATION RAFT + RAG", flush=True)
    print(f"{'='*60}", flush=True)

    raft_rag_results = {}
    for model_short, hf_base_name in HF_BASE_MODELS.items():
        data = evaluate_raft_rag(
            model_short, hf_base_name,
            questions, collection, emb_model
        )
        if data:
            raft_rag_results[model_short] = data

    # ── Tableau final comparatif ─────────────────────────────────────
    print(f"\n{'='*80}", flush=True)
    print(f"  TABLEAU COMPARATIF — RAFT vs RAFT+RAG", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  {'Modèle':<20} {'RAFT seul':<14} {'RAFT + RAG':<14} {'Δ'}", flush=True)
    print(f"  {'-'*60}", flush=True)

    # Charger résultats RAFT seul pour comparaison
    raft_known = {
        "qwen25_7b":  0.18,
        "llama31_8b": 0.43,
        "mistral_7b": 0.54
    }
    # Essayer de charger depuis les fichiers résultats
    for ms in ["qwen25_7b", "llama31_8b", "mistral_7b"]:
        raft_file = os.path.join(RESULTS_DIR, f"raft_{ms}.json")
        if os.path.exists(raft_file):
            try:
                with open(raft_file, "r") as f:
                    d = json.load(f)
                raft_known[ms] = d.get("accuracy", raft_known[ms])
            except Exception:
                pass

    summary = []
    for model_short in ["qwen25_7b", "llama31_8b", "mistral_7b"]:
        raft_acc     = raft_known.get(model_short, 0)
        raft_rag_acc = raft_rag_results.get(model_short, {}).get("accuracy", None)

        if raft_rag_acc is not None:
            delta    = raft_rag_acc - raft_acc
            delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
            correct  = raft_rag_results[model_short]["correct"]
            total    = raft_rag_results[model_short]["total"]
            print(
                f"  {model_short:<20} {raft_acc:<14.1%} "
                f"{raft_rag_acc:<14.1%} {delta_str} ({correct}/{total})",
                flush=True
            )
            summary.append({
                "model":        model_short,
                "pipeline":     "raft_rag",
                "accuracy":     raft_rag_acc,
                "raft_accuracy": raft_acc,
                "delta":        round(raft_rag_acc - raft_acc, 4),
                "correct":      correct,
                "total":        total
            })
        else:
            print(f"  {model_short:<20} {raft_acc:<14.1%} N/A", flush=True)

    summary_file = os.path.join(RESULTS_DIR, "raft_rag_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Résumé sauvegardé : {summary_file}", flush=True)
    print(f"   Étape suivante   : python 09_agents_eval.py", flush=True)
    print("="*80, flush=True)
