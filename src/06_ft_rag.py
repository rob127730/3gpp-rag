# ============================================================
#  06_ft_rag.py
#  Fine-tuning + RAG combiné - 3 modèles
#  - Qwen2.5-7B  : chargé depuis raoulbia/Qwen2.5-7B-3GPP-NR
#  - LLaMA 3.1-8B: modèle fine-tuné (adapter LoRA)
#  - Mistral-7B  : modèle fine-tuné (adapter LoRA)
#  Évaluation sur TeleQnA (ancien dataset)
#  Compatible CPU et GPU
#
#  CORRECTIFS (cf. cache_utils.py) :
#  - Le cache de résultats par modèle (ft_rag_{model}.json) est
#    maintenant validé par signature : il inclut TOP_K, le
#    nombre de questions, la version du prompt ET la signature
#    d'entraînement du modèle. Si le modèle a été ré-entraîné,
#    ou si TOP_K change, le cache se réinvalide automatiquement
#    au lieu d'être silencieusement réutilisé.
#  - truncation_side="left" sur le tokenizer : si le contexte +
#    la question dépassent max_length, on coupe le DÉBUT du
#    contexte plutôt que la fin du prompt (qui contient la
#    question et "### Response:").
#  - FORCE_REFRESH=1 force une réévaluation complète.
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

print("STEP 1 : Imports OK", flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_utils import (
    compute_signature, read_marker, load_json_cache, save_json_cache, FORCE_REFRESH
)

with open(os.environ.get("RAG_CONFIG", "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

CHROMA_DIR  = config["paths"]["chroma_db"]
COLL_NAME   = config["rag_baseline"]["collection_name"]
EMB_MODEL   = config["rag_baseline"]["embedding_model"]
TOP_K       = config["rag_baseline"]["top_k"]
TEST_FILE   = config["paths"]["test_teleqna"]
RESULTS_DIR = config["paths"]["results_dir"]
MODELS_DIR  = config["paths"]["models_dir"]

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_4BIT = torch.cuda.is_available()

# Bump cette chaîne dès que le prompt ou la logique d'inférence change,
# pour forcer l'invalidation des caches existants.
PROMPT_VERSION = "v2-left-trunc"
MAX_NEW_TOKENS = 50

print(f"STEP 2 : Config OK — DEVICE={DEVICE.upper()}", flush=True)
if FORCE_REFRESH:
    print("⚠️  FORCE_REFRESH=1 — tous les résultats seront recalculés", flush=True)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ============================================================
#  MODÈLES
#  Qwen  -> pré-entraîné 3GPP (chargé directement, pas d'adapter)
#  LLaMA -> base + adapter LoRA (fine-tuné par nous)
#  Mistral -> base + adapter LoRA (fine-tuné par nous)
# ============================================================
HF_BASE_MODELS = {
    "qwen25_7b":  "raoulbia/Qwen2.5-7B-3GPP-NR",           # pré-entraîné 3GPP
    "llama31_8b": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3"
}

# Qwen n'a pas d'adapter LoRA, il est chargé directement
HAS_LORA_ADAPTER = {
    "qwen25_7b":  False,   # chargé directement depuis raoulbia
    "llama31_8b": True,    # adapter LoRA dans MODELS_DIR
    "mistral_7b": True     # adapter LoRA dans MODELS_DIR
}


# ============================================================
#  CHARGEMENT QUESTIONS TELEQNA (évaluation)
# ============================================================
def load_teleqna_questions(filepath, max_questions=100):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = []
    for q_id, q_data in list(data.items())[:max_questions]:
        options = {k: v for k, v in q_data.items() if k.startswith("option")}
        raw_answer = q_data.get("answer", None)
        answer = raw_answer.split(":")[0].strip() if raw_answer else None
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
#  RAG : récupération contexte ChromaDB
# ============================================================
def retrieve_context(question_text, collection, emb_model, top_k=5):
    q_embedding = emb_model.encode(
        [question_text],
        device=DEVICE,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    results = collection.query(
        query_embeddings=q_embedding.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    chunks = results["documents"][0]
    return "\n\n---\n\n".join(chunks)


# ============================================================
#  ENTRAINEMENT : récupération de la signature pour chaînage
# ============================================================
def get_training_marker_dir(model_short):
    if model_short == "qwen25_7b":
        return os.path.join(MODELS_DIR, "qwen25_7b_finetuned")
    return os.path.join(MODELS_DIR, f"{model_short}_finetuned")


def get_training_signature(model_short):
    """
    Récupère la signature d'entraînement du modèle, pour qu'un
    ré-entraînement invalide automatiquement le cache d'éval qui en
    dépend. Si seul un ancien marqueur .txt existe (pré-migration),
    on renvoie une valeur fixe distincte pour forcer une réévaluation
    (peu coûteuse) une seule fois.
    """
    base_dir = get_training_marker_dir(model_short)
    marker = read_marker(os.path.join(base_dir, "TRAINING_DONE.json"))
    if marker:
        return marker.get("signature", "unknown")
    if os.path.exists(os.path.join(base_dir, "TRAINING_DONE.txt")):
        return "legacy-pre-migration"
    return "not-trained"


# ============================================================
#  CHARGEMENT MODÈLE FT+RAG
#  - Qwen  : chargé directement depuis raoulbia (pas d'adapter)
#  - LLaMA/Mistral : base + adapter LoRA
# ============================================================
def load_finetuned_model(model_short, hf_base_name):
    adapter_dir  = os.path.join(MODELS_DIR, f"{model_short}_finetuned")
    has_adapter  = HAS_LORA_ADAPTER[model_short]

    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        bnb_config = None

    if model_short == "qwen25_7b":
        # Qwen : chargé directement depuis raoulbia ou local
        src = adapter_dir if os.path.exists(os.path.join(adapter_dir, "config.json")) else hf_base_name
        print(f"  [Qwen] Chargement depuis : {src}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.truncation_side = "left"   # FIX : garder la fin du prompt (question + "### Response:")

        if USE_4BIT:
            model = AutoModelForCausalLM.from_pretrained(
                src,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                src,
                device_map="cpu",
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        model.eval()
        print(f"  [Qwen] Modèle chargé (pré-entraîné 3GPP)", flush=True)

    else:
        # LLaMA / Mistral : base + adapter LoRA
        print(f"  Chargement tokenizer depuis : {adapter_dir}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.truncation_side = "left"   # FIX : garder la fin du prompt (question + "### Response:")

        print(f"  Chargement modèle base : {hf_base_name}", flush=True)
        if USE_4BIT:
            base_model = AutoModelForCausalLM.from_pretrained(
                hf_base_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                offload_buffers=True
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                hf_base_name,
                device_map="cpu",
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )

        print(f"  Chargement adapter LoRA depuis : {adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        model.eval()
        print(f"  Adapter LoRA chargé", flush=True)

    return model, tokenizer


# ============================================================
#  INFÉRENCE FT+RAG
# ============================================================
def ask_finetuned_model(model, tokenizer, question, options, context, max_new_tokens=MAX_NEW_TOKENS):
    # Options présentes (TeleQnA) ou absent (3gpp-5g-nr-qa)
    options_block = ""
    if options:
        options_text  = "\n".join([f"{k}: {v}" for k, v in options.items()])
        options_block = f"\n{options_text}\n\nRespond with ONLY the option number (e.g., \"option 1\")."

    prompt = f"""### Instruction:
You are a 3GPP telecommunications expert. Use the following context to answer the question.

Context: {context[:1500]}

Question: {question}{options_block}

### Response:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response  = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return response


# ============================================================
#  UTILITAIRES
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
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def check_model_ready(model_short):
    """Vérifie si le modèle est prêt (fine-tuning terminé ou Qwen pré-entraîné)."""
    base_dir = get_training_marker_dir(model_short)
    has_marker = (
        os.path.exists(os.path.join(base_dir, "TRAINING_DONE.json")) or
        os.path.exists(os.path.join(base_dir, "TRAINING_DONE.txt"))
    )
    if model_short == "qwen25_7b":
        if has_marker:
            return True
        # Si pas sauvegardé localement, on peut quand même le charger depuis HuggingFace
        print(f"  [Qwen] Non sauvegardé localement — chargement depuis raoulbia directement", flush=True)
        return True   # toujours disponible depuis HuggingFace
    return has_marker


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    print("\n" + "="*60, flush=True)
    print("  3GPP FINE-TUNING + RAG", flush=True)
    print(f"  Évaluation sur : TeleQnA (ancien dataset)", flush=True)
    print(f"  Device : {DEVICE.upper()}", flush=True)
    print("="*60, flush=True)

    if DEVICE == "cpu":
        print("⚠️  Mode CPU — inférence lente mais fonctionnelle", flush=True)

    # Chargement questions TeleQnA
    print(f"\nChargement questions : {TEST_FILE}", flush=True)
    questions = load_teleqna_questions(TEST_FILE, max_questions=100)

    # Embedding model
    print(f"\nChargement embedding : {EMB_MODEL}", flush=True)
    emb_model = SentenceTransformer(EMB_MODEL, device=DEVICE)
    print(f"Embedding chargé sur {DEVICE.upper()}", flush=True)

    # ChromaDB
    print(f"\nConnexion ChromaDB : {CHROMA_DIR}", flush=True)
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLL_NAME)
    print(f"Collection : {collection.count()} chunks", flush=True)

    all_results = {}

    for model_short, hf_base_name in HF_BASE_MODELS.items():

        print(f"\n{'='*60}", flush=True)
        print(f"  Modèle : {model_short}", flush=True)
        print(f"{'='*60}", flush=True)

        cache_file = os.path.join(RESULTS_DIR, f"ft_rag_{model_short}.json")

        # La signature chaîne la signature d'entraînement : si le modèle
        # est ré-entraîné, ce cache se réinvalide automatiquement.
        eval_signature = compute_signature({
            "model":              model_short,
            "training_signature": get_training_signature(model_short),
            "top_k":               TOP_K,
            "max_new_tokens":       MAX_NEW_TOKENS,
            "prompt_version":       PROMPT_VERSION,
            "num_questions":        len(questions)
        })

        cached = load_json_cache(cache_file, eval_signature, force=FORCE_REFRESH)
        if cached:
            all_results[model_short] = cached
            acc, correct, total = compute_accuracy(cached["results"])
            print(f"  {model_short} : {correct}/{total} = {acc:.1%} (cache valide)", flush=True)
            continue
        elif os.path.exists(cache_file):
            print(f"  ⚠️  Cache existant mais périmé (modèle ré-entraîné ou paramètres "
                  f"d'éval changés) — réévaluation", flush=True)

        # Vérifier si modèle prêt
        if not check_model_ready(model_short):
            print(f"  ⚠️  {model_short} non prêt (fine-tuning pas terminé), passage au suivant", flush=True)
            continue

        # Charger modèle
        print(f"  Chargement modèle fine-tuné...", flush=True)
        try:
            model, tokenizer = load_finetuned_model(model_short, hf_base_name)
        except Exception as e:
            print(f"  ❌ Erreur chargement : {e}", flush=True)
            continue

        # Inférence
        model_results = []
        start_time    = time.time()

        for q in tqdm(questions, desc=f"{model_short} FT+RAG"):
            q_start = time.time()
            try:
                context  = retrieve_context(q["question"], collection, emb_model, top_k=TOP_K)
                response = ask_finetuned_model(model, tokenizer, q["question"], q["options"], context)
            except Exception as e:
                response = "error"
                print(f"  ❌ Erreur q{q['id']} : {e}", flush=True)

            q_time = time.time() - q_start
            model_results.append({
                "id":               q["id"],
                "question":         q["question"],
                "true_answer":      q["answer"],
                "predicted_answer": response,
                "extracted_option": extract_option(response),
                "category":         q["category"],
                "response_time_s":  round(q_time, 2)
            })

        acc, correct, total = compute_accuracy(model_results)
        total_time = time.time() - start_time

        result_data = {
            "model":        model_short,
            "pipeline":     "ft_rag",
            "accuracy":     acc,
            "correct":      correct,
            "total":        total,
            "total_time_s": round(total_time, 2),
            "results":      model_results
        }

        save_json_cache(cache_file, result_data, eval_signature)

        all_results[model_short] = result_data
        print(f"  ✅ {model_short} — Accuracy : {correct}/{total} = {acc:.1%}", flush=True)
        print(f"  Temps : {round(total_time/60, 1)} minutes", flush=True)

        del model, tokenizer
        gc.collect()
        if USE_4BIT:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(2)
        print(f"  Mémoire libérée", flush=True)

    # ── Tableau résultats ──────────────────────────────────────────
    print("\n" + "="*60, flush=True)
    print("  RÉSULTATS FT + RAG", flush=True)
    print("="*60, flush=True)
    print(f"{'Modèle':<20} {'Accuracy':<12} {'Correct'}", flush=True)
    print("-"*50, flush=True)

    summary = []
    for model_short, data in all_results.items():
        acc     = data["accuracy"]
        correct = data["correct"]
        total   = data["total"]
        print(f"{model_short:<20} {acc:<12.1%} {correct}/{total}", flush=True)
        summary.append({
            "model":    model_short,
            "pipeline": "ft_rag",
            "accuracy": acc,
            "correct":  correct,
            "total":    total
        })

    summary_file = os.path.join(RESULTS_DIR, "ft_rag_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRésumé sauvegardé : {summary_file}", flush=True)
    print("Prochaine étape   : python 07_raft_fast.py", flush=True)
    print("="*60, flush=True)
