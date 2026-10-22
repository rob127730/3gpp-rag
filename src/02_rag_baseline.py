# ============================================================
#  02_rag_baseline.py
#  RAG Baseline - 3 modèles via Ollama + ChromaDB
#  Sauvegarde progressive pour ne pas tout reprendre si crash
# ============================================================

import os
import json
import yaml
import time
import requests
import chromadb
from tqdm import tqdm
from loguru import logger
from sentence_transformers import SentenceTransformer
import torch

with open(os.environ.get("RAG_CONFIG", "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

CHROMA_DIR  = config["paths"]["chroma_db"]
COLL_NAME   = config["rag_baseline"]["collection_name"]
EMB_MODEL   = config["rag_baseline"]["embedding_model"]
TOP_K       = config["rag_baseline"]["top_k"]
TEST_FILE   = config["paths"]["test_teleqna"]
RESULTS_DIR = config["paths"]["results_dir"]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
logger.add("logs/rag_baseline.log", rotation="10 MB")

MODELS = {
    "qwen25_7b":  "qwen2.5:7b",
    "llama31_8b": "llama3.1:8b",
    "mistral_7b": "mistral:7b-instruct-q4_0"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_teleqna_questions(filepath, max_questions=100):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = []
    for q_id, q_data in list(data.items())[:max_questions]:
        q_text = q_data.get("question", "")
        options = {}
        for key, val in q_data.items():
            if key.startswith("option"):
                options[key] = val
        raw_answer = q_data.get("answer", None)
        answer = raw_answer.split(":")[0].strip() if raw_answer else None
        questions.append({
            "id":       q_id,
            "question": q_text,
            "options":  options,
            "answer":   answer,
            "category": q_data.get("category", "")
        })
    print(f"✅ {len(questions)} questions chargees")
    print(f"   Exemple reponse : {questions[0]['answer']}")
    return questions


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
    context = "\n\n---\n\n".join(chunks)
    return context


def ask_ollama(model_name, question, options, context, timeout=120):
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])

    if "mistral" in model_name.lower():
        short_context = context[:1000]
        prompt = f"""[INST] Answer this multiple choice question with ONLY "option 1", "option 2", etc.

{short_context}

{question}

{options_text} [/INST]"""
    else:
        prompt = f"""You are a 3GPP telecommunications expert. Use the following context from 3GPP specifications to answer the multiple choice question.

CONTEXT:
{context}

QUESTION:
{question}

OPTIONS:
{options_text}

Instructions:
- Read the context carefully
- Select the best answer from the options above
- Respond with ONLY the option number (e.g., "option 1" or "option 2")
- Do not explain, just give the option number

Answer:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 20,
                }
            },
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout pour modele {model_name}")
        return "timeout"
    except Exception as e:
        logger.error(f"Erreur Ollama {model_name}: {e}")
        return "error"


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
    correct = 0
    total = 0
    for r in results:
        if r["true_answer"] and r["predicted_answer"]:
            total += 1
            true = r["true_answer"].lower().strip()
            pred = extract_option(r["predicted_answer"]).lower().strip()
            if true == pred or true in pred or pred in true:
                correct += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


if __name__ == "__main__":

    print("\n" + "="*60)
    print("  3GPP RAG BASELINE")
    print("  3 Modeles x TeleQnA Questions")
    print("="*60)

    print(f"\n📖 Chargement questions : {TEST_FILE}")
    questions = load_teleqna_questions(TEST_FILE, max_questions=100)

    print(f"\n📥 Chargement embedding : {EMB_MODEL}")
    emb_model = SentenceTransformer(EMB_MODEL, device=DEVICE)
    print(f"✅ Embedding charge sur {DEVICE.upper()}")

    print(f"\n💾 Connexion ChromaDB : {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLL_NAME)
    print(f"✅ Collection '{COLL_NAME}' : {collection.count()} chunks")

    print(f"\n🤖 Verification Ollama...")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models_available = [m["name"] for m in r.json().get("models", [])]
        print(f"✅ Ollama actif - Modeles : {models_available}")
    except Exception as e:
        print(f"❌ Ollama n'est pas demarre !")
        exit(1)

    all_results = {}

    for model_short, model_name in MODELS.items():

        cache_file = os.path.join(RESULTS_DIR, f"rag_baseline_{model_short}.json")

        if os.path.exists(cache_file):
            print(f"\n📂 Cache trouve pour {model_short}, chargement...")
            with open(cache_file, "r", encoding="utf-8") as f:
                all_results[model_short] = json.load(f)
            acc, correct, total = compute_accuracy(all_results[model_short]["results"])
            print(f"✅ {model_short} : {correct}/{total} = {acc:.1%} accuracy")
            continue

        print(f"\n{'='*60}")
        print(f"  Modele : {model_name}")
        print(f"{'='*60}")

        model_results = []
        start_time = time.time()

        for q in tqdm(questions, desc=f"🤖 {model_short}"):
            q_start = time.time()
            context = retrieve_context(q["question"], collection, emb_model, top_k=TOP_K)
            response = ask_ollama(model_name, q["question"], q["options"], context)
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
            "model":        model_name,
            "model_short":  model_short,
            "pipeline":     "rag_baseline",
            "accuracy":     acc,
            "correct":      correct,
            "total":        total,
            "total_time_s": round(total_time, 2),
            "avg_time_s":   round(total_time / len(questions), 2),
            "results":      model_results
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        all_results[model_short] = result_data

        print(f"\n✅ {model_short} termine !")
        print(f"   Accuracy  : {correct}/{total} = {acc:.1%}")
        print(f"   Temps     : {round(total_time/60, 1)} minutes")
        print(f"   Cache     : {cache_file}")
        logger.info(f"{model_short} RAG baseline : {acc:.1%} accuracy")

    print("\n" + "="*60)
    print("  RESULTATS RAG BASELINE")
    print("="*60)
    print(f"{'Modele':<20} {'Accuracy':<12} {'Correct':<10} {'Temps'}")
    print("-"*60)

    summary = []
    for model_short, data in all_results.items():
        acc = data["accuracy"]
        correct = data["correct"]
        total = data["total"]
        t = data["total_time_s"]
        score_str = f"{correct}/{total}"
        acc_str = f"{acc:.1%}"
        print(f"{model_short:<20} {acc_str:<12} {score_str:<10} {round(t/60,1)} min")
        summary.append({
            "model":    model_short,
            "pipeline": "rag_baseline",
            "accuracy": acc,
            "correct":  correct,
            "total":    total
        })

    summary_file = os.path.join(RESULTS_DIR, "rag_baseline_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Resume sauvegarde : {summary_file}")
    print("\n  Prochaine etape : python 03_rag_optimized.py")
    print("="*60)
