# ============================================================
#  03_rag_optimized.py
#  RAG Optimisé - Reranking + BM25 hybride + meilleur embedding
#  3 modèles via Ollama + ChromaDB
# ============================================================

import os
import json
import yaml
import time
import requests
import chromadb
from tqdm import tqdm
from loguru import logger
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import torch

with open(os.environ.get("RAG_CONFIG", "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

CHROMA_DIR  = config["paths"]["chroma_db"]
COLL_NAME   = config["rag_baseline"]["collection_name"]
EMB_MODEL   = config["rag_baseline"]["embedding_model"]
TEST_FILE   = config["paths"]["test_teleqna"]
RESULTS_DIR = config["paths"]["results_dir"]

TOP_K_RETRIEVE = 20   # on recupere 20 chunks
TOP_K_RERANK   = 5    # on garde les 5 meilleurs apres reranking
BM25_WEIGHT    = 0.3  # 30% BM25
DENSE_WEIGHT   = 0.7  # 70% semantique

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
logger.add("logs/rag_optimized.log", rotation="10 MB")

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
        options = {}
        for key, val in q_data.items():
            if key.startswith("option"):
                options[key] = val
        raw_answer = q_data.get("answer", None)
        answer = raw_answer.split(":")[0].strip() if raw_answer else None
        questions.append({
            "id":       q_id,
            "question": q_data.get("question", ""),
            "options":  options,
            "answer":   answer,
            "category": q_data.get("category", "")
        })
    print(f"✅ {len(questions)} questions chargees")
    return questions


def retrieve_dense(question_text, collection, emb_model, top_k=20):
    """Recherche semantique dense dans ChromaDB"""
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
    distances = results["distances"][0]
    return chunks, distances


def retrieve_bm25(question_text, bm25_index, all_chunks, top_k=20):
    """Recherche BM25 par mots-cles"""
    tokenized_query = question_text.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    chunks = [all_chunks[i] for i in top_indices]
    chunk_scores = [scores[i] for i in top_indices]
    return chunks, chunk_scores


def hybrid_retrieve(question_text, collection, emb_model, bm25_index, all_chunks, top_k_retrieve=20):
    """
    Combine recherche dense + BM25
    Retourne les top_k_retrieve chunks uniques
    """
    dense_chunks, dense_scores = retrieve_dense(question_text, collection, emb_model, top_k_retrieve)
    bm25_chunks, bm25_scores = retrieve_bm25(question_text, bm25_index, all_chunks, top_k_retrieve)

    # Normaliser les scores BM25
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_scores_norm = [s / max_bm25 for s in bm25_scores]

    # Normaliser les scores dense (distance -> similarite)
    max_dense = max(dense_scores) if max(dense_scores) > 0 else 1
    dense_scores_norm = [1 - (s / max_dense) for s in dense_scores]

    # Combiner dans un dictionnaire
    chunk_scores = {}
    for chunk, score in zip(dense_chunks, dense_scores_norm):
        chunk_scores[chunk[:100]] = {"chunk": chunk, "score": score * DENSE_WEIGHT}

    for chunk, score in zip(bm25_chunks, bm25_scores_norm):
        key = chunk[:100]
        if key in chunk_scores:
            chunk_scores[key]["score"] += score * BM25_WEIGHT
        else:
            chunk_scores[key] = {"chunk": chunk, "score": score * BM25_WEIGHT}

    # Trier par score hybride
    sorted_chunks = sorted(chunk_scores.values(), key=lambda x: x["score"], reverse=True)
    return [c["chunk"] for c in sorted_chunks[:top_k_retrieve]]


def rerank(question_text, chunks, reranker, top_k=5):
    """
    Reranker cross-encoder : evalue chaque chunk par rapport a la question
    Beaucoup plus precis que la recherche initiale
    """
    pairs = [[question_text, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    top_chunks = [chunk for chunk, score in ranked[:top_k]]
    context = "\n\n---\n\n".join(top_chunks)
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
                "options": {"temperature": 0.1, "num_predict": 20}
            },
            timeout=timeout
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.Timeout:
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
    print("  3GPP RAG OPTIMISE")
    print("  Hybrid BM25 + Dense + Reranking")
    print("="*60)

    # 1. Charger questions
    questions = load_teleqna_questions(TEST_FILE, max_questions=100)

    # 2. Charger embedding
    print(f"\n📥 Chargement embedding : {EMB_MODEL}")
    emb_model = SentenceTransformer(EMB_MODEL, device=DEVICE)
    print(f"✅ Embedding charge sur {DEVICE.upper()}")

    # 3. Charger reranker
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"\n📥 Chargement reranker : {RERANKER_MODEL}")
    reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE)
    print(f"✅ Reranker charge")

    # 4. Connexion ChromaDB
    print(f"\n💾 Connexion ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLL_NAME)
    total_chunks = collection.count()
    print(f"✅ Collection : {total_chunks} chunks")

    # 5. Construire index BM25 (cache)
    BM25_CACHE = "bm25_chunks_cache.json"
    if os.path.exists(BM25_CACHE):
        print(f"\n📂 Cache BM25 trouve, chargement...")
        with open(BM25_CACHE, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        print(f"✅ {len(all_chunks)} chunks charges depuis cache")
    else:
        print(f"\n🔨 Construction index BM25 ({total_chunks} chunks)...")
        print(f"   Recuperation par batch de 5000...")
        all_chunks = []
        batch_size = 5000
        # Recuperer tous les chunks de ChromaDB
        result = collection.get(include=["documents"], limit=total_chunks)
        all_chunks = result["documents"]
        with open(BM25_CACHE, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f)
        print(f"✅ {len(all_chunks)} chunks recuperes et sauvegardes")

    print(f"\n🔨 Construction index BM25...")
    tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    print(f"✅ Index BM25 construit")

    # 6. Verifier Ollama
    print(f"\n🤖 Verification Ollama...")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models_available = [m["name"] for m in r.json().get("models", [])]
        print(f"✅ Ollama actif : {models_available}")
    except Exception:
        print(f"❌ Ollama n'est pas demarre !")
        exit(1)

    # 7. Lancer RAG optimise pour chaque modele
    all_results = {}

    for model_short, model_name in MODELS.items():

        cache_file = os.path.join(RESULTS_DIR, f"rag_optimized_{model_short}.json")

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

            # Recherche hybride
            candidates = hybrid_retrieve(
                q["question"], collection, emb_model,
                bm25_index, all_chunks, TOP_K_RETRIEVE
            )

            # Reranking
            context = rerank(q["question"], candidates, reranker, TOP_K_RERANK)

            # Reponse du modele
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
            "pipeline":     "rag_optimized",
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
        logger.info(f"{model_short} RAG optimized : {acc:.1%} accuracy")

    # 8. Comparaison baseline vs optimise
    print("\n" + "="*60)
    print("  COMPARAISON RAG BASELINE vs RAG OPTIMISE")
    print("="*60)
    print(f"{'Modele':<20} {'Baseline':<12} {'Optimise':<12} {'Gain'}")
    print("-"*60)

    baseline_file = os.path.join(RESULTS_DIR, "rag_baseline_summary.json")
    baseline_data = {}
    if os.path.exists(baseline_file):
        with open(baseline_file, "r") as f:
            for item in json.load(f):
                baseline_data[item["model"]] = item["accuracy"]

    summary = []
    for model_short, data in all_results.items():
        acc_opt = data["accuracy"]
        acc_base = baseline_data.get(model_short, 0)
        gain = acc_opt - acc_base
        gain_str = f"+{gain:.1%}" if gain >= 0 else f"{gain:.1%}"
        print(f"{model_short:<20} {acc_base:<12.1%} {acc_opt:<12.1%} {gain_str}")
        summary.append({
            "model":    model_short,
            "pipeline": "rag_optimized",
            "accuracy": acc_opt,
            "correct":  data["correct"],
            "total":    data["total"]
        })

    summary_file = os.path.join(RESULTS_DIR, "rag_optimized_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Resume sauvegarde : {summary_file}")
    print("\n  Prochaine etape : python 04_finetune_qlora.py")
    print("="*60)
