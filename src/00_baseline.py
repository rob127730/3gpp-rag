# ============================================================
#  00_baseline.py
#  BASELINE PURE — zero-shot, zero RAG
#  3 modeles via Ollama
# ============================================================

import os
import json
import yaml
import time
import requests
from tqdm import tqdm
from loguru import logger

with open(os.environ.get("RAG_CONFIG", "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

TEST_FILE   = config["paths"]["test_teleqna"]
RESULTS_DIR = config["paths"]["results_dir"]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
logger.add("logs/baseline.log", rotation="10 MB")

MODELS = {
    "qwen25_7b":  "qwen2.5:7b",
    "llama31_8b": "llama3.1:8b",
    "mistral_7b": "mistral:7b-instruct-q4_0"
}


def load_questions(filepath, max_questions=100):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = []
    for q_id, q_data in list(data.items())[:max_questions]:
        options = {k: v for k, v in q_data.items() if k.startswith("option")}
        raw_answer = q_data.get("answer", None)
        answer = raw_answer.split(":")[0].strip() if raw_answer else None
        if not answer or not options:
            continue
        questions.append({
            "id":       q_id,
            "question": q_data.get("question", ""),
            "options":  options,
            "answer":   answer,
            "category": q_data.get("category", "")
        })
    print(f"  {len(questions)} questions chargees")
    return questions


def ask_ollama(model_name, question, options, timeout=120):
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])

    if "mistral" in model_name.lower():
        prompt = (f"[INST] Answer this multiple choice question with ONLY "
                  f"'option 1', 'option 2', etc.\n\n"
                  f"{question}\n\n{options_text} [/INST]")
    else:
        prompt = (f"You are a 3GPP telecommunications expert.\n\n"
                  f"QUESTION:\n{question}\n\n"
                  f"OPTIONS:\n{options_text}\n\n"
                  f"Instructions: Reply with ONLY the option number "
                  f"(e.g., 'option 1'). Do not explain.\n\nAnswer:")

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt,
                  "stream": False,
                  "options": {"temperature": 0.1, "num_predict": 20}},
            timeout=timeout
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout : {model_name}")
        return "timeout"
    except Exception as e:
        logger.error(f"Erreur Ollama {model_name}: {e}")
        return "error"


def extract_option(text):
    t = text.lower().strip()
    for i in range(1, 6):
        if f"option {i}" in t:
            return f"option {i}"
    for i in range(1, 6):
        if t.startswith(str(i)):
            return f"option {i}"
    return text


def compute_accuracy(results):
    correct, total = 0, 0
    for r in results:
        if r["true_answer"] and r["predicted_answer"]:
            total += 1
            true = r["true_answer"].lower().strip()
            pred = extract_option(r["predicted_answer"]).lower().strip()
            if true == pred or true in pred or pred in true:
                correct += 1
    return (correct / total if total else 0), correct, total


if __name__ == "__main__":

    print("\n" + "="*60)
    print("  3GPP BASELINE PURE — Zero-shot, Zero RAG")
    print("  3 Modeles via Ollama")
    print("="*60)

    # Verifier Ollama
    print("\nVerification Ollama...")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models_available = [m["name"] for m in r.json().get("models", [])]
        print(f"  Ollama actif — Modeles : {models_available}")
    except Exception:
        print("  Ollama n'est pas demarre !")
        exit(1)

    print(f"\nChargement questions : {TEST_FILE}")
    questions = load_questions(TEST_FILE, max_questions=100)

    all_results = {}

    for model_short, model_name in MODELS.items():

        cache_file = os.path.join(RESULTS_DIR, f"baseline_{model_short}.json")

        if os.path.exists(cache_file):
            print(f"\n  Cache trouve : {model_short}")
            with open(cache_file, "r", encoding="utf-8") as f:
                all_results[model_short] = json.load(f)
            acc, correct, total = compute_accuracy(all_results[model_short]["results"])
            print(f"  {model_short} : {correct}/{total} = {acc:.1%}")
            continue

        print(f"\n{'='*60}")
        print(f"  Modele : {model_name}")
        print(f"{'='*60}")

        model_results = []
        start_time    = time.time()

        for q in tqdm(questions, desc=f"{model_short}"):
            q_start  = time.time()
            response = ask_ollama(model_name, q["question"], q["options"])
            q_time   = time.time() - q_start

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
            "pipeline":     "baseline",
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

        print(f"\n  {model_short} termine !")
        print(f"  Accuracy : {correct}/{total} = {acc:.1%}")
        print(f"  Temps    : {round(total_time/60, 1)} minutes")
        logger.info(f"{model_short} baseline : {acc:.1%}")

    # Tableau final
    print("\n" + "="*60)
    print("  RESULTATS BASELINE PURE")
    print("="*60)
    print(f"{'Modele':<20} {'Accuracy':<12} {'Correct':<10} {'Temps'}")
    print("-"*60)

    summary = []
    for model_short, data in all_results.items():
        acc     = data["accuracy"]
        correct = data["correct"]
        total   = data["total"]
        t       = data["total_time_s"]
        print(f"{model_short:<20} {acc:.1%}{'':6} "
              f"{correct}/{total}{'':4} {round(t/60,1)} min")
        summary.append({"model": model_short, "pipeline": "baseline",
                         "accuracy": acc, "correct": correct, "total": total})

    summary_file = os.path.join(RESULTS_DIR, "baseline_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResume sauvegarde : {summary_file}")
    print("Prochaine etape : python 02_rag_baseline.py")
    print("="*60)
