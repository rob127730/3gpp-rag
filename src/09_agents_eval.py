# ============================================================
#  09_agents_eval.py
#  Évaluation des 3 agents sur TeleQnA
#  Agent 1 + 2 : Ollama (~5-10s/question)
#  Agent 3     : Ollama×2 + Arbitre Mistral HF local
# ============================================================

import os
import json
import yaml
import time
from tqdm import tqdm
from agents_core import (
    load_vectorstore, load_hf_arbitre, build_agents,
    extract_option
)

print("STEP 1 : Imports OK", flush=True)

with open(os.environ.get("RAG_CONFIG", "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

TEST_FILE   = config["paths"]["test_teleqna"]
RESULTS_DIR = config["paths"]["results_dir"]
os.makedirs(RESULTS_DIR, exist_ok=True)
MAX_QUESTIONS = 100


# ============================================================
#  UTILITAIRES
# ============================================================
def load_teleqna_questions(filepath, max_questions=MAX_QUESTIONS):
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


def format_mcq(question, options):
    """Formate question + options pour l'agent."""
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    return (
        f"{question}\n\n"
        f"{options_text}\n\n"
        f"Respond with ONLY the option number (e.g., 'option 1')."
    )


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


def evaluate_agent(agent_name, agent_fn, questions, cache_filename):
    cache_file = os.path.join(RESULTS_DIR, cache_filename)

    if os.path.exists(cache_file):
        print(f"\n  📂 Cache trouvé pour {agent_name}", flush=True)
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        acc, correct, total = compute_accuracy(data["results"])
        print(f"  {agent_name} : {correct}/{total} = {acc:.1%}", flush=True)
        return data

    print(f"\n{'='*60}", flush=True)
    print(f"  Évaluation : {agent_name}", flush=True)
    print(f"  Questions  : {len(questions)}", flush=True)
    print(f"{'='*60}", flush=True)

    results    = []
    start_time = time.time()

    for q in tqdm(questions, desc=agent_name):
        q_start = time.time()
        try:
            prompt   = format_mcq(q["question"], q["options"])
            response = agent_fn(prompt)
        except Exception as e:
            response = "error"
            print(f"\n  ❌ Erreur q{q['id']} : {e}", flush=True)

        results.append({
            "id":               q["id"],
            "question":         q["question"],
            "true_answer":      q["answer"],
            "predicted_answer": response,
            "extracted_option": extract_option(response),
            "category":         q["category"],
            "response_time_s":  round(time.time() - q_start, 2)
        })

    acc, correct, total = compute_accuracy(results)
    total_time = time.time() - start_time

    result_data = {
        "agent":        agent_name,
        "pipeline":     "agent",
        "accuracy":     acc,
        "correct":      correct,
        "total":        total,
        "total_time_s": round(total_time, 2),
        "results":      results
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ {agent_name} — {correct}/{total} = {acc:.1%} ({round(total_time/60,1)} min)", flush=True)
    return result_data


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    print("\n" + "="*65, flush=True)
    print("  ÉVALUATION DES 3 AGENTS IA — TeleQnA", flush=True)
    print("  Agent 1+2 : Ollama (rapide ~5-10s/question)", flush=True)
    print("  Agent 3   : Ollama×2 + Arbitre Mistral HF local", flush=True)
    print("="*65, flush=True)

    # Composants partagés
    collection, emb_model = load_vectorstore()

    # Arbitre HuggingFace pour Agent 3
    print(f"\n📥 Chargement arbitre Mistral local pour Agent 3...", flush=True)
    hf_model, hf_tokenizer = load_hf_arbitre()

    # Construire les 3 agents
    agents = build_agents(collection, emb_model, hf_model, hf_tokenizer)

    # Questions TeleQnA
    print(f"\n📖 Chargement questions : {TEST_FILE}", flush=True)
    questions = load_teleqna_questions(TEST_FILE, max_questions=MAX_QUESTIONS)

    # Évaluation
    all_results = {}
    cache_map = {
        "agent1": "agent1_simple.json",
        "agent2": "agent2_rag.json",
        "agent3": "agent3_multi.json"
    }

    for key, info in agents.items():
        data = evaluate_agent(
            info["name"], info["fn"],
            questions, cache_map[key]
        )
        all_results[key] = data

    # Tableau comparatif
    print("\n" + "="*75, flush=True)
    print("  TABLEAU COMPARATIF — TOUS LES PIPELINES (Mistral-7B)", flush=True)
    print("="*75, flush=True)
    print(f"  {'Pipeline':<35} {'Accuracy':<12} {'Correct/Total'}", flush=True)
    print(f"  {'-'*65}", flush=True)

    pipelines_known = [
        ("Baseline Ollama",            "16.0%", "16/100"),
        ("RAG Baseline",               "21.0%", "21/100"),
        ("RAG Optimisé",               "25.0%", "25/100"),
        ("Fine-tuning seul",           "61.0%", "61/100"),
        ("FT + RAG",                   "64.0%", "64/100"),
        ("RAFT",                       "54.0%", "54/100"),
        ("RAFT + RAG",                 "?",     "—"),
    ]
    for name, acc, corr in pipelines_known:
        print(f"  {name:<35} {acc:<12} {corr}", flush=True)

    print(f"  {'-'*65}", flush=True)

    for key, label in [
        ("agent1", "Agent 1 (Ollama + similarity)"),
        ("agent2", "Agent 2 (Ollama + MMR)"),
        ("agent3", "Agent 3 (Multi + HF arbitre + Web)")
    ]:
        if key in all_results:
            d = all_results[key]
            print(
                f"  {label:<35} {d['accuracy']:<12.1%} "
                f"{d['correct']}/{d['total']}",
                flush=True
            )

    # Résumé JSON
    summary = {
        k: {"accuracy": v["accuracy"], "correct": v["correct"],
            "total": v["total"], "time_min": round(v["total_time_s"]/60, 1)}
        for k, v in all_results.items()
    }
    summary_file = os.path.join(RESULTS_DIR, "agents_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Résumé sauvegardé : {summary_file}", flush=True)
    print("="*75, flush=True)
