# ============================================================
#  05_evaluate_finetuned.py
#  Évaluation des 3 modèles fine-tunés QLoRA
#  Même 100 questions que baseline et optimized
#  Génère un tableau comparatif complet
# ============================================================
 
import os
import json
import yaml
import time
import gc
import torch
from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
 
with open(os.environ.get("RAG_CONFIG", "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
 
TEST_FILE   = config["paths"]["test_teleqna"]
MODELS_DIR  = config["paths"]["models_dir"]
RESULTS_DIR = config["paths"]["results_dir"]
 
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
logger.add("logs/evaluate_finetuned.log", rotation="10 MB")
 
# Modèles HuggingFace de base (nécessaires pour charger les adaptateurs LoRA)
HF_BASE_MODELS = {
    "qwen25_7b":  "Qwen/Qwen2.5-7B-Instruct",
    "llama31_8b": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3"
}
 
# Résultats déjà connus pour le tableau comparatif final
KNOWN_RESULTS = {
    "qwen25_7b":  {"rag_baseline": 0.66, "rag_optimized": 0.78},
    "llama31_8b": {"rag_baseline": 0.60, "rag_optimized": 0.75},
    "mistral_7b": {"rag_baseline": 0.21, "rag_optimized": 0.25},
}
 
 
# ============================================================
#  CHARGEMENT DES QUESTIONS (identique à 02_rag_baseline.py)
# ============================================================
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
    print(f"✅ {len(questions)} questions chargées")
    print(f"   Exemple réponse attendue : {questions[0]['answer']}")
    return questions
 
 
# ============================================================
#  EXTRACTION DE L'OPTION (identique à 02_rag_baseline.py)
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
 
 
# ============================================================
#  CALCUL ACCURACY (identique à 02_rag_baseline.py)
# ============================================================
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
 
 
# ============================================================
#  CHARGEMENT MODÈLE FINE-TUNÉ (base + adaptateur LoRA)
# ============================================================
def load_finetuned_model(model_short, hf_base_name):
    adapter_path = os.path.join(MODELS_DIR, f"{model_short}_finetuned")
 
    if not os.path.exists(adapter_path):
        print(f"❌ Modèle fine-tuné introuvable : {adapter_path}")
        return None, None
 
    done_file = os.path.join(adapter_path, "TRAINING_DONE.txt")
    if not os.path.exists(done_file):
        print(f"⚠️  TRAINING_DONE.txt absent pour {model_short} — fine-tuning incomplet ?")
 
    print(f"\n📥 Chargement modèle de base : {hf_base_name}")
 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
 
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,  # tokenizer sauvegardé avec le fine-tuning
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
 
    base_model = AutoModelForCausalLM.from_pretrained(
        hf_base_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        offload_buffers=True
    )
 
    print(f"📥 Chargement adaptateur LoRA : {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
 
    print(f"✅ Modèle fine-tuné chargé sur GPU")
    return model, tokenizer
 
 
# ============================================================
#  INFÉRENCE : poser une question au modèle fine-tuné
# ============================================================
def ask_finetuned(model, tokenizer, question, options, max_new_tokens=30):
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
 
    prompt = f"""### Instruction:
You are a 3GPP telecommunications expert. Answer the following multiple choice question.
 
Question: {question}
 
{options_text}
 
Respond with ONLY the option number (e.g., "option 1").
 
### Response:
"""
 
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)
 
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
 
    # Décoder seulement les nouveaux tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response
 
 
# ============================================================
#  ÉVALUATION D'UN MODÈLE FINE-TUNÉ
# ============================================================
def evaluate_finetuned_model(model_short, hf_base_name, questions):
    cache_file = os.path.join(RESULTS_DIR, f"finetuned_{model_short}.json")
 
    # Cache : ne pas refaire si déjà évalué
    if os.path.exists(cache_file):
        print(f"\n📂 Cache trouvé pour {model_short}, chargement...")
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        acc, correct, total = compute_accuracy(data["results"])
        print(f"✅ {model_short} fine-tuné : {correct}/{total} = {acc:.1%} accuracy")
        return data
 
    # Charger le modèle
    model, tokenizer = load_finetuned_model(model_short, hf_base_name)
    if model is None:
        return None
 
    print(f"\n{'='*60}")
    print(f"  Évaluation fine-tuné : {model_short}")
    print(f"{'='*60}")
 
    model_results = []
    start_time = time.time()
 
    for q in tqdm(questions, desc=f"🔬 {model_short}_finetuned"):
        q_start = time.time()
        try:
            response = ask_finetuned(model, tokenizer, q["question"], q["options"])
        except Exception as e:
            logger.error(f"Erreur inférence {model_short} q{q['id']}: {e}")
            response = "error"
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
        "model":        hf_base_name,
        "model_short":  model_short,
        "pipeline":     "finetuned",
        "accuracy":     acc,
        "correct":      correct,
        "total":        total,
        "total_time_s": round(total_time, 2),
        "avg_time_s":   round(total_time / len(questions), 2),
        "results":      model_results
    }
 
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
 
    print(f"\n✅ {model_short} fine-tuné évalué !")
    print(f"   Accuracy : {correct}/{total} = {acc:.1%}")
    print(f"   Temps    : {round(total_time/60, 1)} minutes")
    print(f"   Cache    : {cache_file}")
    logger.info(f"{model_short} finetuned : {acc:.1%} accuracy")
 
    # Libérer la VRAM avant le modèle suivant
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(3)
    print(f"🧹 VRAM libérée")
 
    return result_data
 
 
# ============================================================
#  TABLEAU COMPARATIF FINAL
# ============================================================
def print_comparison_table(finetuned_results):
    print("\n" + "="*70)
    print("  TABLEAU COMPARATIF COMPLET — 3 MODÈLES × 3 PIPELINES")
    print("="*70)
    print(f"{'Modèle':<20} {'Baseline':>12} {'Optimized':>12} {'Fine-tuné':>12} {'Δ FT vs Base':>14}")
    print("-"*70)
 
    summary = []
    for model_short in ["qwen25_7b", "llama31_8b", "mistral_7b"]:
        baseline  = KNOWN_RESULTS[model_short]["rag_baseline"]
        optimized = KNOWN_RESULTS[model_short]["rag_optimized"]
 
        ft_data = finetuned_results.get(model_short)
        if ft_data:
            finetuned = ft_data["accuracy"]
            delta = finetuned - baseline
            delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
        else:
            finetuned = None
            delta_str = "N/A"
 
        ft_str = f"{finetuned:.1%}" if finetuned is not None else "❌ erreur"
 
        print(f"{model_short:<20} {baseline:>11.1%} {optimized:>11.1%} {ft_str:>12} {delta_str:>14}")
 
        summary.append({
            "model":        model_short,
            "rag_baseline": baseline,
            "rag_optimized": optimized,
            "finetuned":    finetuned,
            "delta_ft_vs_baseline": round(finetuned - baseline, 4) if finetuned else None
        })
 
    print("="*70)
 
    # Sauvegarder le résumé comparatif
    summary_file = os.path.join(RESULTS_DIR, "comparison_all_pipelines.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Résumé comparatif sauvegardé : {summary_file}")
 
    return summary
 
 
# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
 
    print("\n" + "="*60)
    print("  3GPP — ÉVALUATION MODÈLES FINE-TUNÉS")
    print("  3 Modèles QLoRA sur 100 questions TeleQnA")
    print("="*60)
 
    if not torch.cuda.is_available():
        print("❌ CUDA non disponible !")
        exit(1)
 
    vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    print(f"\n🖥️  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM : {vram} GB")
 
    print(f"\n📖 Chargement questions : {TEST_FILE}")
    questions = load_teleqna_questions(TEST_FILE, max_questions=100)
 
    finetuned_results = {}
 
    for model_short, hf_base_name in HF_BASE_MODELS.items():
        data = evaluate_finetuned_model(model_short, hf_base_name, questions)
        if data:
            finetuned_results[model_short] = data
 
    # Tableau comparatif final
    print_comparison_table(finetuned_results)
 
    print("\n" + "="*60)
    print("  ✅ ÉVALUATION TERMINÉE")
    print(f"  Résultats dans : {RESULTS_DIR}/")
    print("  Prochaine étape : RAG + Fine-tuning combiné")
    print("="*60)