# ============================================================
#  04_finetune_qlora.py
#  Fine-tuning QLoRA - 3 modèles
#  - Qwen2.5-7B  : chargé depuis raoulbia/Qwen2.5-7B-3GPP-NR (déjà fine-tuné)
#  - LLaMA 3.1-8B: fine-tuning QLoRA sur raoulbia/3gpp-5g-nr-qa
#  - Mistral-7B  : fine-tuning QLoRA sur raoulbia/3gpp-5g-nr-qa
#  Reprise automatique depuis le dernier checkpoint
#  Compatible CPU et GPU
#
#  CORRECTIFS (cf. cache_utils.py) :
#  - Les "TRAINING_DONE" ne sont plus de simples fichiers texte
#    dont on vérifie juste l'existence : ils contiennent une
#    signature (modèle, taille du dataset, hyperparamètres,
#    device). Si la config a changé depuis le dernier run,
#    l'ancien dossier est archivé (jamais supprimé) et
#    l'entraînement repart proprement.
#  - Le chargement du dataset HuggingFace retente automatiquement
#    un téléchargement forcé s'il renvoie 0 exemple formaté
#    (cache HF local corrompu/incomplet).
#  - FORCE_RETRAIN=1 force un ré-entraînement complet même si la
#    config n'a pas changé.
# ============================================================

import os
import sys
import gc
import json
import yaml
import time
import torch
from loguru import logger
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print("STEP 1 : Imports OK", flush=True)

from trl import SFTTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_utils import (
    compute_signature, write_marker, read_marker, is_marker_valid,
    migrate_legacy_marker, archive_stale_dir, FORCE_RETRAIN
)

with open(os.environ.get("RAG_CONFIG", "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

MODELS_DIR  = config["paths"]["models_dir"]
RESULTS_DIR = config["paths"]["results_dir"]
FT_CFG      = config["finetuning"]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
logger.add("logs/finetune.log", rotation="10 MB")

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_4BIT = torch.cuda.is_available()

print(f"STEP 2 : Config OK — DEVICE={DEVICE.upper()}", flush=True)
if FORCE_RETRAIN:
    print("⚠️  FORCE_RETRAIN=1 — tous les modèles seront ré-entraînés depuis zéro", flush=True)

# ============================================================
#  MODELES
#  Qwen  -> modèle déjà fine-tuné 3GPP (raoulbia)
#  LLaMA -> fine-tuning depuis base HuggingFace
#  Mistral -> fine-tuning depuis base HuggingFace
# ============================================================
HF_MODELS = {
    "qwen25_7b":  "raoulbia/Qwen2.5-7B-3GPP-NR",          # déjà fine-tuné 3GPP !
    "llama31_8b": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3"
}

# Qwen est déjà fine-tuné, on le sauvegarde juste localement
QWEN_PRETRAINED = "raoulbia/Qwen2.5-7B-3GPP-NR"
SKIP_FINETUNE   = {"qwen25_7b"}   # ces modèles ne passent pas par QLoRA


# ============================================================
#  CALLBACK : Checkpoint + log progression
# ============================================================
class SaveProgressCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir    = output_dir
        self.progress_file = os.path.join(output_dir, "training_progress.json")
        self.start_time    = time.time()

    def on_save(self, args, state, control, **kwargs):
        progress = {
            "step":            state.global_step,
            "epoch":           round(state.epoch, 2),
            "best_loss":       state.best_metric,
            "elapsed_min":     round((time.time() - self.start_time) / 60, 1),
            "last_checkpoint": os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
        }
        with open(self.progress_file, "w") as f:
            json.dump(progress, f, indent=2)
        logger.info(f"Checkpoint sauvegardé step {state.global_step}")
        print(f"\n💾 Checkpoint : step {state.global_step} | epoch {round(state.epoch, 2)}", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            elapsed = round((time.time() - self.start_time) / 60, 1)
            print(f"   Step {state.global_step} | Loss: {logs['loss']:.4f} | {elapsed} min", flush=True)


# ============================================================
#  CHARGEMENT DATASET : raoulbia/3gpp-5g-nr-qa
# ============================================================
def _load_and_format_hf_dataset(force_redownload: bool = False):
    """Télécharge + formate le dataset. Renvoie [] si rien d'exploitable."""
    if force_redownload:
        ds = load_dataset("raoulbia/3gpp-5g-nr-qa", download_mode="force_redownload")
    else:
        ds = load_dataset("raoulbia/3gpp-5g-nr-qa")

    splits = list(ds.keys())
    print(f"   Splits disponibles : {splits}", flush=True)

    split_data = ds["train"] if "train" in splits else ds[splits[0]]
    cols       = split_data.column_names
    print(f"   Colonnes : {cols}", flush=True)
    print(f"   Exemples bruts : {len(split_data)}", flush=True)
    if len(split_data) > 0:
        print(f"   Aperçu   : {dict(split_data[0])}", flush=True)

    samples = []
    for item in split_data:

        # Récupérer la question depuis 'instruction' en priorité
        question = ""
        for col in ["instruction", "question", "Question", "prompt"]:
            if col in cols and item.get(col):
                question = str(item[col]).strip()
                break

        # Récupérer la réponse depuis 'output' en priorité
        answer = ""
        for col in ["output", "answer", "Answer", "response", "label"]:
            if col in cols and item.get(col):
                answer = str(item[col]).strip()
                # Nettoyer les tokens spéciaux résiduels
                answer = answer.replace("<|im_end|>", "").strip()
                answer = answer.replace("<|endoftext|>", "").strip()
                break

        # Récupérer le contexte depuis 'input' (si non vide)
        context = ""
        for col in ["input", "context", "Context", "explanation", "passage"]:
            if col in cols and item.get(col):
                val = str(item[col]).strip()
                if val:   # ignorer les chaînes vides
                    context = val
                    break

        if not question or not answer:
            continue

        context_block = f"\nContext: {context}" if context else ""

        text = f"""### Instruction:
You are a 3GPP 5G NR telecommunications expert. Answer the following question accurately.

Question: {question}{context_block}

### Response:
{answer}"""

        samples.append({"text": text})

    return samples


def load_training_data_hf():
    """
    Charge raoulbia/3gpp-5g-nr-qa depuis HuggingFace et le formate
    pour le fine-tuning. Si le premier essai renvoie 0 exemple
    (cache HF local corrompu/incomplet), retente une fois en
    forçant un re-téléchargement avant d'abandonner.
    """
    print("\n📥 Chargement dataset : raoulbia/3gpp-5g-nr-qa", flush=True)
    samples = _load_and_format_hf_dataset(force_redownload=False)

    if len(samples) == 0:
        print("⚠️  0 exemple formaté avec le cache local — "
              "tentative de re-téléchargement forcé du dataset...", flush=True)
        samples = _load_and_format_hf_dataset(force_redownload=True)

    print(f"✅ {len(samples)} exemples formatés", flush=True)
    logger.info(f"Dataset chargé : {len(samples)} exemples")
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
    print(f"📂 Checkpoint trouvé : {last}", flush=True)
    return last


# ============================================================
#  SAUVEGARDE QWEN (déjà fine-tuné — pas de QLoRA)
# ============================================================
def save_qwen_pretrained():
    """
    Qwen2.5-7B est déjà fine-tuné sur 3GPP par raoulbia.
    On le télécharge et sauvegarde localement pour usage uniforme.
    """
    output_dir  = os.path.join(MODELS_DIR, "qwen25_7b_finetuned")
    done_file   = os.path.join(output_dir, "TRAINING_DONE.json")
    legacy_file = os.path.join(output_dir, "TRAINING_DONE.txt")

    signature = compute_signature({
        "source":   QWEN_PRETRAINED,
        "device":   DEVICE,
        "use_4bit": USE_4BIT
    })

    # Compat : un ancien marqueur .txt vaut "déjà fait" une fois, puis migre vers .json
    migrate_legacy_marker(legacy_file, done_file, signature, {
        "model": QWEN_PRETRAINED, "migrated_from_txt": True
    })

    if is_marker_valid(done_file, signature, force=FORCE_RETRAIN):
        print(f"\n✅ Qwen déjà sauvegardé localement (config identique) : {output_dir}", flush=True)
        return output_dir

    if os.path.exists(output_dir):
        existing_sig = (read_marker(done_file) or {}).get("signature")
        if FORCE_RETRAIN:
            archive_stale_dir(output_dir, reason="FORCE_RETRAIN demandé")
        elif existing_sig and existing_sig != signature:
            archive_stale_dir(output_dir, reason="source ou device différents")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  Qwen2.5-7B : téléchargement depuis raoulbia/Qwen2.5-7B-3GPP-NR", flush=True)
    print(f"  (modèle déjà fine-tuné sur 3GPP — pas de QLoRA nécessaire)", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"📥 Chargement tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PRETRAINED, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"📥 Chargement modèle...", flush=True)
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_PRETRAINED,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_PRETRAINED,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

    print(f"💾 Sauvegarde locale...", flush=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    write_marker(done_file, signature, {
        "model": QWEN_PRETRAINED,
        "note": "Pré-entraîné sur 3GPP — pas de QLoRA nécessaire",
        "dataset": "raoulbia/3gpp-5g-nr-qa (déjà inclus dans l'entraînement original)"
    })

    print(f"✅ Qwen sauvegardé : {output_dir}", flush=True)
    logger.info(f"Qwen saved from {QWEN_PRETRAINED}")

    del model, tokenizer
    gc.collect()
    if USE_4BIT:
        torch.cuda.empty_cache()

    return output_dir


# ============================================================
#  FINE-TUNING QLORA (LLaMA + Mistral)
# ============================================================
def finetune_model(model_short, hf_model_name, samples):
    output_dir   = os.path.join(MODELS_DIR, f"{model_short}_finetuned")
    done_file    = os.path.join(output_dir, "TRAINING_DONE.json")
    legacy_file  = os.path.join(output_dir, "TRAINING_DONE.txt")
    run_sig_file = os.path.join(output_dir, "RUN_SIGNATURE.json")

    # La signature englobe tout ce qui, si ça change, doit invalider
    # un entraînement déjà fait : modèle, dataset (taille + empreinte
    # du premier/dernier exemple), hyperparamètres, device.
    signature = compute_signature({
        "model":          hf_model_name,
        "dataset_size":   len(samples),
        "dataset_first":  samples[0]["text"] if samples else "",
        "dataset_last":   samples[-1]["text"] if samples else "",
        "lora_r":         FT_CFG["lora_r"],
        "lora_alpha":     FT_CFG["lora_alpha"],
        "lora_dropout":   FT_CFG["lora_dropout"],
        "target_modules": FT_CFG["target_modules"],
        "epochs":         FT_CFG["num_train_epochs"],
        "lr":             FT_CFG["learning_rate"],
        "max_seq_length": FT_CFG["max_seq_length"],
        "device":         DEVICE
    })

    # Compat : un ancien marqueur .txt vaut "déjà fait" une fois, puis migre vers .json
    migrate_legacy_marker(legacy_file, done_file, signature, {
        "model": hf_model_name, "migrated_from_txt": True
    })

    if is_marker_valid(done_file, signature, force=FORCE_RETRAIN):
        print(f"\n✅ {model_short} déjà fine-tuné (config identique). Passage au suivant.", flush=True)
        return output_dir

    if os.path.exists(output_dir):
        existing_sig = (read_marker(done_file) or read_marker(run_sig_file) or {}).get("signature")
        if FORCE_RETRAIN:
            archive_stale_dir(output_dir, reason="FORCE_RETRAIN demandé")
        elif existing_sig and existing_sig != signature:
            archive_stale_dir(output_dir, reason="dataset ou hyperparamètres différents")
        # si existing_sig == signature mais done_file absent -> entraînement
        # interrompu avec la MÊME config -> on ne touche à rien, on reprend
        # normalement depuis le dernier checkpoint plus bas.

    os.makedirs(output_dir, exist_ok=True)
    write_marker(run_sig_file, signature, {"status": "in_progress", "model": hf_model_name})

    last_checkpoint = find_last_checkpoint(output_dir)

    if last_checkpoint:
        print(f"\n🔄 Reprise depuis : {last_checkpoint}", flush=True)
    else:
        print(f"\n🚀 Démarrage fine-tuning depuis zéro", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  Fine-tuning : {hf_model_name}", flush=True)
    print(f"  Output      : {output_dir}", flush=True)
    print(f"  Exemples    : {len(samples)}", flush=True)
    print(f"  Device      : {DEVICE.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    # Tokenizer
    print(f"📥 Chargement tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Modèle
    if USE_4BIT:
        print(f"📥 Chargement modèle en 4-bit (QLoRA GPU)...", flush=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print(f"📥 Chargement modèle float32 (CPU)...", flush=True)
        print(f"   ⚠️  Mode CPU — entraînement lent mais fonctionnel", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

    # LoRA
    lora_config = LoraConfig(
        r=FT_CFG["lora_r"],
        lora_alpha=FT_CFG["lora_alpha"],
        lora_dropout=FT_CFG["lora_dropout"],
        target_modules=FT_CFG["target_modules"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"✅ Paramètres entraînables : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)", flush=True)

    # Dataset
    dataset = Dataset.from_list(samples)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # Arguments entraînement
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=FT_CFG["num_train_epochs"],
        per_device_train_batch_size=1 if DEVICE == "cpu" else FT_CFG["per_device_train_batch_size"],
        gradient_accumulation_steps=16 if DEVICE == "cpu" else FT_CFG["gradient_accumulation_steps"],
        learning_rate=FT_CFG["learning_rate"],
        fp16=USE_4BIT,
        bf16=False,
        logging_steps=FT_CFG["logging_steps"],
        save_steps=FT_CFG["save_steps"],
        eval_steps=FT_CFG["eval_steps"],
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        warmup_ratio=FT_CFG["warmup_ratio"],
        lr_scheduler_type=FT_CFG["lr_scheduler_type"],
        report_to="none",
        dataloader_num_workers=0,
        optim="adamw_torch" if DEVICE == "cpu" else "paged_adamw_32bit",
        resume_from_checkpoint=last_checkpoint
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=FT_CFG["max_seq_length"],
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[SaveProgressCallback(output_dir)]
    )

    print(f"\n🔥 Début entraînement ({DEVICE.upper()})...", flush=True)
    start = time.time()
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print(f"\n💾 Sauvegarde modèle final...", flush=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    elapsed = round((time.time() - start) / 60, 1)

    write_marker(done_file, signature, {
        "elapsed_min": elapsed,
        "model":       hf_model_name,
        "dataset":     "raoulbia/3gpp-5g-nr-qa",
        "samples":     len(samples),
        "device":      DEVICE
    })

    print(f"\n✅ Fine-tuning {model_short} terminé en {elapsed} minutes !", flush=True)
    logger.info(f"{model_short} done in {elapsed} min | device={DEVICE}")

    del model, tokenizer
    gc.collect()
    if USE_4BIT:
        torch.cuda.empty_cache()

    return output_dir


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    print("\n" + "="*60, flush=True)
    print("  3GPP FINE-TUNING QLORA", flush=True)
    print(f"  Device  : {DEVICE.upper()}", flush=True)
    print(f"  Dataset : raoulbia/3gpp-5g-nr-qa", flush=True)
    print("="*60, flush=True)

    if DEVICE == "cpu":
        print("\n⚠️  Mode CPU détecté — entraînement lent mais fonctionnel", flush=True)

    # ── ETAPE 1 : Qwen → sauvegarde locale du modèle pré-entraîné ──
    print(f"\n{'='*60}", flush=True)
    print(f"  [1/3] QWEN2.5-7B (pré-entraîné 3GPP)", flush=True)
    print(f"{'='*60}", flush=True)
    save_qwen_pretrained()

    # ── ETAPE 2 : Charger dataset pour LLaMA + Mistral ──────────────
    print(f"\n📖 Chargement dataset HuggingFace pour LLaMA + Mistral...", flush=True)
    samples = load_training_data_hf()

    if len(samples) == 0:
        print("❌ Aucun exemple chargé même après re-téléchargement forcé !", flush=True)
        print("   Vérifier la connectivité HuggingFace et le nom du dataset.", flush=True)
        exit(1)

    # ── ETAPE 3 : LLaMA ─────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  [2/3] LLAMA 3.1-8B (QLoRA fine-tuning)", flush=True)
    print(f"{'='*60}", flush=True)
    finetune_model("llama31_8b", HF_MODELS["llama31_8b"], samples)

    # ── ETAPE 4 : Mistral ───────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  [3/3] MISTRAL-7B (QLoRA fine-tuning)", flush=True)
    print(f"{'='*60}", flush=True)
    finetune_model("mistral_7b", HF_MODELS["mistral_7b"], samples)

    print("\n" + "="*60, flush=True)
    print("  ✅ TOUS LES MODÈLES PRÊTS !", flush=True)
    print(f"  Modèles sauvegardés dans : {MODELS_DIR}", flush=True)
    print(f"\n  Étape suivante : python 06_ft_rag.py", flush=True)
    print("="*60, flush=True)
