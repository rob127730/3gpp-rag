# ============================================================
#  agents_core.py
#  3 Agents 3GPP — Architecture mixte
#
#  Agent 1 : Simple     → Ollama (rapide) + ChromaDB similarity
#  Agent 2 : RAG Avancé → Ollama (rapide) + ChromaDB MMR
#  Agent 3 : Multi-Agent→ Agent1 + Agent2 + Arbitre HuggingFace local
#                         + Web search si désaccord
#
#  Ollama : rapide (~5-10s/question)
#  HuggingFace local : Mistral fine-tuné (plus précis pour arbitrage)
# ============================================================

import os
import re
import gc
import yaml
import time
import requests
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, pipeline as hf_pipeline
)
from peft import PeftModel

with open(os.environ.get("RAG_CONFIG", "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

CHROMA_DIR  = config["paths"]["chroma_db"]
COLL_NAME   = config["rag_baseline"]["collection_name"]
EMB_MODEL   = config["rag_baseline"]["embedding_model"]
TOP_K       = config["rag_baseline"]["top_k"]
TOP_K_ADV   = 6

# ── Ollama ───────────────────────────────────────────────────
OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "mistral:7b-instruct-q4_0"
OLLAMA_TIMEOUT = 120

# ── HuggingFace local (Agent 3 arbitre) ─────────────────────
MISTRAL_LOCAL = os.environ.get("MISTRAL_LOCAL", "./models/mistral_7b_finetuned")
MISTRAL_BASE  = "mistralai/Mistral-7B-Instruct-v0.3"

# ── Web search ───────────────────────────────────────────────
WEB_SNIPPET_LIMIT    = 400
WEB_MAX_RESULTS      = 3

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_4BIT = torch.cuda.is_available()

CHUNK_LIMIT = 1200   # chars max contexte RAG


# ============================================================
#  UTILITAIRES
# ============================================================
def extract_option(text: str) -> str:
    t = text.lower().strip()
    for i in range(1, 6):
        if f"option {i}" in t:
            return f"option {i}"
    for i in range(1, 6):
        if t.startswith(str(i)):
            return f"option {i}"
    return t


def options_agree(a1: str, a2: str) -> bool:
    o1, o2 = extract_option(a1), extract_option(a2)
    return o1 == o2 and o1.startswith("option")


# ============================================================
#  WEB SEARCH (DuckDuckGo gratuit)
# ============================================================
def web_search(query: str) -> str:
    # Essai Tavily d'abord si clé dispo
    try:
        from tavily import TavilyClient
        key = os.environ.get("TAVILY_API_KEY", "")
        if key:
            client  = TavilyClient(api_key=key)
            results = client.search(query=query, max_results=WEB_MAX_RESULTS)
            snippets = [r.get("content", "")[:WEB_SNIPPET_LIMIT]
                        for r in results.get("results", []) if r.get("content")]
            if snippets:
                return "\n\n".join(snippets)
    except Exception:
        pass

    # Fallback DuckDuckGo
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; 3GPP-Agent/1.0)"}
        url     = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        resp    = requests.get(url, headers=headers, timeout=10)
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</a>', resp.text, re.DOTALL
        )
        clean = []
        for s in snippets[:WEB_MAX_RESULTS]:
            s = re.sub(r'<[^>]+>', '', s).strip()
            s = re.sub(r'\s+', ' ', s)
            if s:
                clean.append(s[:WEB_SNIPPET_LIMIT])
        return "\n\n".join(clean)
    except Exception as e:
        print(f"  ⚠️  Web search échoué : {e}", flush=True)
        return ""


# ============================================================
#  CHROMADB + EMBEDDINGS
# ============================================================
def load_vectorstore():
    print(f"\n🗄️  Connexion ChromaDB : {CHROMA_DIR}", flush=True)
    emb_model   = SentenceTransformer(EMB_MODEL, device=DEVICE)
    client      = chromadb.PersistentClient(path=CHROMA_DIR)
    collection  = client.get_collection(name=COLL_NAME)
    n = collection.count()
    print(f"   ✅ ChromaDB — {n} chunks | Embedding : {EMB_MODEL}", flush=True)
    return collection, emb_model


def retrieve_context(question: str, collection, emb_model, top_k: int) -> str:
    emb = emb_model.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    )
    results = collection.query(
        query_embeddings=emb.tolist(),
        n_results=top_k,
        include=["documents"]
    )
    chunks = [c[:CHUNK_LIMIT] for c in results["documents"][0]]
    return "\n\n---\n\n".join(chunks)


def retrieve_context_mmr(question: str, collection, emb_model, top_k: int) -> str:
    """MMR simplifié : récupère plus de chunks puis garde les plus diversifiés."""
    emb = emb_model.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    )
    results = collection.query(
        query_embeddings=emb.tolist(),
        n_results=top_k * 3,   # fetch plus large
        include=["documents", "distances"]
    )
    all_chunks   = results["documents"][0]
    all_distances = results["distances"][0]

    # Garder les top_k les plus pertinents parmi fetch_k
    sorted_chunks = [c for _, c in sorted(
        zip(all_distances, all_chunks), key=lambda x: x[0]
    )][:top_k]

    return "\n\n---\n\n".join([c[:CHUNK_LIMIT] for c in sorted_chunks])


# ============================================================
#  OLLAMA : Agents 1 et 2
# ============================================================
def ask_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 32,
            "top_p":       0.9
        }
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama non disponible. Lancez : ollama serve")
    except Exception as e:
        return f"error: {e}"


# ============================================================
#  HUGGINGFACE LOCAL : Agent 3 (arbitre)
# ============================================================
def load_hf_arbitre():
    """
    Charge Mistral fine-tuné local pour l'arbitrage Agent 3.
    Chargé une seule fois au démarrage.
    """
    print(f"\n📥 Chargement arbitre HuggingFace : {MISTRAL_LOCAL}", flush=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MISTRAL_LOCAL, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            MISTRAL_LOCAL, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if USE_4BIT:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        if os.path.exists(os.path.join(MISTRAL_LOCAL, "adapter_config.json")):
            base = AutoModelForCausalLM.from_pretrained(
                MISTRAL_BASE, quantization_config=bnb,
                device_map="auto", trust_remote_code=True,
                torch_dtype=torch.float16, low_cpu_mem_usage=True)
            model = PeftModel.from_pretrained(base, MISTRAL_LOCAL)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MISTRAL_LOCAL, quantization_config=bnb,
                device_map="auto", trust_remote_code=True,
                torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MISTRAL_LOCAL, device_map="cpu", trust_remote_code=True,
            torch_dtype=torch.float32, low_cpu_mem_usage=True)

    model.eval()
    print("   ✅ Arbitre HuggingFace prêt", flush=True)
    return model, tokenizer


def ask_hf_arbitre(model, tokenizer, prompt: str, max_new_tokens: int = 32) -> str:
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ============================================================
#  CONSTRUCTION DES 3 AGENTS
# ============================================================
def build_agents(collection, emb_model, hf_model=None, hf_tokenizer=None):
    """
    Construit les 3 agents.
    hf_model / hf_tokenizer : arbitre HuggingFace pour Agent 3
    Si non fournis, Agent 3 utilisera Ollama comme arbitre.
    """

    # ── Agent 1 : Ollama + similarity search ────────────────
    def run_agent1(question: str) -> str:
        try:
            context = retrieve_context(question, collection, emb_model, TOP_K)
            prompt  = (
                f"You are a 3GPP 5G NR expert.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer with ONLY the option number (e.g., 'option 1'):"
            )
            return ask_ollama(prompt)
        except Exception as e:
            return f"error: {e}"

    # ── Agent 2 : Ollama + MMR search ───────────────────────
    def run_agent2(question: str) -> str:
        try:
            context = retrieve_context_mmr(question, collection, emb_model, TOP_K_ADV)
            prompt  = (
                f"You are an expert in 3GPP 5G NR technical specifications.\n\n"
                f"Analyze the following technical context carefully.\n\n"
                f"Technical Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Instructions:\n"
                f"1. Read the context carefully\n"
                f"2. Identify the relevant 3GPP technical information\n"
                f"3. Select the most accurate option\n\n"
                f"Answer with ONLY the option number (e.g., 'option 1'):"
            )
            return ask_ollama(prompt)
        except Exception as e:
            return f"error: {e}"

    # ── Agent 3 : Agent1 + Agent2 + Arbitre HF + Web ────────
    def run_agent3(question: str) -> dict:
        a1 = run_agent1(question)
        a2 = run_agent2(question)

        web_used    = False
        web_context = ""
        web_query   = ""

        if options_agree(a1, a2):
            # Accord → confirmer avec l'arbitre
            arb_prompt = (
                f"You are a 3GPP expert. Both agents agree.\n\n"
                f"Question: {question}\n"
                f"Agent 1: {a1}\nAgent 2: {a2}\n\n"
                f"Confirm with ONLY the option number:"
            )
        else:
            # Désaccord → web search puis arbitre HF
            web_used  = True
            web_query = f"3GPP 5G NR {question[:100]}"
            print(f"\n  🌐 Désaccord — web search: {web_query[:70]}...", flush=True)
            web_context = web_search(web_query)

            if web_context:
                print(f"  ✅ Web context ({len(web_context)} chars)", flush=True)
            else:
                print(f"  ⚠️  Web search vide", flush=True)

            arb_prompt = (
                f"You are a 3GPP 5G NR expert arbitrating between two agents.\n\n"
                f"Question: {question}\n\n"
                f"Agent 1: {a1}\nAgent 2: {a2}\n\n"
                f"Web context:\n{web_context[:600]}\n\n"
                f"Based on agents AND web context, give the most accurate answer.\n"
                f"Answer with ONLY the option number (e.g., 'option 1'):"
            )

        # Arbitre : HuggingFace local si disponible, sinon Ollama
        if hf_model is not None and hf_tokenizer is not None:
            final = ask_hf_arbitre(hf_model, hf_tokenizer, arb_prompt)
        else:
            final = ask_ollama(arb_prompt)

        return {
            "answer":      final,
            "agent1":      a1,
            "agent2":      a2,
            "agreed":      options_agree(a1, a2),
            "web_used":    web_used,
            "web_query":   web_query,
            "web_context": web_context[:200] if web_context else ""
        }

    def run_agent3_simple(question: str) -> str:
        return run_agent3(question)["answer"]

    agents = {
        "agent1": {
            "name":      "Agent 1 — Ollama + ChromaDB similarity",
            "fn":        run_agent1,
            "fn_detail": run_agent1
        },
        "agent2": {
            "name":      "Agent 2 — Ollama + ChromaDB MMR avancé",
            "fn":        run_agent2,
            "fn_detail": run_agent2
        },
        "agent3": {
            "name":      "Agent 3 — Multi-Agent (Ollama×2 + Arbitre HF + 🌐 Web)",
            "fn":        run_agent3_simple,
            "fn_detail": run_agent3
        }
    }

    print("\n✅ Les 3 agents construits !", flush=True)
    print("   Agent 1 : Ollama + similarity search", flush=True)
    print("   Agent 2 : Ollama + MMR search", flush=True)
    print("   Agent 3 : Ollama×2 + Arbitre Mistral local + 🌐 Web si désaccord", flush=True)
    return agents
