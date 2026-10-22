# ============================================================
#  08_agent_api.py
#  API REST FastAPI — 3GPP Agent IA
#
#  Agent 1 : Ollama + ChromaDB similarity (rapide)
#  Agent 2 : Ollama + ChromaDB MMR (rapide)
#  Agent 3 : Ollama×2 + Arbitre Mistral HF local + Web search
#
#  Lancer :
#    ollama serve                        (terminal séparé)
#    uvicorn 08_agent_api:app --port 8000
#  Docs : http://localhost:8000/docs
# ============================================================

import os
import time
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from agents_core import (
    load_vectorstore, load_hf_arbitre,
    build_agents, extract_option
)

print("🚀 Démarrage API 3GPP Agent IA...", flush=True)

with open(os.environ.get("RAG_CONFIG", "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
RESULTS_DIR = config["paths"]["results_dir"]
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── INITIALISATION (une seule fois au démarrage) ─────────────
collection, emb_model = load_vectorstore()

print("\n📥 Chargement arbitre Mistral local (Agent 3)...", flush=True)
hf_model, hf_tokenizer = load_hf_arbitre()

agents = build_agents(collection, emb_model, hf_model, hf_tokenizer)

# ── FASTAPI ──────────────────────────────────────────────────
app = FastAPI(
    title="3GPP Agent IA",
    description="""
Agent IA spécialisé 3GPP 5G NR :

- **Agent 1** : Ollama (Mistral) + ChromaDB similarity — rapide
- **Agent 2** : Ollama (Mistral) + ChromaDB MMR — plus précis
- **Agent 3** : Agent1 + Agent2 + Arbitre Mistral local + 🌐 Web search si désaccord
""",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ── SCHEMAS ──────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    agent: Optional[str] = "agent3"       # agent1 | agent2 | agent3
    show_details: Optional[bool] = False   # pour agent3 : montre accord/web


class AskResponse(BaseModel):
    question: str
    answer: str
    extracted_option: str
    agent_used: str
    response_time_s: float
    details: Optional[dict] = None


# ── ENDPOINTS ────────────────────────────────────────────────
@app.get("/", summary="Accueil")
def root():
    return {
        "message": "3GPP Agent IA — API REST",
        "docs":    "http://localhost:8000/docs",
        "agents": {
            "agent1": agents["agent1"]["name"],
            "agent2": agents["agent2"]["name"],
            "agent3": agents["agent3"]["name"]
        }
    }


@app.get("/health", summary="État de l'API")
def health():
    import requests as req
    try:
        ollama_ok = req.get(
            "http://localhost:11434/api/tags", timeout=3
        ).status_code == 200
    except Exception:
        ollama_ok = False

    return {
        "status":          "ok",
        "ollama":          "✅ disponible" if ollama_ok else "❌ non disponible",
        "chromadb_chunks": collection.count(),
        "mistral_local":   "✅ chargé" if hf_model is not None else "❌ non chargé",
        "agents":          list(agents.keys())
    }


@app.get("/agents", summary="Description des agents")
def list_agents():
    return {
        k: {
            "name":        v["name"],
            "description": {
                "agent1": "Ollama Mistral + ChromaDB similarity search (top_k chunks)",
                "agent2": "Ollama Mistral + ChromaDB MMR search (diversité chunks)",
                "agent3": "Agent1 + Agent2 + Arbitre Mistral HF local + Web search si désaccord"
            }.get(k, "")
        }
        for k, v in agents.items()
    }


@app.post("/ask", response_model=AskResponse, summary="Poser une question 3GPP")
def ask(req: AskRequest):
    """
    Pose une question à l'agent 3GPP choisi.

    - **agent1** : réponse rapide via Ollama + RAG basique
    - **agent2** : réponse Ollama + RAG avancé MMR
    - **agent3** : meilleure réponse — si Agent1 ≠ Agent2,
                   cherche sur le web et arbitre avec Mistral local
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question vide.")

    key = req.agent.lower()
    if key not in agents:
        raise HTTPException(
            status_code=400,
            detail=f"Agent inconnu : {key}. Choisir : agent1, agent2, agent3"
        )

    start   = time.time()
    details = None

    if key == "agent3" and req.show_details:
        result  = agents["agent3"]["fn_detail"](req.question)
        answer  = result["answer"]
        details = {
            "agent1_answer":   result["agent1"],
            "agent2_answer":   result["agent2"],
            "agents_agreed":   result["agreed"],
            "web_search_used": result["web_used"],
            "web_query":       result["web_query"],
            "web_snippet":     result["web_context"]
        }
    else:
        answer = agents[key]["fn"](req.question)

    elapsed = round(time.time() - start, 2)

    return AskResponse(
        question=req.question,
        answer=answer,
        extracted_option=extract_option(answer),
        agent_used=agents[key]["name"],
        response_time_s=elapsed,
        details=details
    )


@app.post("/compare", summary="Comparer les 3 agents sur une question")
def compare(question: str):
    """
    Compare les réponses des 3 agents sur la même question.
    Utile pour voir où ils s'accordent ou divergent.
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question vide.")

    results = {}

    # Agent 1 et 2 via Ollama (rapides)
    for key in ["agent1", "agent2"]:
        start = time.time()
        answer = agents[key]["fn"](question)
        results[key] = {
            "agent":           agents[key]["name"],
            "answer":          answer,
            "option":          extract_option(answer),
            "response_time_s": round(time.time() - start, 2)
        }

    # Agent 3 avec détails complets
    start  = time.time()
    detail = agents["agent3"]["fn_detail"](question)
    results["agent3"] = {
        "agent":           agents["agent3"]["name"],
        "answer":          detail["answer"],
        "option":          extract_option(detail["answer"]),
        "agent1_said":     detail["agent1"],
        "agent2_said":     detail["agent2"],
        "agreed":          detail["agreed"],
        "web_used":        detail["web_used"],
        "web_snippet":     detail["web_context"][:150] if detail["web_context"] else None,
        "response_time_s": round(time.time() - start, 2)
    }

    return {
        "question": question,
        "results":  results,
        "consensus": extract_option(detail["answer"])
    }


@app.get("/search", summary="Recherche RAG seule (sans LLM)")
def search_only(question: str, top_k: int = 3):
    """
    Cherche les chunks les plus pertinents dans ChromaDB
    sans passer par le LLM — utile pour debug.
    """
    from agents_core import retrieve_context
    context = retrieve_context(question, collection, emb_model, top_k)
    return {
        "question": question,
        "top_k":    top_k,
        "context":  context[:2000]
    }


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  3GPP AGENT IA — FastAPI REST")
    print(f"  ChromaDB : {collection.count()} chunks")
    print(f"  Docs     : http://localhost:8000/docs")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
