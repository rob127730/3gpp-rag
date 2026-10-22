"""
RAGAS evaluation of the 3GPP RAG pipeline (local judge via Ollama, no API key).

STATUS: written but NOT yet run. Smoke-test first:  python 10_ragas_eval.py --limit 5

Why a re-run is needed: the stored result files contain no retrieved contexts, and
RAGAS needs (question, answer, contexts, reference). This script re-retrieves the
contexts, generates a free-form answer, then scores it.

Metrics: faithfulness, answer_relevancy, context_precision, context_recall.
Caveats: LLM-as-judge with a small local model is noisy; TeleQnA is multiple-choice,
so the reference is the correct option text; report scores with n and judge model.
Run from the project root (needs chroma_db/ and Ollama running).
"""
import argparse, json, os, sys, time
import requests, yaml

sys.path.insert(0, os.path.dirname(__file__))
import importlib
opt = importlib.import_module("03_rag_optimized")  # reuse retrieval code


def ollama_answer(model, question, context):
    prompt = ("Use ONLY the 3GPP context to answer the question in 1-3 sentences.\n\n"
              f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:")
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False,
                            "options": {"temperature": 0.1, "num_predict": 200}}, timeout=180)
    return r.json().get("response", "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--pipeline", choices=["baseline", "optimized"], default="optimized")
    ap.add_argument("--gen-model", default="qwen2.5:7b")
    ap.add_argument("--judge-model", default="qwen2.5:7b")
    ap.add_argument("--out", default="results/ragas_{pipeline}.json")
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer, CrossEncoder
    import chromadb
    from rank_bm25 import BM25Okapi

    questions = opt.load_teleqna_questions(opt.TEST_FILE, max_questions=a.limit)
    emb = SentenceTransformer(opt.EMB_MODEL, device=opt.DEVICE)
    col = chromadb.PersistentClient(path=opt.CHROMA_DIR).get_collection(opt.COLL_NAME)
    if a.pipeline == "optimized":
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=opt.DEVICE)
        chunks = json.load(open("bm25_chunks_cache.json", encoding="utf-8"))
        bm25 = BM25Okapi([c.lower().split() for c in chunks])

    rows = []
    for q in questions:
        if a.pipeline == "optimized":
            cands = opt.hybrid_retrieve(q["question"], col, emb, bm25, chunks, opt.TOP_K_RETRIEVE)
            ctx_text = opt.rerank(q["question"], cands, reranker, opt.TOP_K_RERANK)
            ctxs = [c for c in ctx_text.split("\n\n---\n\n") if c.strip()]
        else:
            res = col.query(query_embeddings=emb.encode([q["question"]], normalize_embeddings=True).tolist(),
                            n_results=5, include=["documents"])
            ctxs = res["documents"][0]
            ctx_text = "\n\n---\n\n".join(ctxs)
        ans = ollama_answer(a.gen_model, q["question"], ctx_text)
        ref = q["options"].get(q["answer"], "") if q["answer"] else ""
        rows.append({"user_input": q["question"], "response": ans,
                     "retrieved_contexts": [c[:1000] for c in ctxs], "reference": ref})
        print(f"[{len(rows)}/{len(questions)}] done", flush=True)

    from ragas import evaluate, EvaluationDataset
    from ragas.run_config import RunConfig
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    llm = LangchainLLMWrapper(ChatOllama(model=a.judge_model, temperature=0, num_ctx=4096))
    emb_j = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))  # ollama pull nomic-embed-text
    res = evaluate(EvaluationDataset.from_list(rows),
                   metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                   llm=llm, embeddings=emb_j,
                   run_config=RunConfig(timeout=900, max_workers=1, max_retries=2))
    df = res.to_pandas()
    summary = {k: float(df[k].mean()) for k in df.columns
               if k in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}
    out = a.out.format(pipeline=a.pipeline)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"pipeline": a.pipeline, "n": len(rows), "gen_model": a.gen_model,
               "judge_model": a.judge_model, "scores": summary, "date": time.strftime("%Y-%m-%d")},
              open(out, "w", encoding="utf-8"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
