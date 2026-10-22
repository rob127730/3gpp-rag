# ============================================================
#  01_indexer_fast.py
#  Version RAPIDE - GPU + batch embeddings
#  Objectif : indexer 15000 docs en 15-30 min au lieu de 30h
# ============================================================

import os
import glob
import yaml
import chromadb
import torch
import numpy as np
import numpy as np
from tqdm import tqdm
from loguru import logger
from sentence_transformers import SentenceTransformer

# --- Charger la config ---
with open(os.environ.get("RAG_CONFIG", "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

DOCS_DIR   = config["paths"]["docs_dir"]
CHROMA_DIR = config["paths"]["chroma_db"]

# On utilise le modèle baseline uniquement pour l'instant
EMB_MODEL  = config["rag_baseline"]["embedding_model"]   # BAAI/bge-small-en-v1.5
CHUNK_SIZE = config["rag_baseline"]["chunk_size"]        # 512
CHUNK_OVER = config["rag_baseline"]["chunk_overlap"]     # 50
COLL_NAME  = config["rag_baseline"]["collection_name"]   # 3gpp_baseline

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
logger.add("logs/indexer_fast.log", rotation="10 MB")

# Détection GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n  Device détecté : {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"   GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM : {round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB")


# ============================================================
#  FONCTION : Découper un texte en chunks
# ============================================================
def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:  # ignorer les chunks trop courts
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ============================================================
#  FONCTION : Lire tous les fichiers .md
# ============================================================
def load_all_md_files(docs_dir):
    pattern_rel17 = os.path.join(docs_dir, "Rel-17", "**", "*.md")
    pattern_rel18 = os.path.join(docs_dir, "Rel-18", "**", "*.md")
    files = glob.glob(pattern_rel17, recursive=True) + glob.glob(pattern_rel18, recursive=True)
    print(f"\n {len(files)} fichiers .md trouvés")
    logger.info(f"Fichiers .md : {len(files)}")

    documents = []
    for filepath in tqdm(files, desc=" Lecture fichiers"):
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
            if len(content) > 100:  # ignorer fichiers quasi vides
                parts = filepath.replace("\\", "/").split("/")
                release = next((p for p in parts if p.startswith("Rel-")), "Unknown")
                series  = next((p for p in parts if "_series" in p), "Unknown")
                documents.append({
                    "path":     filepath,
                    "content":  content,
                    "release":  release,
                    "series":   series,
                    "filename": os.path.basename(filepath)
                })
        except Exception as e:
            logger.warning(f"Erreur {filepath} : {e}")

    print(f" {len(documents)} documents valides chargés")
    return documents


# ============================================================
#  FONCTION : Construire tous les chunks en mémoire
# ============================================================
def build_chunks(documents):
    print(f"\n  Découpage en chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVER})...")
    all_ids       = []
    all_texts     = []
    all_metadatas = []

    for doc in tqdm(documents, desc="  Chunking"):
        chunks = chunk_text(doc["content"], CHUNK_SIZE, CHUNK_OVER)
        for i, chunk in enumerate(chunks):
            # ID unique et court pour éviter les doublons
            chunk_id = f"{doc['path'].replace(chr(92), '/').replace(':', '')}|{i}"
            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_metadatas.append({
                "release":  doc["release"],
                "series":   doc["series"],
                "filename": doc["filename"],
                "chunk_id": i
            })

    print(f" {len(all_texts)} chunks créés au total")
    return all_ids, all_texts, all_metadatas


# ============================================================
#  FONCTION : Calculer TOUS les embeddings en batch GPU
# ============================================================
def compute_embeddings_gpu(texts, model, batch_size=256):
    """
    Calcule tous les embeddings en une seule passe GPU.
    Batch size 256 = optimal pour 8GB VRAM avec bge-small.
    """
    print(f"\n Calcul embeddings sur {DEVICE.upper()} (batch={batch_size})...")
    print(f"   Modèle : {EMB_MODEL}")
    print(f"   Nombre de textes : {len(texts)}")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        device=DEVICE,
        normalize_embeddings=True,   # important pour cosine similarity
        convert_to_numpy=True
    )

    print(f" Embeddings calculés : shape {embeddings.shape}")
    return embeddings


# ============================================================
#  FONCTION : Insérer dans ChromaDB par gros batch
# ============================================================
def insert_into_chroma(client, collection_name, all_ids, all_texts, all_metadatas, embeddings):
    """
    Insère tout dans ChromaDB avec les embeddings précalculés.
    Batch de 5000 pour aller vite.
    """
    print(f"\n Insertion dans ChromaDB...")

    # Créer la collection SANS embedding function (on passe les embeddings directement)
    try:
        client.delete_collection(collection_name)
        print(f"   Collection existante supprimée : {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    BATCH = 5000
    total = len(all_ids)

    for start in tqdm(range(0, total, BATCH), desc=" Insertion ChromaDB"):
        end = min(start + BATCH, total)
        collection.add(
            ids=all_ids[start:end],
            documents=all_texts[start:end],
            metadatas=all_metadatas[start:end],
            embeddings=embeddings[start:end].tolist()
        )

    count = collection.count()
    print(f" Collection '{collection_name}' : {count} chunks insérés")
    logger.info(f"Collection {collection_name} : {count} chunks")
    return collection


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    print("\n" + "="*60)
    print("  3GPP RAG - INDEXATION RAPIDE (GPU)")
    print("="*60)

    # 1. Charger les fichiers
    documents = load_all_md_files(DOCS_DIR)
    if not documents:
        print(" Aucun document trouvé dans", DOCS_DIR)
        exit(1)

    # 2. Découper en chunks
    all_ids, all_texts, all_metadatas = build_chunks(documents)

    # 3. Charger le modèle d'embedding sur GPU
    print(f"\n Chargement modèle embedding : {EMB_MODEL}")
    model = SentenceTransformer(EMB_MODEL, device=DEVICE)
    print(f" Modèle chargé sur {DEVICE.upper()}")

    # 4. Calculer TOUS les embeddings en batch GPU
    CACHE_FILE = "embeddings_cache.npy"
    if os.path.exists(CACHE_FILE):
        print("📂 Embeddings trouvés en cache, chargement...")
        embeddings = np.load(CACHE_FILE)
        print(f"✅ Cache chargé : {embeddings.shape}")
    else:
        embeddings = compute_embeddings_gpu(all_texts, model, batch_size=64)
        np.save(CACHE_FILE, embeddings)
        print("💾 Embeddings sauvegardés dans embeddings_cache.npy")

    # 5. Libérer la mémoire GPU après embeddings
    del model
    torch.cuda.empty_cache()
    print(" Mémoire GPU libérée")

    # 6. Insérer dans ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    insert_into_chroma(
        client, COLL_NAME,
        all_ids, all_texts, all_metadatas, embeddings
    )

    print("\n" + "="*60)
    print("   INDEXATION TERMINÉE !")
    print(f"  Base ChromaDB : {CHROMA_DIR}")
    print(f"  Collection    : {COLL_NAME}")
    print(f"  Chunks totaux : {len(all_texts)}")
    print("\n  Prochaine étape : python 02_rag_baseline.py")
    print("="*60)
