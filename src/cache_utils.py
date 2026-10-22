# ============================================================
#  cache_utils.py
#  Utilitaires de cache "intelligent" partagés par les 3 scripts
#  (04_finetune_qlora.py, 06_ft_rag.py, 07_raft_fast.py)
#
#  PROBLEME RESOLU :
#  Avant, un cache (dataset généré, modèle déjà fine-tuné,
#  résultats d'évaluation déjà calculés) était réutilisé dès
#  qu'un fichier existait — peu importe s'il avait été produit
#  avec une configuration différente de la configuration
#  actuelle (autre taille de dataset, autre prompt, autre
#  modèle ré-entraîné entre temps, etc.). C'est ce qui causait
#  des résultats silencieusement faux ou périmés.
#
#  PRINCIPE :
#  - on calcule une "signature" (hash court) à partir de TOUS
#    les paramètres qui comptent pour un calcul donné (nom du
#    modèle, taille du dataset, hyperparamètres, version du
#    prompt, etc.)
#  - cette signature est stockée à côté du résultat (.json)
#  - au prochain run, si la signature stockée ne correspond
#    pas à la signature attendue, le cache est considéré
#    PÉRIMÉ : on l'archive (jamais supprimé silencieusement)
#    et on recalcule.
#  - les signatures peuvent être "chaînées" : la signature
#    d'une évaluation inclut la signature de l'entraînement
#    dont elle dépend, donc si on ré-entraîne, l'éval se
#    réinvalide automatiquement.
#
#  VARIABLES D'ENVIRONNEMENT :
#  - FORCE_REFRESH=1   -> ignore les caches "légers" (dataset
#                         généré, résultats d'évaluation)
#  - FORCE_RETRAIN=1   -> ignore aussi les caches "lourds"
#                         (modèles déjà fine-tunés / RAFT),
#                         donc relance un entraînement complet
# ============================================================

import os
import json
import time
import shutil
import hashlib


def env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes"}


FORCE_REFRESH = env_flag("FORCE_REFRESH")   # caches légers (dataset, résultats d'éval)
FORCE_RETRAIN = env_flag("FORCE_RETRAIN")   # caches lourds (entraînements complets)


def compute_signature(payload: dict) -> str:
    """Hash court et stable d'un dict de paramètres de run."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def write_marker(path: str, signature: str, meta: dict = None):
    payload = {"signature": signature, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if meta:
        payload.update(meta)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_marker(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def is_marker_valid(path: str, signature: str, force: bool = False) -> bool:
    """True si un marqueur existe et correspond exactement à la signature attendue."""
    if force:
        return False
    marker = read_marker(path)
    return bool(marker) and marker.get("signature") == signature


def migrate_legacy_marker(old_path: str, new_path: str, signature: str, meta: dict = None) -> bool:
    """
    Compat ascendante : si un ancien marqueur texte (ex: TRAINING_DONE.txt)
    existe sans équivalent JSON, on le considère valide une seule fois
    (pour ne pas relancer un entraînement déjà terminé juste parce qu'on
    a changé le format des marqueurs), et on écrit le nouveau marqueur
    JSON pour que les prochains runs utilisent la validation par signature.
    """
    if os.path.exists(old_path) and not os.path.exists(new_path):
        print(f"  ℹ️  Ancien marqueur détecté ({os.path.basename(old_path)}) "
              f"— migration vers {os.path.basename(new_path)}", flush=True)
        write_marker(new_path, signature, meta or {})
        return True
    return False


def archive_stale_dir(path: str, reason: str = "configuration changée"):
    """Renomme un dossier périmé au lieu de le supprimer (rien n'est jamais perdu)."""
    if not os.path.exists(path):
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archived = f"{path}_stale_{stamp}"
    print(f"  ⚠️  Cache périmé ({reason}) — archivage : "
          f"{os.path.basename(path)} → {os.path.basename(archived)}", flush=True)
    shutil.move(path, archived)


def load_json_cache(path: str, expected_signature: str = None, force: bool = False):
    """
    Charge un cache JSON (dict contenant une clé '_signature').
    Retourne None si absent, illisible, forcé à ignorer, ou périmé
    (signature différente de celle attendue).
    """
    if force or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if expected_signature is not None:
        cached_sig = data.get("_signature") if isinstance(data, dict) else None
        if cached_sig != expected_signature:
            print(f"  ⚠️  Cache périmé détecté ({path}) — signature différente, recalcul", flush=True)
            return None
    return data


def save_json_cache(path: str, data: dict, signature: str):
    """Sauvegarde un cache JSON en y injectant sa signature."""
    payload = dict(data)
    payload["_signature"] = signature
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
