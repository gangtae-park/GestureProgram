"""Fixed object database (3 objects for the CHI 2027 study).

On startup we:
  1. Read objects.json (id, name, description, typical_use, info).
  2. Scan object_db/images/<id>/ for reference images (any .jpg/.png).
  3. Build CLIP image embeddings for each reference image and stack them into a
     single (N, D) matrix for fast cosine-similarity lookup. The embeddings are
     cached to object_db/embeddings.npz so subsequent launches skip the GPU pass
     unless an image is added/changed (we compare mtimes).

At query time, the Search/Find Info handler calls:
    db = get_db()
    match = clip_matcher.match_against_db(query_emb, db)

and forwards db.get_object(match["object"]["id"]) verbatim to Unity.

No GPT calls -- this whole module is pure local lookup.
"""
import json
import os
import time

import cv2
import numpy as np

from . import config


_db = None  # populated by load_object_db()


def get_db():
    return _db


class ObjectDB:
    def __init__(self, objects: list, embeddings: np.ndarray,
                 embedding_object_ids: list, source_paths: list):
        self.objects = objects
        self._by_id = {o["id"]: o for o in objects}
        # (N, D) float32, rows already L2-normalised so cosine = dot product.
        self.embedding_matrix = embeddings
        self.embedding_object_ids = embedding_object_ids  # length N
        self.source_paths = source_paths                  # length N

    def get_object(self, obj_id: str) -> dict:
        return self._by_id.get(obj_id)

    def summary(self) -> str:
        per_obj_counts = {oid: 0 for oid in self._by_id.keys()}
        for oid in self.embedding_object_ids:
            per_obj_counts[oid] = per_obj_counts.get(oid, 0) + 1
        return ", ".join(f"{oid}:{n}" for oid, n in per_obj_counts.items())


# ---------------- public API ----------------
def load_object_db() -> bool:
    """Load metadata + (re)build embedding cache. Returns True on success."""
    global _db

    if not os.path.isfile(config.OBJECT_DB_JSON):
        print(f"[OBJDB][ERROR] missing {config.OBJECT_DB_JSON}")
        _db = None
        return False

    try:
        with open(config.OBJECT_DB_JSON, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as exc:
        print(f"[OBJDB][ERROR] failed to parse {config.OBJECT_DB_JSON}: {exc}")
        _db = None
        return False

    objects = doc.get("objects") or []
    if not objects:
        print("[OBJDB][WARN] objects.json has no entries.")
        _db = ObjectDB([], np.zeros((0, 1), dtype=np.float32), [], [])
        return True

    # Gather reference image paths per object.
    image_paths_by_obj: dict = {}
    for obj in objects:
        oid = obj["id"]
        folder = os.path.join(config.OBJECT_DB_IMAGES_DIR, oid)
        if not os.path.isdir(folder):
            print(f"[OBJDB][WARN] no image folder for {oid} at {folder}")
            image_paths_by_obj[oid] = []
            continue
        paths = []
        for fn in sorted(os.listdir(folder)):
            if fn.startswith("."):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                paths.append(os.path.join(folder, fn))
        image_paths_by_obj[oid] = paths

    embeddings, embedding_object_ids, source_paths = _load_or_build_embeddings(
        objects, image_paths_by_obj
    )

    _db = ObjectDB(objects, embeddings, embedding_object_ids, source_paths)
    total_refs = len(embedding_object_ids)
    print(
        f"[OBJDB] {len(objects)} objects, {total_refs} reference images. "
        f"Counts: {_db.summary()}"
    )
    return True


# ---------------- embedding cache ----------------
def _cache_signature(image_paths_by_obj: dict) -> dict:
    """A snapshot of every reference image's mtime -- if anything changes the
    cache is invalidated and we re-embed.
    """
    sig = {}
    for oid, paths in image_paths_by_obj.items():
        for p in paths:
            try:
                sig[p] = os.path.getmtime(p)
            except OSError:
                sig[p] = 0.0
    return sig


def _load_or_build_embeddings(objects: list, image_paths_by_obj: dict):
    """Try to reuse the .npz cache if every reference image still has the same
    mtime; otherwise re-embed everything via CLIP.
    """
    expected_sig = _cache_signature(image_paths_by_obj)

    cached = _try_load_cache(expected_sig)
    if cached is not None:
        emb, obj_ids, paths = cached
        print(f"[OBJDB] reused embedding cache ({len(obj_ids)} refs).")
        return emb, obj_ids, paths

    # Build from scratch.
    from . import clip_matcher
    if not clip_matcher.is_ready():
        print("[OBJDB][WARN] CLIP model not ready -- DB will have no embeddings.")
        return np.zeros((0, 1), dtype=np.float32), [], []

    embeddings_list = []
    obj_ids = []
    paths = []

    t0 = time.perf_counter()
    for obj in objects:
        oid = obj["id"]
        for p in image_paths_by_obj.get(oid, []):
            img = cv2.imread(p)
            if img is None:
                print(f"[OBJDB][WARN] could not read {p}; skipping")
                continue
            try:
                emb = clip_matcher.embed_image(img)
            except Exception as exc:
                print(f"[OBJDB][WARN] embed failed for {p}: {exc}")
                continue
            embeddings_list.append(emb)
            obj_ids.append(oid)
            paths.append(p)

    if not embeddings_list:
        print("[OBJDB][WARN] no reference images embedded.")
        return np.zeros((0, 1), dtype=np.float32), [], []

    matrix = np.stack(embeddings_list, axis=0).astype(np.float32)
    elapsed = time.perf_counter() - t0
    print(f"[OBJDB] embedded {len(obj_ids)} refs in {elapsed*1000:.0f} ms.")

    _save_cache(matrix, obj_ids, paths, expected_sig)
    return matrix, obj_ids, paths


def _try_load_cache(expected_sig: dict):
    path = config.OBJECT_DB_EMBEDDINGS_PATH
    if not os.path.isfile(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        emb = data["embeddings"]
        obj_ids = list(data["object_ids"])
        paths = list(data["source_paths"])
        stored_sig = data["signature"].item()  # 0-d object array -> dict
    except Exception as exc:
        print(f"[OBJDB][WARN] cache unreadable ({exc}); rebuilding.")
        return None

    if stored_sig != expected_sig:
        return None

    return emb.astype(np.float32), obj_ids, paths


def _save_cache(matrix, obj_ids, paths, signature):
    os.makedirs(config.OBJECT_DB_DIR, exist_ok=True)
    try:
        np.savez(
            config.OBJECT_DB_EMBEDDINGS_PATH,
            embeddings=matrix,
            object_ids=np.array(obj_ids, dtype=object),
            source_paths=np.array(paths, dtype=object),
            signature=np.array(signature, dtype=object),
        )
        print(f"[OBJDB] cache saved -> {config.OBJECT_DB_EMBEDDINGS_PATH}")
    except Exception as exc:
        print(f"[OBJDB][WARN] failed to write cache: {exc}")
