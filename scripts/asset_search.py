#!/usr/bin/env python3
"""
Semantic 3D asset search over Objaverse LVIS using CLIP embeddings.

No bag-of-words. No hardcoded keywords.

The query (e.g. "ethereal kitsune fox spirit, wireframe glowing") is embedded
with the CLIP text encoder. All Objaverse LVIS category names are also embedded.
Cosine similarity picks the best-matching categories, then downloads GLBs from
those categories. A second CLIP pass re-ranks by thumbnail image similarity
(if renders are available) before returning the top result.

Output: work/asset_manifest.json
  {
    "obj_0": { "label": "kitsune fox", "glb_path": "work/assets/xxx.glb", "uid": "..." },
    "obj_1": { "label": "lantern drone", "glb_path": null, "fallback": "make_lantern_drone" }
  }
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

CACHE_DIR = Path.home() / ".objaverse_clip_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLIP semantic search over Objaverse LVIS")
    parser.add_argument("--workdir", default="work")
    parser.add_argument("--story-plan", default="work/story_plan.json")
    parser.add_argument("--top-cats", type=int, default=4,
                        help="Top CLIP-matched categories to pull UIDs from")
    parser.add_argument("--per-cat", type=int, default=3,
                        help="GLBs to download per category")
    parser.add_argument("--clip-model", default="ViT-B-32")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CLIP helpers
# ---------------------------------------------------------------------------

def load_clip(model_name: str, device: str):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def embed_texts(texts: list[str], model, tokenizer, device: str) -> torch.Tensor:
    tokens = tokenizer(texts).to(device)
    embs = model.encode_text(tokens)
    return embs / embs.norm(dim=-1, keepdim=True)


@torch.no_grad()
def embed_images(paths: list[Path], model, preprocess, device: str) -> torch.Tensor:
    from PIL import Image
    imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)
    embs = model.encode_image(imgs)
    return embs / embs.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Objaverse search
# ---------------------------------------------------------------------------

def load_lvis_categories() -> dict[str, list[str]]:
    """Return {category_name: [uid, ...]} for the Objaverse LVIS subset."""
    import objaverse
    return objaverse.load_lvis_annotations()


def top_categories(query: str, lvis: dict[str, list[str]],
                   model, tokenizer, device: str, top_k: int) -> list[str]:
    cats = list(lvis.keys())
    # Batch embed in chunks to avoid OOM
    chunk, all_embs = 512, []
    for i in range(0, len(cats), chunk):
        all_embs.append(embed_texts(cats[i: i + chunk], model, tokenizer, device))
    cat_embs = torch.cat(all_embs, dim=0)

    q_emb = embed_texts([query], model, tokenizer, device)
    sims = (q_emb @ cat_embs.T).squeeze(0)
    top_idx = sims.topk(min(top_k, len(cats))).indices.tolist()
    return [cats[i] for i in top_idx]


def download_glbs(uids: list[str], dest: Path) -> dict[str, Path]:
    """Download GLBs and copy to dest/. Returns {uid: local_path}."""
    import objaverse
    dest.mkdir(parents=True, exist_ok=True)
    raw = objaverse.load_objects(uids=uids, download_processes=4)
    result: dict[str, Path] = {}
    for uid, src in raw.items():
        if src and Path(src).exists():
            dst = dest / f"{uid}.glb"
            shutil.copy2(src, dst)
            result[uid] = dst
    return result


def rank_by_annotations(
    uid_paths: dict[str, Path], query: str,
    model, tokenizer, device: str,
) -> list[tuple[str, Path, float]]:
    """
    Rank candidates by CLIP text similarity between the query and each
    object's Objaverse annotation (name + tags).

    This replaces the thumbnail re-rank which required pre-rendered images
    that aren't reliably available. Text-to-text annotation ranking gives
    meaningful signal within a category without any image downloads.
    """
    import objaverse

    try:
        annotations = objaverse.load_annotations(list(uid_paths.keys()))
    except Exception:
        annotations = {}

    q_emb = embed_texts([query], model, tokenizer, device)
    scored: list[tuple[str, Path, float]] = []

    for uid, glb_path in uid_paths.items():
        ann = annotations.get(uid, {}) or {}
        name = ann.get("name", "")
        tags = ann.get("tags", [])
        # tags can be list of strings or list of dicts {"name": ...}
        tag_strs = [t["name"] if isinstance(t, dict) else str(t) for t in tags[:12]]
        ann_text = f"{name} {' '.join(tag_strs)}".strip()

        if ann_text:
            ann_emb = embed_texts([ann_text], model, tokenizer, device)
            score = float((q_emb @ ann_emb.T).squeeze())
        else:
            score = 0.0

        scored.append((uid, glb_path, score))
        print(f"    {uid[:8]}… name='{name}' score={score:.3f}")

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Main search function
# ---------------------------------------------------------------------------

def search_for_character(
    obj_id: str, label: str, story_role: str, geometry_function: str,
    lvis: dict[str, list[str]], model, tokenizer, device: str,
    top_cats: int, per_cat: int, asset_dir: Path,
) -> dict:
    """Find the best GLB for one story character. Returns manifest entry."""
    query = f"{label}: {story_role}"
    print(f"\n[{obj_id}] Searching: '{query}'")

    # Stage 1 — CLIP category matching
    best_cats = top_categories(query, lvis, model, tokenizer, device, top_cats)
    print(f"  Top categories: {best_cats}")

    # Stage 2 — collect UIDs from top categories
    candidate_uids: list[str] = []
    for cat in best_cats:
        candidate_uids.extend(lvis[cat][:per_cat])
    candidate_uids = list(dict.fromkeys(candidate_uids))[:top_cats * per_cat]

    if not candidate_uids:
        print(f"  No candidates found — will use procedural fallback ({geometry_function})")
        return {
            "id": obj_id, "label": label,
            "glb_path": None, "uid": None,
            "fallback": geometry_function,
        }

    # Stage 3 — download GLBs
    print(f"  Downloading {len(candidate_uids)} candidate GLBs...")
    uid_paths = download_glbs(candidate_uids, asset_dir / obj_id)

    if not uid_paths:
        print(f"  Download failed — using procedural fallback ({geometry_function})")
        return {
            "id": obj_id, "label": label,
            "glb_path": None, "uid": None,
            "fallback": geometry_function,
        }

    # Stage 4 — rank by Objaverse annotation text (name + tags) via CLIP
    ranked = rank_by_annotations(uid_paths, query, model, tokenizer, device)
    best_uid, best_path, best_score = ranked[0]
    print(f"  Best match: uid={best_uid} score={best_score:.3f} → {best_path.name}")

    return {
        "id": obj_id, "label": label,
        "glb_path": str(best_path), "uid": best_uid,
        "clip_score": best_score,
        "fallback": geometry_function,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    work = Path(args.workdir)
    story_path = Path(args.story_plan)

    if not story_path.exists():
        raise SystemExit(f"story_plan.json not found at {story_path}")

    story = json.loads(story_path.read_text())
    objects = story.get("objects", [])
    if not objects:
        raise SystemExit("story_plan.json has no objects.")

    asset_dir = work / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP ({args.clip_model}) on {device}...")
    model, preprocess, tokenizer = load_clip(args.clip_model, device)

    print("Loading Objaverse LVIS annotations...")
    lvis = load_lvis_categories()
    print(f"  {len(lvis)} categories, "
          f"{sum(len(v) for v in lvis.values())} total objects")

    manifest: dict[str, dict] = {}
    for obj in objects:
        # Use visual_description as primary query — richer than label alone
        query_text = (
            obj.get("visual_description")
            or f"{obj.get('label', '')} {obj.get('story_role', '')}"
        )
        entry = search_for_character(
            obj_id=obj.get("id", "obj_0"),
            label=query_text,
            story_role=obj.get("story_role", ""),
            geometry_function=obj.get("geometry_function", ""),
            lvis=lvis,
            model=model, tokenizer=tokenizer,
            device=device,
            top_cats=args.top_cats,
            per_cat=args.per_cat,
            asset_dir=asset_dir,
        )
        manifest[entry["id"]] = entry

    out = work / "asset_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nAsset manifest → {out}")
    for k, v in manifest.items():
        status = f"GLB: {Path(v['glb_path']).name}" if v.get("glb_path") else f"fallback: {v.get('fallback')}"
        print(f"  {k}: {v['label']} → {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
