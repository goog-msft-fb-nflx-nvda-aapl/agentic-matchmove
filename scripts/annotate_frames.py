#!/usr/bin/env python3
"""
Annotate keyframes with SAM2 masks + DINO bounding boxes.

Color-codes spatial regions so Qwen VLM sees WHERE things are:
  RED    = avoid  (person, railing, tree, tent, sign, ...)
  GREEN  = safe ground surface (walkway, ground, floor, pavement, ...)
  CYAN   = safe air / open space (sky, background above crowd)
  YELLOW = neutral context (building, clock, ...)

Outputs:
  work/annotated_frames/frame_XXXXXX.jpg   annotated images for VLM
  work/annotated_contact_sheet.jpg         grid overview
  work/spatial_summary.json                text description for LLM prompt
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Label → semantic category
# ---------------------------------------------------------------------------

AVOID_LABELS = {
    "person", "face", "hand", "people", "pedestrian",
    "railing", "fence", "tree", "branch", "tent", "sign",
    "bicycle", "motorcycle", "car", "bus", "truck",
    "pole", "wire", "pillar",
}
GROUND_LABELS = {
    "ground", "walkway", "floor", "pavement", "road",
    "sidewalk", "path", "ground walkway", "street", "plaza",
    "stairs", "step", "platform",
}
AIR_LABELS = {
    "sky", "open space", "background", "air", "ceiling",
    "cloud", "upper area",
}

# RGBA overlay colors (semi-transparent)
RGBA = {
    "avoid":   (220,  50,  50, 110),
    "ground":  ( 40, 200,  80, 110),
    "air":     ( 40, 160, 220,  90),
    "neutral": (220, 200,  40,  70),
}
# BGR box colors for OpenCV text/rectangle
BGR = {
    "avoid":   (  0,   0, 220),
    "ground":  (  0, 200,  40),
    "air":     (220, 160,  40),
    "neutral": (  0, 200, 220),
}


def label_category(label: str) -> str:
    l = label.lower().strip()
    if l in AVOID_LABELS:
        return "avoid"
    if l in GROUND_LABELS:
        return "ground"
    if l in AIR_LABELS:
        return "air"
    return "neutral"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def apply_mask_overlay(img_pil: Image.Image, mask_path: str, category: str) -> Image.Image:
    """Alpha-blend a SAM2 binary mask onto the frame."""
    try:
        mask = np.array(Image.open(mask_path).convert("L"))
    except Exception:
        return img_pil

    r, g, b, a = RGBA[category]
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    draw_arr = np.zeros((*mask.shape, 4), dtype=np.uint8)
    binary = mask > 128
    draw_arr[binary] = [r, g, b, a]
    overlay = Image.fromarray(draw_arr, mode="RGBA")
    base = img_pil.convert("RGBA")
    return Image.alpha_composite(base, overlay)


def draw_boxes(img_np: np.ndarray, segments: list[dict], w: int, h: int) -> np.ndarray:
    """Draw bounding boxes + labels on a numpy BGR image."""
    for seg in segments:
        x1, y1, x2, y2 = seg.get("bbox_xyxy", [0, 0, 0, 0])
        label = seg.get("label", "?")
        conf = seg.get("confidence", 0.0)
        cat = label_category(label)
        color = BGR[cat]

        cv2.rectangle(img_np, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(int(y1) - 4, th + 2)
        cv2.rectangle(img_np, (int(x1), ty - th - 2), (int(x1) + tw + 4, ty + 2), color, -1)
        cv2.putText(img_np, text, (int(x1) + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return img_np


def draw_legend(img_np: np.ndarray) -> np.ndarray:
    entries = [
        ("AVOID (person/tree/railing)", BGR["avoid"]),
        ("SAFE GROUND (walkway/floor)", BGR["ground"]),
        ("SAFE AIR (sky/open space)", BGR["air"]),
        ("NEUTRAL (building/context)", BGR["neutral"]),
    ]
    x0, y0 = 8, 8
    for i, (text, color) in enumerate(entries):
        y = y0 + i * 22
        cv2.rectangle(img_np, (x0, y), (x0 + 16, y + 16), color, -1)
        cv2.putText(img_np, text, (x0 + 22, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                    lineType=cv2.LINE_AA)
    return img_np


# ---------------------------------------------------------------------------
# Spatial summary helpers
# ---------------------------------------------------------------------------

def bbox_normalized(bbox: list[float], w: int, h: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [round(x1 / w, 3), round(y1 / h, 3), round(x2 / w, 3), round(y2 / h, 3)]


def summarize_frame_segments(
    segments: list[dict], w: int, h: int
) -> dict:
    avoid, ground, air = [], [], []
    for seg in segments:
        cat = label_category(seg.get("label", ""))
        bbox_n = bbox_normalized(seg.get("bbox_xyxy", [0, 0, w, h]), w, h)
        area = (seg.get("mask_area_px") or
                (seg["bbox_xyxy"][2] - seg["bbox_xyxy"][0]) *
                (seg["bbox_xyxy"][3] - seg["bbox_xyxy"][1])
                if "bbox_xyxy" in seg else 0)
        area_frac = round(area / (w * h), 4)
        entry = {
            "label": seg.get("label", "?"),
            "screen_bbox_normalized": bbox_n,
            "area_fraction": area_frac,
        }
        if cat == "avoid":
            avoid.append(entry)
        elif cat == "ground":
            ground.append(entry)
        elif cat == "air":
            air.append(entry)

    # Derive insertion gaps: horizontal slices of the frame not occupied by "avoid"
    gaps = _find_ground_gaps(avoid, ground, w, h)

    return {
        "obstacles": avoid,
        "safe_ground": ground,
        "safe_air": air,
        "insertion_gaps": gaps,
    }


def _find_ground_gaps(
    avoid: list[dict], ground: list[dict], w: int, h: int
) -> list[dict]:
    """Find horizontal screen regions at ground level not occupied by obstacles."""
    # Build a coarse 10-column occupancy at y > 0.55 (lower half)
    cols = 10
    occupied = [False] * cols
    for seg in avoid:
        x1, _, x2, y2 = seg["screen_bbox_normalized"]
        if y2 > 0.55:
            c1 = max(0, int(x1 * cols))
            c2 = min(cols, int(x2 * cols) + 1)
            for c in range(c1, c2):
                occupied[c] = True

    gaps = []
    i = 0
    while i < cols:
        if not occupied[i]:
            j = i
            while j < cols and not occupied[j]:
                j += 1
            if j - i >= 2:  # at least 20% of width
                gaps.append({
                    "screen_x_range": [round(i / cols, 2), round(j / cols, 2)],
                    "screen_y_range": [0.60, 0.92],
                    "width_fraction": round((j - i) / cols, 2),
                })
            i = j
        else:
            i += 1
    return gaps


# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------

def make_contact_sheet(
    annotated_paths: list[Path], output: Path, cols: int = 4, thumb_w: int = 480
) -> None:
    if not annotated_paths:
        return
    imgs = []
    for p in annotated_paths:
        im = Image.open(p).convert("RGB")
        ratio = im.height / im.width
        im = im.resize((thumb_w, int(thumb_w * ratio)), Image.LANCZOS)
        imgs.append(im)

    thumb_h = imgs[0].height
    rows = math.ceil(len(imgs) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (20, 20, 20))
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, cols)
        sheet.paste(im, (c * thumb_w, r * thumb_h))
    sheet.save(output, quality=88)
    print(f"contact sheet → {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate frames with spatial semantics")
    parser.add_argument("--workdir", default="work")
    parser.add_argument("--max-frames", type=int, default=20,
                        help="Max annotated frames to produce")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work = Path(args.workdir)
    out_dir = work / "annotated_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load perception context (has bounding boxes merged from DINO+SAM2)
    context_path = work / "perception_context.json"
    if not context_path.exists():
        raise SystemExit(f"perception_context.json not found in {work}")
    context = json.loads(context_path.read_text())

    frames = context.get("frames", [])
    video_w = int(context.get("video", {}).get("width", 1920))
    video_h = int(context.get("video", {}).get("height", 1080))

    # Also load SAM2 segments for mask paths (may differ from context instances)
    sam2_by_image: dict[str, list[dict]] = {}
    sam2_path = work / "sam2_segments.json"
    if sam2_path.exists():
        sam2_data = json.loads(sam2_path.read_text())
        for sf in sam2_data.get("frames", []):
            sam2_by_image[sf.get("image_path", "")] = sf.get("segments", [])

    stride = max(1, len(frames) // args.max_frames)
    selected = frames[::stride][: args.max_frames]

    annotated_paths: list[Path] = []
    spatial_frames: list[dict] = []

    for frame in selected:
        img_path = Path(frame.get("image_path", ""))
        if not img_path.exists():
            continue

        img_pil = Image.open(img_path).convert("RGB")
        w, h = img_pil.size

        # Merge instances from context + SAM2 segments
        instances = frame.get("instances", [])
        sam2_segs = sam2_by_image.get(str(img_path), [])
        all_segs = instances if instances else sam2_segs

        # 1. Apply SAM2 mask overlays (colour-coded by category)
        for seg in all_segs:
            mask_path = seg.get("mask_path", "")
            if mask_path and Path(mask_path).exists():
                cat = label_category(seg.get("label", ""))
                img_pil = apply_mask_overlay(img_pil, mask_path, cat)

        # 2. Draw bounding boxes + labels
        img_np = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        img_np = draw_boxes(img_np, all_segs, w, h)

        # 3. Add timestamp + legend
        ts = frame.get("timestamp", 0.0)
        cv2.putText(img_np, f"t={ts:.1f}s  frame={frame.get('frame_index', 0)}",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 200), 1)
        img_np = draw_legend(img_np)

        # 4. Save
        fname = Path(img_path).stem
        out_path = out_dir / f"{fname}.jpg"
        cv2.imwrite(str(out_path), img_np, [cv2.IMWRITE_JPEG_QUALITY, 88])
        annotated_paths.append(out_path)

        # 5. Build spatial summary for this frame
        summary = summarize_frame_segments(all_segs, w, h)
        summary["frame_index"] = frame.get("frame_index", 0)
        summary["timestamp"] = ts
        summary["image_path"] = str(out_path)
        spatial_frames.append(summary)

    # Contact sheet
    make_contact_sheet(annotated_paths, work / "annotated_contact_sheet.jpg")

    # Spatial summary JSON + human-readable text description
    spatial_summary = {
        "video": context.get("video", {}),
        "frames": spatial_frames,
        "aggregate": _aggregate_summary(spatial_frames),
    }
    summary_path = work / "spatial_summary.json"
    summary_path.write_text(json.dumps(spatial_summary, indent=2) + "\n")
    print(f"spatial summary → {summary_path}")

    # Also write a plain-text version for direct LLM injection
    text_path = work / "spatial_summary.txt"
    text_path.write_text(_to_text(spatial_summary) + "\n")
    print(f"spatial text    → {text_path}")

    print(f"annotated frames: {len(annotated_paths)}")
    return 0


def _aggregate_summary(frames: list[dict]) -> dict:
    """Summarise across all frames: most common obstacles, largest safe zones."""
    obstacle_counts: dict[str, int] = {}
    ground_present = 0
    air_present = 0
    gaps_all: list[dict] = []

    for f in frames:
        for obs in f.get("obstacles", []):
            label = obs["label"]
            obstacle_counts[label] = obstacle_counts.get(label, 0) + 1
        if f.get("safe_ground"):
            ground_present += 1
        if f.get("safe_air"):
            air_present += 1
        gaps_all.extend(f.get("insertion_gaps", []))

    # Best insertion gap (widest, appears most often)
    gap_scores: dict[str, float] = {}
    for g in gaps_all:
        key = f"{g['screen_x_range']}"
        gap_scores[key] = gap_scores.get(key, 0) + g["width_fraction"]
    best_gap = max(gap_scores, key=lambda k: gap_scores[k]) if gap_scores else None

    return {
        "dominant_obstacles": sorted(obstacle_counts, key=lambda k: -obstacle_counts[k])[:5],
        "ground_surface_visible_in_frames": ground_present,
        "air_region_visible_in_frames": air_present,
        "best_insertion_gap_x_range": best_gap,
        "total_frames_analysed": len(frames),
    }


def _to_text(summary: dict) -> str:
    agg = summary.get("aggregate", {})
    lines = [
        "=== SPATIAL SUMMARY FOR CGI STORY PLANNING ===",
        "",
        f"Dominant obstacles (AVOID placing CGI on these): {', '.join(agg.get('dominant_obstacles', []))}",
        f"Safe ground surface visible in {agg.get('ground_surface_visible_in_frames', 0)} / {agg.get('total_frames_analysed', 0)} frames",
        f"Safe air region visible in {agg.get('air_region_visible_in_frames', 0)} / {agg.get('total_frames_analysed', 0)} frames",
        f"Best horizontal insertion gap (x): {agg.get('best_insertion_gap_x_range', 'none found')}",
        "",
        "Per-frame insertion gaps (normalized screen coords, y=0.60-0.92 = lower ground area):",
    ]
    for f in summary.get("frames", []):
        gaps = f.get("insertion_gaps", [])
        if gaps:
            gap_str = "; ".join(
                f"x={g['screen_x_range']} ({int(g['width_fraction']*100)}% wide)"
                for g in gaps
            )
            lines.append(f"  t={f['timestamp']:.1f}s: {gap_str}")
    lines += [
        "",
        "Use annotated_frames/ images (red=avoid, green=ground, cyan=air) alongside this text.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
