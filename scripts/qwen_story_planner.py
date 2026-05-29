#!/usr/bin/env python3
"""
Story planner using Qwen2.5-VL already on the GPU server.

Feeds Qwen:
  - annotated keyframes (red=avoid, green=ground, cyan=air)
  - spatial_summary.txt  (named insertion zones with x/y ranges)
  - scene_brief.json     (location, lighting, atmosphere)
  - geometry_catalog     (available render-able characters/objects)

Qwen sees WHAT can be rendered (catalog) and WHERE it is safe (spatial zones),
then outputs a story where each character has a distinct zone and path.

Output: work/story_plan.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Load geometry catalog from blender/ directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blender"))
from geometry_catalog import CATALOG, catalog_for_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL story planner")
    parser.add_argument("--workdir", default="work")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {}


def _named_zones(spatial_text: str) -> str:
    """Extract insertion gap lines and format as named zones for the prompt."""
    zones = []
    for line in spatial_text.splitlines():
        if "x=[" in line and "wide)" in line:
            zones.append(f"  {line.strip()}")
    if not zones:
        return "  (no specific gaps detected — use full lower frame y>=0.70)"
    return "\n".join(zones[:8])


def _build_prompt(scene_brief: dict, spatial_text: str) -> str:
    summary = scene_brief.get("scene_summary", {})
    concept = scene_brief.get("first_cgi_concept", {})
    vlm = scene_brief.get("vlm_scene_understanding", {}).get("parsed_response", {})

    scene_info = {
        "location": summary.get("likely_location", ""),
        "environment": summary.get("environment", ""),
        "time_and_lighting": summary.get("lighting_notes", ""),
        "camera_motion": summary.get("camera_motion", ""),
        "foreground_occluders": summary.get("foreground_occluders", []),
        "safe_insertion_regions": vlm.get("safe_insertion_regions", []),
        "initial_cgi_suggestion": concept.get("object", ""),
    }

    zones_text = _named_zones(spatial_text)

    system = """\
You are a VFX story director for a Blender matchmove project.
Annotated keyframes use colour overlays:
  RED   = obstacles (people, trees, railings) — NO CGI here
  GREEN = safe ground surfaces (walkway, floor) — ground characters
  CYAN  = safe air / open space — floating characters

RULES:
1. Each character MUST use a DIFFERENT zone and a DIFFERENT route.
   Two characters with the same path is wrong.
2. ground characters: screen_path y >= 0.70 (lower = walkway level)
   air characters:    screen_path y 0.30-0.60 (upper = above crowd)
3. Use different x-ranges for each character so they don't overlap.
4. geometry_function MUST be chosen from the catalog by semantic fit:
   - An animal/creature → make_kitsune_fox (quadruped wireframe)
   - A floating orb/drone/light → make_lantern_drone
   - A humanoid robot → make_robot
   - A flat holographic display/billboard → make_hologram_panel
   A 'holographic fox' is still a fox (make_kitsune_fox), not a panel.
5. Return ONLY valid JSON. No markdown, no extra keys.
"""

    schema = {
        "narrative": "2-3 sentences: what happens and why it fits this exact scene",
        "objects": [
            {
                "id": "obj_0",
                "label": "vivid name e.g. 'amber lantern drone' or 'wireframe kitsune spirit'",
                "story_role": "what this character does in the scene",
                "placement": "ground | air",
                "geometry_function": "make_lantern_drone | make_kitsune_fox | make_robot | make_hologram_panel",
                "screen_path": [
                    [0.10, 0.82], [0.25, 0.79], [0.38, 0.81], [0.50, 0.78]
                ],
                "appearance": {
                    "color_rgb": [1.0, 0.72, 0.2],
                    "emission_strength": 2.0,
                    "scale": 0.42
                },
                "action": "hover | walk | dance | spin | jump | orbit | pulse",
                "action_frequency_seconds": 2.5
            }
        ]
    }

    return f"""{system}

SCENE CONTEXT:
{json.dumps(scene_info, indent=2)}

SPATIAL INSERTION GAPS (use different gaps for each character):
{zones_text}

AVAILABLE CGI GEOMETRY (render library — choose geometry_function from this list):
{catalog_for_prompt()}

Design 2-3 characters. Assign each to a DIFFERENT spatial zone.
Return JSON matching this schema:
{json.dumps(schema, indent=2)}
"""


def _normalise(obj: dict, i: int, video: dict) -> dict:
    """Ensure every required field exists with sensible defaults."""
    obj.setdefault("id", f"obj_{i}")
    obj.setdefault("placement", "ground")
    obj.setdefault("action", "walk")
    obj.setdefault("action_frequency_seconds", 2.5)

    app = obj.setdefault("appearance", {})
    # Support both color and color_rgb keys from Qwen
    if "color_rgb" in app and "color" not in app:
        app["color"] = app.pop("color_rgb")
    app.setdefault("color", [0.9, 0.75, 0.2])
    app.setdefault("emission_strength", 2.0)
    app.setdefault("scale", 0.40)
    obj["scale"] = app["scale"]
    # Flatten appearance for insert_cgi.py
    obj["appearance"] = {
        "color": app["color"],
        "emission_strength": app["emission_strength"],
    }

    # Validate geometry_function against catalog
    fn = obj.get("geometry_function", "")
    if fn not in CATALOG:
        # Let semantic dispatch handle it at render time
        obj["geometry_function"] = ""

    # Clamp paths: ground objects stay low, air stays mid-frame
    raw_path = obj.get("screen_path", [[0.25, 0.78], [0.5, 0.75], [0.72, 0.77]])
    placement = obj.get("placement", "ground")
    if placement == "ground":
        obj["screen_path"] = [[x, max(0.70, y)] for x, y in raw_path]
    else:
        obj["screen_path"] = [[x, min(0.65, max(0.28, y))] for x, y in raw_path]

    # Stamp video duration
    duration = float(video.get("duration_seconds", 10.0))
    obj.setdefault("animation", {})["duration_seconds"] = duration
    obj["animation"]["screen_path"] = obj["screen_path"]
    obj["animation"]["scale"] = obj["scale"]
    obj["animation"]["rotation_turns"] = 1.0

    return obj


def main() -> int:
    args = parse_args()
    work = Path(args.workdir)

    scene_brief = _load(work / "scene_brief.json")
    spatial_text = (work / "spatial_summary.txt").read_text(encoding="utf-8") \
        if (work / "spatial_summary.txt").exists() else ""

    if not scene_brief:
        raise SystemExit("scene_brief.json not found. Run qwen_vl_scene.py first.")

    # Pick annotated frames (show spatial context to Qwen)
    frame_dir = work / "annotated_frames"
    if not frame_dir.exists():
        frame_dir = work / "frames"
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    stride = max(1, len(frame_paths) // args.max_frames)
    selected = frame_paths[::stride][: args.max_frames]

    print(f"Loading {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
    )

    prompt = _build_prompt(scene_brief, spatial_text)
    content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in selected]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    print(f"Generating story with {len(selected)} annotated frames + geometry catalog...")
    text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)
    inputs = processor(text=[text_in], images=img_in, videos=vid_in,
                       padding=True, return_tensors="pt").to(model.device)

    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                               do_sample=True, temperature=0.7, top_p=0.9)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated, strict=True)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]

    story = _parse_json(raw)
    if not story.get("objects"):
        (work / "story_plan_raw.txt").write_text(raw, encoding="utf-8")
        raise SystemExit(f"Could not parse JSON. Raw saved.\n{raw[:400]}")

    # Load video info for duration stamping
    ctx = _load(work / "perception_context.json")
    video = ctx.get("video", {})

    story["objects"] = [_normalise(obj, i, video)
                        for i, obj in enumerate(story["objects"])]

    # Verify no two objects have identical paths
    paths = [str(o["screen_path"]) for o in story["objects"]]
    if len(set(paths)) < len(paths):
        print("WARNING: two objects share identical paths — Qwen may not have diversified routes.")

    out = work / "story_plan.json"
    out.write_text(json.dumps(story, indent=2) + "\n", encoding="utf-8")
    print(f"\nStory → {out}")
    print(f"Narrative: {story.get('narrative', '')[:200]}")
    for o in story["objects"]:
        print(f"  [{o['id']}] {o['label']} | fn={o.get('geometry_function','?')} "
              f"| placement={o['placement']} | path_pts={len(o['screen_path'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
