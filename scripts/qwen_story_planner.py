#!/usr/bin/env python3
"""
Story planner using Qwen2.5-VL already on the GPU server.

Feeds Qwen:
  - annotated keyframes (red=avoid, green=ground, cyan=air)
  - spatial_summary.txt  (insertion gaps, obstacle positions)
  - scene_brief.json     (location, lighting, atmosphere)

Qwen outputs a multi-object CGI story grounded in the actual scene.

Output: work/story_plan.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


SYSTEM_PROMPT = """\
You are a creative VFX story director working on a matchmove project.
You have been given annotated video keyframes where:
  RED regions   = obstacles to avoid (people, trees, railings, signs)
  GREEN regions = safe ground surfaces (walkways, floor, pavement)
  CYAN regions  = safe air/open space (sky, above crowd)

Your job: design a creative, scene-appropriate CGI story with 2-3 characters
that fit this specific scene. Ground every placement decision in what you
see in the annotated frames and the spatial summary.

IMPORTANT:
- Characters must be placed in GREEN or CYAN regions, never RED
- Paths must avoid RED obstacle positions
- Story must reflect the real scene (location, time of day, atmosphere)
- Return ONLY valid JSON. No markdown fences, no extra text.
"""

SCHEMA_EXAMPLE = {
    "narrative": "2-3 sentences describing what happens and why it fits this scene",
    "objects": [
        {
            "id": "obj_0",
            "label": "descriptive name matching the scene (e.g. glowing kitsune fox, holographic torii gate)",
            "story_role": "what this character does and why it belongs here",
            "placement": "ground | air",
            "screen_path": [
                [0.15, 0.82], [0.30, 0.78], [0.48, 0.74], [0.65, 0.76]
            ],
            "appearance": {
                "shape": "sphere | box | cylinder | torus | combined",
                "color_rgb": [1.0, 0.85, 0.2],
                "emission_strength": 1.4,
                "scale": 0.35
            },
            "action": "walk | jump | hover | spin | dance | orbit | pulse",
            "action_frequency_seconds": 3.0
        }
    ]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL story planner for CGI insertion")
    parser.add_argument("--workdir", default="work")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-frames", type=int, default=8,
                        help="Number of annotated frames to show Qwen")
    parser.add_argument("--max-new-tokens", type=int, default=1600)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
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


def _build_text_prompt(scene_brief: dict, spatial_text: str) -> str:
    summary = scene_brief.get("scene_summary", {})
    concept = scene_brief.get("first_cgi_concept", {})
    vlm = scene_brief.get("vlm_scene_understanding", {}).get("parsed_response", {})

    scene_info = {
        "location": summary.get("likely_location", ""),
        "environment": summary.get("environment", ""),
        "time_of_day_and_lighting": summary.get("lighting_notes", ""),
        "camera_motion": summary.get("camera_motion", ""),
        "foreground_occluders": summary.get("foreground_occluders", []),
        "dominant_surfaces": summary.get("dominant_surfaces", []),
        "safe_insertion_regions": vlm.get("safe_insertion_regions", []),
        "vlm_initial_suggestion": concept.get("object", ""),
    }

    return f"""{SYSTEM_PROMPT}

SCENE CONTEXT:
{json.dumps(scene_info, indent=2)}

SPATIAL ANALYSIS (from annotated frames):
{spatial_text}

Design a story with 2-3 CGI characters. Use the annotated frames above to
confirm safe placement positions. Output JSON matching this schema:
{json.dumps(SCHEMA_EXAMPLE, indent=2)}
"""


def main() -> int:
    args = parse_args()
    work = Path(args.workdir)

    scene_brief = _load_json(work / "scene_brief.json")
    spatial_text = (work / "spatial_summary.txt").read_text(encoding="utf-8") \
        if (work / "spatial_summary.txt").exists() else ""

    if not scene_brief:
        raise SystemExit("scene_brief.json not found. Run qwen_vl_scene.py first.")
    if not spatial_text:
        raise SystemExit("spatial_summary.txt not found. Run annotate_frames.py first.")

    # Pick annotated frames (prefer those with insertion gaps)
    annotated_dir = work / "annotated_frames"
    if annotated_dir.exists():
        frame_paths = sorted(annotated_dir.glob("*.jpg"))
    else:
        frame_paths = sorted((work / "frames").glob("*.jpg"))

    stride = max(1, len(frame_paths) // args.max_frames)
    selected = frame_paths[::stride][: args.max_frames]

    print(f"Loading {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    text_prompt = _build_text_prompt(scene_brief, spatial_text)

    # Build multimodal message: images first, then text
    content = [
        {"type": "image", "image": Image.open(p).convert("RGB")}
        for p in selected
    ]
    content.append({"type": "text", "text": text_prompt})
    messages = [{"role": "user", "content": content}]

    print(f"Generating story with {len(selected)} annotated frames...")
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated, strict=True)
    ]
    raw = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"\nRaw response:\n{raw[:500]}...")
    story = _parse_json(raw)

    if not story.get("objects"):
        # Save raw for debugging and exit gracefully
        (work / "story_plan_raw.txt").write_text(raw, encoding="utf-8")
        raise SystemExit(
            f"Could not parse story JSON. Raw saved to work/story_plan_raw.txt\n{raw[:300]}"
        )

    # Normalise each object to ensure required fields exist
    for i, obj in enumerate(story["objects"]):
        obj.setdefault("id", f"obj_{i}")
        obj.setdefault("placement", "ground")
        obj.setdefault("action", "walk")
        obj.setdefault("action_frequency_seconds", 3.0)
        app = obj.setdefault("appearance", {})
        app.setdefault("color_rgb", [0.9, 0.75, 0.2])
        app.setdefault("emission_strength", 1.2)
        app.setdefault("scale", 0.38)
        app.setdefault("shape", "combined")
        # Map appearance to insert_cgi.py format
        obj["appearance"] = {
            "color": app.get("color_rgb", [0.9, 0.75, 0.2]),
            "emission_strength": app.get("emission_strength", 1.2),
        }
        obj["scale"] = app.get("scale", 0.38)
        obj.setdefault("screen_path", [[0.2, 0.8], [0.45, 0.75], [0.7, 0.78]])

        # Map action + shape to an asset type insert_cgi.py understands
        action = obj.get("action", "walk")
        shape = app.get("shape", "combined")
        placement = obj.get("placement", "ground")
        if placement == "air" or action in ("hover", "orbit", "pulse"):
            obj["asset"] = "lantern_drone"
        elif "panel" in obj.get("label", "").lower() or "holo" in obj.get("label", "").lower():
            obj["asset"] = "hologram_panel"
        else:
            obj["asset"] = "procedural_robot"

    out_path = work / "story_plan.json"
    out_path.write_text(json.dumps(story, indent=2) + "\n", encoding="utf-8")
    print(f"\nStory plan → {out_path}")
    print(f"Narrative: {story.get('narrative', '')[:200]}")
    print(f"Objects:   {[o.get('label', o['id']) for o in story['objects']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
