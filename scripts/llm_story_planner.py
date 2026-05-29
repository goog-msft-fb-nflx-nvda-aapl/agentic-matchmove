#!/usr/bin/env python3
"""
LLM story planner: calls Claude API with the scene brief and perception
context to generate a multi-object, semantically grounded CGI story.

Output: work/story_plan.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import anthropic

ASSETS = {
    "lantern_drone": "glowing floating lantern sphere with emissive ring; hovers and drifts; best for outdoor/dusk",
    "procedural_robot": "small walking robot with body, eyes, arms, antenna; best for urban/tech scenes",
    "hologram_panel": "floating holographic billboard/display; best for futuristic/urban scenes",
}

SYSTEM = """\
You are a VFX story director for a Blender matchmove pipeline.
You receive VLM scene understanding and object detection/segmentation output,
and you design a concrete, scene-appropriate multi-object CGI insertion story.

Rules:
- Return ONLY valid JSON matching the schema. No markdown fences, no extra keys.
- screen_path: 4-6 [x, y] pairs, normalized coords (0=left/top, 1=right/bottom).
- Paths must use the safe_insertion_regions and avoid occlusion_risks positions.
- Spread objects across different screen regions so they don't overlap.
- Colors as [R, G, B] floats 0-1. Emission 0.5-2.5. Scale 0.15-0.55.
- 2-3 objects. Each must have a distinct role and position.
- The story must reflect the specific scene semantics (location, lighting, atmosphere).
"""

SCHEMA = {
    "narrative": "2-3 sentences: what happens, why it fits this specific scene",
    "objects": [
        {
            "id": "obj_0",
            "asset": "lantern_drone | procedural_robot | hologram_panel",
            "label": "human-readable descriptive name",
            "story_role": "1 sentence: what this object does in the scene",
            "screen_path": [[0.2, 0.75], [0.38, 0.68], [0.55, 0.70], [0.72, 0.73]],
            "appearance": {"color": [1.0, 0.72, 0.22], "emission_strength": 1.2},
            "scale": 0.38,
            "rotation_turns": 0.5,
        }
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-object CGI story via Claude API")
    parser.add_argument("--workdir", default="work")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY", ""))
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _summarize_instances(context: dict, max_instances: int = 20) -> list[dict]:
    """Flatten instances across frames, deduplicate by label."""
    seen: dict[str, dict] = {}
    width = max(1, context.get("video", {}).get("width", 1))
    height = max(1, context.get("video", {}).get("height", 1))
    for frame in context.get("frames", []):
        for inst in frame.get("instances", []):
            label = inst.get("label", "object")
            if label not in seen:
                x1, y1, x2, y2 = inst.get("bbox_xyxy", [0, 0, 0, 0])
                seen[label] = {
                    "label": label,
                    "screen_center": [
                        round((x1 + x2) * 0.5 / width, 2),
                        round((y1 + y2) * 0.5 / height, 2),
                    ],
                    "confidence": round(inst.get("confidence", 0), 2),
                }
            if len(seen) >= max_instances:
                break
    return list(seen.values())


def _build_prompt(scene_brief: dict, context: dict) -> str:
    summary = scene_brief.get("scene_summary", {})
    concept = scene_brief.get("first_cgi_concept", {})
    vlm_parsed = (
        scene_brief.get("vlm_scene_understanding", {}).get("parsed_response", {})
    )
    instances = _summarize_instances(context)

    scene_info = {
        "location": summary.get("likely_location") or context.get("user_context", {}).get("location", ""),
        "environment": summary.get("environment", ""),
        "camera_motion": summary.get("camera_motion", ""),
        "lighting": summary.get("lighting_notes", ""),
        "dominant_surfaces": summary.get("dominant_surfaces", []),
        "foreground_occluders": summary.get("foreground_occluders", []),
        "safe_insertion_regions": vlm_parsed.get("safe_insertion_regions", []),
        "occlusion_risks": vlm_parsed.get("occlusion_risks", []),
        "detected_objects": instances,
        "vlm_cgi_suggestion": {
            "object": concept.get("object", ""),
            "proposed_action": concept.get("proposed_action", ""),
            "screen_path_draft": concept.get("screen_path_draft", []),
            "must_avoid": concept.get("must_avoid", []),
        },
    }

    asset_list = "\n".join(f'  "{k}": {v}' for k, v in ASSETS.items())

    return f"""Scene context:
{json.dumps(scene_info, indent=2)}

Available CGI assets:
{asset_list}

Output schema (return this exact structure as JSON):
{json.dumps(SCHEMA, indent=2)}

Design a creative, scene-appropriate story using 2-3 of the available assets.
Ground every placement decision in the detected objects and safe insertion regions above.
"""


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
    return {}


def main() -> int:
    args = parse_args()
    work = Path(args.workdir)

    if not args.api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set. Pass --api-key or export the env var.")

    scene_brief = _load_json(work / "scene_brief.json")
    context = _load_json(work / "perception_context.json")

    if not scene_brief:
        raise SystemExit(f"scene_brief.json not found in {work}. Run qwen_vl_scene.py first.")

    prompt = _build_prompt(scene_brief, context)
    client = anthropic.Anthropic(api_key=args.api_key)

    print(f"Calling {args.model} for story planning...")
    response = client.messages.create(
        model=args.model,
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text
    story = _parse_json(raw)

    if not story.get("objects"):
        raise SystemExit(f"LLM returned invalid story JSON:\n{raw}")

    # Validate and normalise each object
    video = context.get("video", {})
    duration = float(video.get("duration_seconds", 10.0))
    for obj in story["objects"]:
        obj.setdefault("rotation_turns", 1.0)
        obj.setdefault("scale", 0.4)
        anim = obj.setdefault("animation", {})
        anim["screen_path"] = obj.get("screen_path", [[0.25, 0.75], [0.5, 0.65], [0.72, 0.70]])
        anim["scale"] = obj["scale"]
        anim["rotation_turns"] = obj["rotation_turns"]
        anim["duration_seconds"] = duration

    out_path = work / "story_plan.json"
    out_path.write_text(json.dumps(story, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    print(f"narrative: {story.get('narrative', '')[:120]}")
    print(f"objects:   {[o['label'] for o in story['objects']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
