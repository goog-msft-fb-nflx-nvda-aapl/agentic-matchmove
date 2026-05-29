#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL scene understanding on sampled keyframes")
    parser.add_argument("--workdir", default="work")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--user-location", default="")
    return parser.parse_args()


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {}


def _synthesize_scene_brief(
    parsed: dict,
    raw_response: str,
    model: str,
    selected_frames: list[str],
    user_location: str,
) -> dict:
    suggested = parsed.get("suggested_cgi", {})
    return {
        "source_video": "data/input.mp4",
        "status": "vlm_scene_understanding_complete",
        "scene_summary": {
            "likely_location": user_location or "",
            "environment": parsed.get("scene_summary", ""),
            "camera_motion": parsed.get("camera_motion", ""),
            "dominant_surfaces": parsed.get("objects_and_regions", []),
            "foreground_occluders": parsed.get("occlusion_risks", []),
            "lighting_notes": parsed.get("lighting_notes", ""),
        },
        "first_cgi_concept": {
            "object": suggested.get("object", ""),
            "reasoning": suggested.get("reasoning", ""),
            "proposed_action": suggested.get("action", ""),
            "screen_path_draft": suggested.get("screen_path", [[0.25, 0.75], [0.50, 0.65], [0.72, 0.70]]),
            "must_avoid": list(parsed.get("occlusion_risks", [])),
        },
        "next_recommended_step": (
            "Run Grounded-SAM2 with the segmentation_prompts for occlusion masks, "
            "then run COLMAP/SfM for camera poses before final render."
        ),
        "user_context": {"location": user_location},
        "vlm_scene_understanding": {
            "model": model,
            "selected_frames": selected_frames,
            "raw_response": raw_response,
            "parsed_response": parsed,
        },
    }


def main() -> int:
    args = parse_args()
    work = Path(args.workdir)
    frames = sorted((work / "frames").glob("*.jpg"))
    if not frames:
        raise SystemExit(f"No frames found under {work / 'frames'}")
    stride = max(1, len(frames) // args.max_frames)
    selected = frames[::stride][: args.max_frames]

    prompt = {
        "task": (
            "You are a VFX supervisor analyzing keyframes for a Blender CLI matchmove pipeline. "
            "Study the scene carefully and propose ONE specific CGI object to insert that fits the "
            "environment, lighting mood, and available empty space."
        ),
        "known_location": args.user_location,
        "instructions": [
            "Identify the time of day, lighting color temperature, and environment type.",
            "Find safe empty regions where a CGI object can be placed without landing on people or occluders.",
            "Propose a small-to-medium CGI object that would plausibly exist or visit this location.",
            "The object must move, rotate, or articulate — it must not be static.",
            "Choose appearance (color, glow) that matches the scene's lighting mood.",
            "screen_path: 3-5 [x,y] pairs in normalized screen coords (0=left/top, 1=right/bottom).",
        ],
        "format": "Return one compact valid JSON object only. No markdown fences. No extra keys.",
        "schema": {
            "scene_summary": "1-2 sentences describing the scene",
            "camera_motion": "1 sentence",
            "lighting_notes": "time of day, color temperature, indoor/outdoor",
            "objects_and_regions": ["observed object labels"],
            "occlusion_risks": ["labels the CGI must avoid"],
            "safe_insertion_regions": ["screen region descriptions"],
            "suggested_cgi": {
                "object": "specific CGI object name fitting the scene (e.g. 'glowing lantern drone', 'holographic billboard', 'small delivery robot', 'neon butterfly')",
                "reasoning": "1 sentence: why this object fits this scene and lighting",
                "action": "short narrative of the object's movement through the scene",
                "screen_path": [[0.2, 0.75], [0.5, 0.65], [0.75, 0.7]],
            },
            "segmentation_prompts": ["person", "railing", "tree", "ground", "sky"],
        },
    }

    content = [{"type": "image", "image": Image.open(path).convert("RGB")} for path in selected]
    content.append({"type": "text", "text": json.dumps(prompt, indent=2)})
    messages = [{"role": "user", "content": content}]

    processor = AutoProcessor.from_pretrained(args.model)
    model_obj = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model_obj.device)

    generated = model_obj.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated, strict=True)
    ]
    raw_response = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    parsed = _parse_json_response(raw_response)
    selected_strs = [str(p) for p in selected]

    # Write scene_brief.json — the authoritative structured output consumed by merge_perception_outputs.py
    scene_brief = _synthesize_scene_brief(parsed, raw_response, args.model, selected_strs, args.user_location)
    scene_brief_path = work / "scene_brief.json"
    scene_brief_path.write_text(json.dumps(scene_brief, indent=2) + "\n", encoding="utf-8")

    # Write vlm_scene_qwen.json for backward compatibility
    legacy = {
        "model": args.model,
        "selected_frames": selected_strs,
        "user_location": args.user_location,
        "raw_response": raw_response,
        "parsed_response": parsed,
    }
    (work / "vlm_scene_qwen.json").write_text(json.dumps(legacy, indent=2) + "\n", encoding="utf-8")

    print(scene_brief_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
