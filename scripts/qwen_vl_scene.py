#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--user-location", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work = Path(args.workdir)
    frames = sorted((work / "frames").glob("*.jpg"))
    if not frames:
        raise SystemExit(f"No frames found under {work / 'frames'}")
    stride = max(1, len(frames) // args.max_frames)
    selected = frames[::stride][: args.max_frames]

    prompt = {
        "task": "Analyze these keyframes for a Blender CLI matchmove/VFX agent.",
        "known_location": args.user_location,
        "format": "Return one compact valid JSON object only. No markdown fences.",
        "schema": {
            "scene_summary": "1-2 sentences",
            "camera_motion": "1 sentence",
            "objects_and_regions": ["short labels"],
            "occlusion_risks": ["short labels"],
            "safe_insertion_regions": ["short labels"],
            "suggested_cgi": {
                "object": "short name",
                "action": "short action",
                "screen_path": [[0.2, 0.75], [0.5, 0.65], [0.75, 0.7]],
            },
            "tracking_requirements": ["short labels"],
            "segmentation_prompts": ["person", "railing", "tree", "ground", "sky"],
            "quality_checks": ["short labels"],
        },
    }

    content = [{"type": "image", "image": Image.open(path).convert("RGB")} for path in selected]
    content.append({"type": "text", "text": json.dumps(prompt, indent=2)})
    messages = [{"role": "user", "content": content}]

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
    ).to(model.device)

    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated, strict=True)
    ]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    out = {
        "model": args.model,
        "selected_frames": [str(path) for path in selected],
        "user_location": args.user_location,
        "raw_response": decoded,
    }
    output_path = work / "vlm_scene_qwen.json"
    output_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
