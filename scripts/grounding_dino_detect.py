#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Grounding DINO open-vocabulary detection on keyframes")
    parser.add_argument("--workdir", default="work")
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--prompts",
        default="person. railing. tree. ground. sky. sign. tent. building. stairs. walkway. clock. arch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work = Path(args.workdir)
    frames = sorted((work / "frames").glob("*.jpg"))
    if not frames:
        raise SystemExit(f"No frames found under {work / 'frames'}")
    stride = max(1, len(frames) // args.max_frames)
    selected = frames[::stride][: args.max_frames]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device)
    model.eval()

    rows = []
    for path in selected:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, text=args.prompts, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        detections = []
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("labels", [])
        for box, score, label in zip(boxes, scores, labels, strict=False):
            detections.append(
                {
                    "label": str(label),
                    "confidence": float(score),
                    "bbox_xyxy": [round(float(v), 2) for v in box.tolist()],
                }
            )
        rows.append({"image_path": str(path), "detections": detections})

    output = {
        "model": args.model,
        "prompts": args.prompts,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "frames": rows,
    }
    out_path = work / "grounding_dino_detections.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
