#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM2 segmentation from Grounding DINO boxes")
    parser.add_argument("--workdir", default="work")
    parser.add_argument("--model", default="facebook/sam2-hiera-large")
    parser.add_argument("--detections", default="work/grounding_dino_detections.json")
    parser.add_argument("--max-boxes-per-frame", type=int, default=8)
    parser.add_argument("--min-confidence", type=float, default=0.30)
    parser.add_argument(
        "--labels",
        default="person,railing,tree,tent,sign,ground,ground walkway,walkway,stairs,clock,arch",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work = Path(args.workdir)
    data = json.loads(Path(args.detections).read_text())
    labels = {label.strip() for label in args.labels.split(",") if label.strip()}
    mask_dir = work / "sam2_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Sam2Processor.from_pretrained(args.model)
    model = Sam2Model.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    rows = []
    for frame_idx, frame in enumerate(data["frames"]):
        candidates = [
            det
            for det in frame["detections"]
            if det["confidence"] >= args.min_confidence and det["label"] in labels
        ][: args.max_boxes_per_frame]
        if not candidates:
            rows.append({"image_path": frame["image_path"], "segments": []})
            continue

        image = Image.open(frame["image_path"]).convert("RGB")
        boxes = [det["bbox_xyxy"] for det in candidates]
        inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt").to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            outputs = model(**inputs, multimask_output=False)
        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        masks = masks.squeeze(1).numpy() > 0

        segments = []
        for i, (det, mask) in enumerate(zip(candidates, masks, strict=False)):
            mask_path = mask_dir / f"frame_{frame_idx:03d}_{i:02d}_{det['label'].replace(' ', '_')}.png"
            cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))
            segments.append(
                {
                    "label": det["label"],
                    "confidence": det["confidence"],
                    "bbox_xyxy": det["bbox_xyxy"],
                    "mask_path": str(mask_path),
                    "mask_area_px": int(mask.sum()),
                }
            )
        rows.append({"image_path": frame["image_path"], "segments": segments})

    out = {"model": args.model, "source_detections": args.detections, "frames": rows}
    out_path = work / "sam2_segments.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

