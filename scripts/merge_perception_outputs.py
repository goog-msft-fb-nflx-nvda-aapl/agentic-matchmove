#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge VLM, detection, and SAM2 outputs into perception context")
    parser.add_argument("--workdir", default="work")
    parser.add_argument("--location", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work = Path(args.workdir)
    context_path = work / "perception_context.json"
    segments_path = work / "sam2_segments.json"
    scene_path = work / "scene_brief.json"

    context = json.loads(context_path.read_text())
    by_image = {frame["image_path"]: frame for frame in context["frames"]}

    if segments_path.exists():
        segments = json.loads(segments_path.read_text())
        for segment_frame in segments["frames"]:
            context_frame = by_image.get(segment_frame["image_path"])
            if not context_frame:
                continue
            instances = []
            for idx, segment in enumerate(segment_frame["segments"]):
                label_slug = segment["label"].replace(" ", "_")
                instances.append(
                    {
                        "track_id": f"{label_slug}_{idx:02d}_{context_frame['frame_index']:06d}",
                        "label": segment["label"],
                        "confidence": segment["confidence"],
                        "bbox_xyxy": segment["bbox_xyxy"],
                        "mask_path": segment["mask_path"],
                        "attributes": {
                            "source": "GroundingDINO+SAM2",
                            "mask_area_px": segment["mask_area_px"],
                        },
                    }
                )
            context_frame["instances"] = instances

    if scene_path.exists():
        scene = json.loads(scene_path.read_text())
        context["scene_brief_path"] = str(scene_path)
        context["scene_summary"] = scene.get("scene_summary", {})
        context["vlm_scene_understanding"] = scene.get("vlm_scene_understanding", {})

    context.setdefault("notes", []).append(
        "Merged Qwen2.5-VL scene understanding, Grounding DINO detections, and SAM2 masks."
    )
    context["models"] = {
        "vlm": "Qwen/Qwen2.5-VL-7B-Instruct",
        "detector": "IDEA-Research/grounding-dino-base",
        "segmenter": "facebook/sam2-hiera-large",
    }
    if args.location:
        context["user_context"] = {"location": args.location}

    context_path.write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
    frames_with_instances = sum(1 for frame in context["frames"] if frame.get("instances"))
    instance_count = sum(len(frame.get("instances", [])) for frame in context["frames"])
    print(context_path)
    print(f"frames_with_instances={frames_with_instances}")
    print(f"instances={instance_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

