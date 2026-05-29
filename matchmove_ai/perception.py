from __future__ import annotations

from pathlib import Path

from .io import read_json
from .schema import FrameContext, PerceptionContext, VideoInfo
from .video import extract_keyframes, probe_video


def build_context(video_path: str | Path, workdir: str | Path, config: dict) -> dict:
    work = Path(workdir)
    external_json = config.get("perception", {}).get("external_json")
    if external_json:
        return normalize_external_context(external_json, video_path)

    video = probe_video(video_path)
    max_seconds = float(config.get("max_video_seconds", 60))
    if video.duration_seconds > max_seconds:
        raise ValueError(
            f"Input is {video.duration_seconds:.1f}s, exceeding configured limit {max_seconds:.1f}s"
        )

    stride = float(config.get("keyframe_stride_seconds", 1.0))
    frame_rows = extract_keyframes(video_path, work / "frames", stride)
    prompt_text = ", ".join(config.get("perception", {}).get("target_prompts", []))
    frames = [
        FrameContext(
            frame_index=frame_index,
            timestamp=timestamp,
            image_path=str(frame_path),
            caption=f"Keyframe reserved for VLM/SAM2 analysis. Suggested prompts: {prompt_text}",
        )
        for frame_index, timestamp, frame_path in frame_rows
    ]
    context = PerceptionContext(
        video=video,
        frames=frames,
        notes=[
            "No external detector JSON was configured; generated keyframe context only.",
            "Run Grounded-SAM2/SAM2 or a VLM on these frames, then fill instances with boxes/masks.",
        ],
    )
    return _context_to_dict(context)


def normalize_external_context(external_json: str | Path, fallback_video_path: str | Path) -> dict:
    data = read_json(external_json)
    if "video" not in data:
        video = probe_video(fallback_video_path)
        data["video"] = video.__dict__
    data.setdefault("frames", [])
    data.setdefault("notes", [])
    data["notes"].append(f"Loaded perception context from {external_json}")
    return data


def _context_to_dict(context: PerceptionContext) -> dict:
    return {
        "video": context.video.__dict__,
        "frames": [
            {
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
                "image_path": frame.image_path,
                "caption": frame.caption,
                "instances": [instance.__dict__ for instance in frame.instances],
            }
            for frame in context.frames
        ],
        "notes": context.notes,
    }

