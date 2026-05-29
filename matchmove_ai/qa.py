from __future__ import annotations

from pathlib import Path

from .io import read_json
from .quality_gate import quality_gate_status
from .video import extract_keyframes, probe_video


def qa_report(video_path: str | Path, workdir: str | Path, config: dict) -> dict:
    work = Path(workdir)
    video = probe_video(video_path)
    frames = extract_keyframes(video_path, work / "qa_frames", max(1.0, float(config.get("keyframe_stride_seconds", 1.0))))
    expected = config.get("qa", {}).get("expected_inserted_object", "inserted CGI object")
    context = _maybe_read(work / "perception_context.json")
    plan = _maybe_read(work / "cgi_plan.json")
    quality_gate = quality_gate_status(context, plan, config)
    return {
        "video": video.__dict__,
        "sampled_frames": [str(path) for _, _, path in frames],
        "status": "needs_human_review" if quality_gate["all_required_satisfied"] else "missing_quality_requirements",
        "quality_gate": quality_gate,
        "checks": [
            {
                "name": "vlm_visibility_prompt",
                "prompt": (
                    f"Inspect these sampled frames. Is the {expected} visible, temporally stable, "
                    "well-positioned, and not incorrectly floating or occluding foreground objects?"
                ),
            },
            {
                "name": "duration_limit",
                "passed": video.duration_seconds <= float(config.get("max_video_seconds", 60)),
            },
        ],
    }


def _maybe_read(path: Path) -> dict | None:
    if not path.exists():
        return None
    return read_json(path)
