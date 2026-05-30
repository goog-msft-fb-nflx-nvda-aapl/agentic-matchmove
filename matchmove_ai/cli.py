from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .io import read_json, write_json
from .perception import build_context
from .planner import make_plan
from .qa import qa_report

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_BLENDER_DIR = Path(__file__).resolve().parent.parent / "blender"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agentic matchmove pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── run: single command for any video ────────────────────────────────────
    run_p = sub.add_parser("run", help="Full pipeline: video → story → search → render")
    run_p.add_argument("--video", required=True)
    run_p.add_argument("--workdir", required=True)
    run_p.add_argument("--location", default="", help="User-provided location hint")
    run_p.add_argument("--duration", type=float, default=3.0, help="Render duration (seconds)")
    run_p.add_argument("--blender", default="blender")
    run_p.add_argument("--output", default="")
    run_p.add_argument("--samples", type=int, default=16)
    run_p.add_argument("--max-video-seconds", type=float, default=60.0)
    run_p.add_argument("--skip-perception", action="store_true",
                       help="Skip prepare/qwen_scene/annotate if already done")

    # ── individual steps ──────────────────────────────────────────────────────
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--video", required=True)
    prepare.add_argument("--workdir", default="work")
    prepare.add_argument("--config", default="config.json")

    story = sub.add_parser("story")
    story.add_argument("--workdir", default="work")
    story.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")

    search = sub.add_parser("search")
    search.add_argument("--workdir", default="work")
    search.add_argument("--top-cats", type=int, default=4)
    search.add_argument("--per-cat", type=int, default=3)

    plan = sub.add_parser("plan")
    plan.add_argument("--workdir", default="work")
    plan.add_argument("--config", default="config.json")

    render = sub.add_parser("render")
    render.add_argument("--workdir", default="work")
    render.add_argument("--config", default="config.json")
    render.add_argument("--blender", default="blender")
    render.add_argument("--output", default="result.mp4")
    render.add_argument("--max-duration", type=float, default=0.0)
    render.add_argument("--samples", type=int, default=16)

    qa_p = sub.add_parser("qa")
    qa_p.add_argument("--video", required=True)
    qa_p.add_argument("--workdir", default="work")
    qa_p.add_argument("--config", default="config.json")

    args = parser.parse_args(argv)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # ── run: orchestrate everything ───────────────────────────────────────────
    if args.command == "run":
        return _run_pipeline(args, workdir)

    if args.command == "prepare":
        config = read_json(args.config)
        context = build_context(args.video, workdir, config)
        write_json(workdir / "perception_context.json", context)
        print(workdir / "perception_context.json")
        return 0

    if args.command == "story":
        return _run_script("qwen_story_planner.py",
                           ["--workdir", str(workdir), "--model", args.model])

    if args.command == "search":
        rc = _run_script("asset_search.py", [
            "--workdir", str(workdir),
            "--story-plan", str(workdir / "story_plan.json"),
            "--top-cats", str(args.top_cats),
            "--per-cat", str(args.per_cat),
        ])
        _merge_glb_paths(workdir)
        return rc

    if args.command == "plan":
        config = read_json(args.config)
        context = read_json(workdir / "perception_context.json")
        story_plan = _maybe_read(workdir / "story_plan.json")
        write_json(workdir / "cgi_plan.json",
                   make_plan(context, config, story_plan=story_plan))
        print(workdir / "cgi_plan.json")
        return 0

    if args.command == "render":
        return _blender_render(
            blender=args.blender,
            workdir=workdir,
            output=args.output,
            max_duration=args.max_duration,
            samples=args.samples,
        )

    if args.command == "qa":
        config = read_json(args.config)
        report = qa_report(args.video, workdir, config)
        write_json(workdir / "qa_report.json", report)
        print(workdir / "qa_report.json")
        return 0

    return 2


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------

def _run_pipeline(args: argparse.Namespace, workdir: Path) -> int:
    blender = args.blender
    location = args.location
    py = sys.executable
    output = args.output or str(workdir / "result.mp4")

    def health_check(step: str) -> None:
        import shutil
        free_gb = shutil.disk_usage(workdir).free // (1024 ** 3)
        if free_gb < 20:
            raise RuntimeError(f"[{step}] ABORT: only {free_gb}GB disk free")
        print(f"[health:{step}] disk_free={free_gb}GB  ok")

    if not args.skip_perception:
        # 1. Write a minimal config
        config_path = workdir / "run_config.json"
        _write_run_config(config_path, args.max_video_seconds)

        health_check("prepare")
        ctx = build_context(args.video, workdir, read_json(config_path))
        write_json(workdir / "perception_context.json", ctx)

        health_check("qwen_scene")
        _run_script("qwen_vl_scene.py",
                    ["--workdir", str(workdir), "--user-location", location])

        health_check("merge")
        _run_script("merge_perception_outputs.py",
                    ["--workdir", str(workdir), "--location", location])

        health_check("annotate")
        _run_script("annotate_frames.py",
                    ["--workdir", str(workdir), "--max-frames", "16"])
    else:
        config_path = workdir / "run_config.json"
        if not config_path.exists():
            _write_run_config(config_path, args.max_video_seconds)

    health_check("story")
    _run_script("qwen_story_planner.py", ["--workdir", str(workdir)])

    health_check("search")
    _run_script("asset_search.py", [
        "--workdir", str(workdir),
        "--story-plan", str(workdir / "story_plan.json"),
        "--top-cats", "4", "--per-cat", "3",
    ])
    _merge_glb_paths(workdir)

    health_check("plan")
    ctx = read_json(workdir / "perception_context.json")
    cfg = read_json(config_path)
    story_plan = _maybe_read(workdir / "story_plan.json")
    write_json(workdir / "cgi_plan.json", make_plan(ctx, cfg, story_plan=story_plan))

    health_check("render")
    rc = _blender_render(blender, workdir, output,
                         max_duration=args.duration, samples=args.samples)

    health_check("done")
    print(f"Output: {output}")
    return rc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_script(script_name: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, str(_SCRIPTS / script_name)] + extra_args
    result = subprocess.run(cmd)
    return result.returncode


def _blender_render(blender: str, workdir: Path, output: str,
                    max_duration: float, samples: int) -> int:
    cmd = [
        blender, "--background",
        "--python", str(_BLENDER_DIR / "insert_cgi.py"),
        "--",
        "--manifest", str(workdir / "perception_context.json"),
        "--plan", str(workdir / "cgi_plan.json"),
        "--output", output,
        "--samples", str(samples),
    ]
    if max_duration > 0:
        cmd += ["--max-duration", str(max_duration)]
    return subprocess.run(cmd).returncode


def _merge_glb_paths(workdir: Path) -> None:
    manifest_path = workdir / "asset_manifest.json"
    story_path = workdir / "story_plan.json"
    if manifest_path.exists() and story_path.exists():
        manifest = read_json(manifest_path)
        story = read_json(story_path)
        for obj in story.get("objects", []):
            obj["glb_path"] = manifest.get(obj.get("id", ""), {}).get("glb_path")
        write_json(story_path, story)


def _write_run_config(path: Path, max_video_seconds: float) -> None:
    write_json(path, {
        "project_name": path.parent.name,
        "max_video_seconds": max_video_seconds,
        "keyframe_stride_seconds": 1.0,
        "perception": {"backend": "manual_or_external_json", "external_json": None,
                       "target_prompts": ["person", "ground", "walkway", "road",
                                          "building", "sky", "sign", "car", "tree"]},
        "tracking": {"source": None, "camera_poses_path": None, "blender_tracking_file": None},
        "cgi": {"asset": None, "story_intent": None, "preferred_screen_path": None,
                "scale": 0.9, "color": None, "emission_strength": 2.0},
        "audio": {"music_path": None, "sound_effects": []},
        "qa": {"expected_inserted_object": "CGI character", "min_visible_frames_ratio": 0.65},
    })


def _maybe_read(path: Path) -> dict | None:
    return read_json(path) if path.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
