from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .io import read_json, write_json
from .perception import build_context
from .planner import make_plan
from .qa import qa_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI-assisted Blender matchmove pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Extract keyframes and build perception context")
    prepare.add_argument("--video", required=True)
    prepare.add_argument("--workdir", default="work")
    prepare.add_argument("--config", default="config.json")

    story = sub.add_parser("story", help="Call Qwen to generate multi-object CGI story plan")
    story.add_argument("--workdir", default="work")
    story.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")

    search = sub.add_parser("search", help="CLIP semantic search for 3D assets on Objaverse")
    search.add_argument("--workdir", default="work")
    search.add_argument("--top-cats", type=int, default=4)
    search.add_argument("--per-cat", type=int, default=3)

    plan = sub.add_parser("plan", help="Create CGI insertion plan from perception context")
    plan.add_argument("--workdir", default="work")
    plan.add_argument("--config", default="config.json")

    render = sub.add_parser("render", help="Render the planned CGI insertion with Blender")
    render.add_argument("--workdir", default="work")
    render.add_argument("--config", default="config.json")
    render.add_argument("--blender", default="blender")
    render.add_argument("--output", default="result.mp4")
    render.add_argument("--max-duration", type=float, default=0.0,
                        help="Clamp render to this many seconds (0 = full video)")
    render.add_argument("--samples", type=int, default=0,
                        help="Override render samples (0 = use plan default)")

    qa = sub.add_parser("qa", help="Sample rendered video and write QA report")
    qa.add_argument("--video", required=True)
    qa.add_argument("--workdir", default="work")
    qa.add_argument("--config", default="config.json")

    args = parser.parse_args(argv)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if args.command == "prepare":
        config = read_json(args.config)
        context = build_context(args.video, workdir, config)
        write_json(workdir / "perception_context.json", context)
        print(workdir / "perception_context.json")
        return 0

    if args.command == "story":
        script = Path(__file__).resolve().parent.parent / "scripts" / "qwen_story_planner.py"
        cmd = ["python3", str(script), "--workdir", str(workdir), "--model", args.model]
        subprocess.run(cmd, check=True)
        return 0

    if args.command == "search":
        script = Path(__file__).resolve().parent.parent / "scripts" / "asset_search.py"
        cmd = [
            "python3", str(script),
            "--workdir", str(workdir),
            "--story-plan", str(workdir / "story_plan.json"),
            "--top-cats", str(args.top_cats),
            "--per-cat", str(args.per_cat),
        ]
        subprocess.run(cmd, check=True)
        # Merge glb_path into story_plan objects so the render step picks them up
        manifest_path = workdir / "asset_manifest.json"
        story_path = workdir / "story_plan.json"
        if manifest_path.exists() and story_path.exists():
            manifest = read_json(manifest_path)
            story = read_json(story_path)
            for obj in story.get("objects", []):
                entry = manifest.get(obj.get("id", ""), {})
                obj["glb_path"] = entry.get("glb_path")  # None if not found
            write_json(story_path, story)
            print(f"GLB paths merged into {story_path}")
        return 0

    if args.command == "plan":
        config = read_json(args.config)
        context = read_json(workdir / "perception_context.json")
        story_plan = _maybe_read(workdir / "story_plan.json")
        cgi_plan = make_plan(context, config, story_plan=story_plan)
        write_json(workdir / "cgi_plan.json", cgi_plan)
        print(workdir / "cgi_plan.json")
        return 0

    if args.command == "render":
        manifest = workdir / "perception_context.json"
        cgi_plan = workdir / "cgi_plan.json"
        script = Path(__file__).resolve().parent.parent / "blender" / "insert_cgi.py"
        cmd = [
            args.blender,
            "--background",
            "--python", str(script),
            "--",
            "--manifest", str(manifest),
            "--plan", str(cgi_plan),
            "--output", args.output,
        ]
        if args.max_duration > 0:
            cmd += ["--max-duration", str(args.max_duration)]
        if args.samples > 0:
            cmd += ["--samples", str(args.samples)]
        subprocess.run(cmd, check=True)
        return 0

    if args.command == "qa":
        config = read_json(args.config)
        report = qa_report(args.video, workdir, config)
        write_json(workdir / "qa_report.json", report)
        print(workdir / "qa_report.json")
        return 0

    return 2


def _maybe_read(path: Path) -> dict | None:
    if path.exists():
        return read_json(path)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
