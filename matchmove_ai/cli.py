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

    plan = sub.add_parser("plan", help="Create CGI insertion plan from perception context")
    plan.add_argument("--workdir", default="work")
    plan.add_argument("--config", default="config.json")

    render = sub.add_parser("render", help="Render the planned CGI insertion with Blender")
    render.add_argument("--workdir", default="work")
    render.add_argument("--config", default="config.json")
    render.add_argument("--blender", default="blender")
    render.add_argument("--output", default="result.mp4")

    qa = sub.add_parser("qa", help="Sample rendered video and write QA prompt/report")
    qa.add_argument("--video", required=True)
    qa.add_argument("--workdir", default="work")
    qa.add_argument("--config", default="config.json")

    args = parser.parse_args(argv)
    config = read_json(args.config)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if args.command == "prepare":
        context = build_context(args.video, workdir, config)
        write_json(workdir / "perception_context.json", context)
        print(workdir / "perception_context.json")
        return 0

    if args.command == "plan":
        context = read_json(workdir / "perception_context.json")
        cgi_plan = make_plan(context, config)
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
            "--python",
            str(script),
            "--",
            "--manifest",
            str(manifest),
            "--plan",
            str(cgi_plan),
            "--output",
            args.output,
        ]
        subprocess.run(cmd, check=True)
        return 0

    if args.command == "qa":
        report = qa_report(args.video, workdir, config)
        write_json(workdir / "qa_report.json", report)
        print(workdir / "qa_report.json")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

