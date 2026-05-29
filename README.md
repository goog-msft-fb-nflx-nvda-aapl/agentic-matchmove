# Agentic MatchMove

Agentic MatchMove is a remote-first VFX pipeline where a vision-language agent understands an input video, uses segmentation/detection/tracking context, controls Blender through CLI/Python APIs, renders inserted CGI, and self-checks the result before asking a human to inspect it.

The project is designed for non-GUI execution on a GPU server. Human interaction should be limited to uploading input media, editing optional intent/config fields, and reviewing final candidate videos.

## Core Idea

```text
video
  -> keyframes
  -> VLM scene understanding
  -> detection / segmentation / instance tracks
  -> camera tracking / SfM
  -> LLM/VLM CGI plan
  -> Blender CLI/API render
  -> VLM + deterministic QA
  -> final candidate
```

Context comes from models. Actions happen through tools:

- **Context**: VLM captions, object detections, segmentation masks, instance tracks, camera poses, optional user intent.
- **Tool**: Blender CLI/Python API for assets, cameras, lights, animation, compositing, and rendering.
- **Agent loop**: plan, execute, render, sample output, critique, revise.

## Model Stack

Recommended interchangeable components:

- **Segmentation/tracking**: Meta SAM2 or IDEA-Research Grounded-SAM-2.
- **Open-vocabulary detection**: Grounding DINO / Grounding DINO 1.5 / Florence-2 / YOLO-World.
- **3D/camera pose**: Blender camera tracking, COLMAP/SfM, or experimental fast models such as VGGT.
- **VLM planning/QA**: Qwen2.5-VL, InternVL, LLaVA-OneVision, GPT-4.1/4o-class APIs, or any local model able to inspect keyframes.
- **Rendering**: Blender in `--background` mode with Python scripts.

## Quality Gates

The final `qa_report.json` checks whether a render is a real VFX candidate rather than just generated pixels:

- `camera_tracking`: camera tracking or SfM evidence is required.
- `cgi_complexity`: inserted CGI should be more than a static primitive.
- `cgi_action`: the CGI must visibly move, rotate, or articulate.
- `lighting_shadow`: lighting and shadow/contact with the scene must be present.
- `storyline`: the plan must include a clear short story action.
- `audio`: music or sound effects should be attached before final output.

Intermediate runs are allowed to be incomplete, but the report stays `missing_quality_requirements` until these gates have evidence.

## Remote Safety Gate

Before downloads, installs, model inference, or rendering on the GPU server:

```bash
cd /home/jtan/matchmove_ai
REQUIRE_IDLE_GPU=1 scripts/remote_safe_check.sh
```

Default thresholds:

- at least 64 GB available RAM
- at least 50 GB free disk under `$HOME`
- no more than 25% swap used
- optionally at least one idle GPU when `REQUIRE_IDLE_GPU=1`

Tune thresholds per job:

```bash
MIN_DISK_GB=120 REQUIRE_IDLE_GPU=1 scripts/remote_safe_check.sh
```

## Folder Layout

```text
matchmove_ai/            Python orchestration package
blender/insert_cgi.py    Blender script executed by the pipeline
scripts/matchmove_ai.py  CLI wrapper
examples/config.json     Example config
```

## Quick Start

Put a video at `data/input.mp4`, then:

```bash
cp examples/config.json config.json
python3 scripts/matchmove_ai.py prepare --video data/input.mp4 --workdir work --config config.json
python3 scripts/matchmove_ai.py plan --workdir work --config config.json
python3 scripts/matchmove_ai.py render --workdir work --config config.json --blender /path/to/blender --output result.mp4
python3 scripts/matchmove_ai.py qa --video result.mp4 --workdir work --config config.json
```

Direct Blender invocation:

```bash
/path/to/blender --background --python blender/insert_cgi.py -- --manifest work/perception_context.json --plan work/cgi_plan.json --output result.mp4
```

## Perception Context

The pipeline expects normalized context at `work/perception_context.json`. A SAM2/Grounded-SAM2/VLM pipeline can export:

```json
{
  "video": {
    "path": "data/input.mp4",
    "fps": 30,
    "width": 1920,
    "height": 1080,
    "frame_count": 300,
    "duration_seconds": 10.0
  },
  "frames": [
    {
      "frame_index": 0,
      "timestamp": 0.0,
      "image_path": "work/frames/frame_000000.jpg",
      "caption": "indoor desk scene with a visible floor region",
      "instances": [
        {
          "track_id": "person_1",
          "label": "person",
          "confidence": 0.91,
          "bbox_xyxy": [300, 120, 820, 980],
          "mask_path": "work/masks/person_1_000000.png",
          "attributes": {"moving": true}
        }
      ]
    }
  ],
  "tracking": {
    "source": "colmap",
    "camera_poses_path": "work/tracking/cameras.json"
  }
}
```

Detection and segmentation provide semantics and placement constraints. Camera tracking/SfM provides the geometry needed for a proper matchmove composite.

## GitHub

Suggested repository name:

```text
goog-msft-fb-nflx-nvda-aapl/agentic-matchmove
```

