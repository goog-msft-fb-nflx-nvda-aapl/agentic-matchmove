from __future__ import annotations


REQUIRED_ITEMS = [
    "camera_tracking",
    "cgi_complexity",
    "cgi_action",
    "lighting_shadow",
    "storyline",
    "audio",
]


def quality_gate_status(context: dict | None, plan: dict | None, config: dict) -> dict:
    context = context or {}
    plan = plan or {}
    checks = {
        "camera_tracking": _tracking_check(context, plan),
        "cgi_complexity": _cgi_complexity_check(plan, config),
        "cgi_action": _cgi_action_check(plan),
        "lighting_shadow": _lighting_shadow_check(plan),
        "storyline": _storyline_check(plan, config),
        "audio": _audio_check(plan, config),
    }
    return {
        "all_required_satisfied": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "notes": [
            "Camera tracking must be backed by Blender tracking, COLMAP/SfM, or another camera-pose source.",
            "Detection/segmentation tracks provide semantic context but do not replace camera matchmove tracking.",
            "A production candidate should include visible object motion, nontrivial CGI, shadows/lighting, story intent, and audio.",
        ],
    }


def _tracking_check(context: dict, plan: dict) -> dict:
    tracking = plan.get("tracking", {}) or context.get("tracking", {})
    source = tracking.get("source")
    has_camera_poses = bool(tracking.get("camera_poses_path") or tracking.get("blender_tracking_file"))
    return {
        "passed": bool(source and has_camera_poses),
        "source": source or "missing",
        "evidence": tracking,
        "required_next_step": "Run Blender camera tracking or COLMAP/SfM and attach camera pose evidence.",
    }


def _cgi_complexity_check(plan: dict, config: dict) -> dict:
    cgi = config.get("cgi", {})
    asset = plan.get("asset") or cgi.get("asset")
    features = plan.get("cgi_features", [])
    passed = asset not in {None, "", "cube"} and len(features) >= 3
    return {
        "passed": passed,
        "asset": asset or "missing",
        "features": features,
        "required_next_step": "Use a multi-part/animated asset, e.g. body, eyes, arms, glow, head turn.",
    }


def _cgi_action_check(plan: dict) -> dict:
    animation = plan.get("animation", {})
    screen_path = animation.get("screen_path", [])
    rotation_turns = float(animation.get("rotation_turns", 0) or 0)
    passed = len(screen_path) >= 2 and (screen_path[0] != screen_path[-1] or rotation_turns > 0)
    return {
        "passed": passed,
        "screen_path_points": len(screen_path),
        "rotation_turns": rotation_turns,
        "required_next_step": "Animate translation, rotation, or articulation over time.",
    }


def _lighting_shadow_check(plan: dict) -> dict:
    render = plan.get("render", {})
    passed = bool(render.get("uses_lights") and render.get("uses_shadow_catcher"))
    return {
        "passed": passed,
        "evidence": render,
        "required_next_step": "Enable lights and a shadow catcher or contact shadow pass in Blender.",
    }


def _storyline_check(plan: dict, config: dict) -> dict:
    intent = plan.get("intent") or config.get("cgi", {}).get("story_intent", "")
    passed = len(intent.strip()) >= 20
    return {
        "passed": passed,
        "intent": intent,
        "required_next_step": "Define a short story action for the inserted CGI.",
    }


def _audio_check(plan: dict, config: dict) -> dict:
    audio = plan.get("audio", {}) or config.get("audio", {})
    passed = bool(audio.get("music_path") or audio.get("sound_effects"))
    return {
        "passed": passed,
        "evidence": audio,
        "required_next_step": "Add music or sound effects before final output.",
    }

