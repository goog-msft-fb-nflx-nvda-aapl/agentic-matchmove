from __future__ import annotations

from statistics import median


def make_plan(context: dict, config: dict) -> dict:
    video = context["video"]
    cgi = config.get("cgi", {})
    path = _choose_screen_path(context, cgi)
    duration = min(float(video.get("duration_seconds", 0) or 0), float(config.get("max_video_seconds", 60)))
    if duration <= 0:
        duration = max(1.0, len(context.get("frames", [])) * float(config.get("keyframe_stride_seconds", 1.0)))

    tracking = config.get("tracking", {})
    audio = config.get("audio", {})
    return {
        "version": 1,
        "intent": cgi.get("story_intent", "Insert a small animated CGI object into the scene."),
        "asset": cgi.get("asset", "procedural_robot"),
        "tracking": {
            "source": tracking.get("source"),
            "camera_poses_path": tracking.get("camera_poses_path"),
            "blender_tracking_file": tracking.get("blender_tracking_file"),
            "status": "required_for_final" if not tracking.get("source") else "provided",
        },
        "video": video,
        "animation": {
            "duration_seconds": duration,
            "screen_path": path,
            "scale": float(cgi.get("scale", 0.5)),
            "rotation_turns": 1.0,
        },
        "cgi_features": [
            "multi-part procedural robot",
            "eye geometry",
            "arm geometry",
            "translation",
            "rotation",
            "emissive material",
        ],
        "appearance": {
            "color": cgi.get("color", [0.1, 0.55, 1.0]),
            "emission_strength": float(cgi.get("emission_strength", 0.8)),
        },
        "render": {
            "uses_lights": True,
            "uses_shadow_catcher": True,
            "background_video_composite": True,
        },
        "audio": {
            "music_path": audio.get("music_path"),
            "sound_effects": audio.get("sound_effects", []),
            "status": "required_for_final" if not audio else "provided",
        },
        "constraints": _summarize_constraints(context),
        "qa_expectations": config.get("qa", {}),
    }


def _choose_screen_path(context: dict, cgi: dict) -> list[list[float]]:
    preferred = cgi.get("preferred_screen_path")
    if preferred:
        return preferred

    occupied_x = []
    occupied_y = []
    width = max(1, int(context.get("video", {}).get("width", 1)))
    height = max(1, int(context.get("video", {}).get("height", 1)))
    for frame in context.get("frames", []):
        for inst in frame.get("instances", []):
            x1, y1, x2, y2 = inst.get("bbox_xyxy", [0, 0, 0, 0])
            occupied_x.append(((x1 + x2) * 0.5) / width)
            occupied_y.append(((y1 + y2) * 0.5) / height)

    if not occupied_x:
        return [[0.25, 0.75], [0.50, 0.62], [0.75, 0.72]]

    center_x = median(occupied_x)
    y = min(0.82, max(0.58, median(occupied_y) + 0.20))
    if center_x < 0.5:
        return [[0.65, y], [0.50, y - 0.08], [0.28, y]]
    return [[0.25, y], [0.48, y - 0.08], [0.72, y]]


def _summarize_constraints(context: dict) -> dict:
    labels: dict[str, int] = {}
    tracks: set[str] = set()
    for frame in context.get("frames", []):
        for inst in frame.get("instances", []):
            label = inst.get("label", "object")
            labels[label] = labels.get(label, 0) + 1
            if inst.get("track_id"):
                tracks.add(inst["track_id"])
    return {
        "avoid_labels": sorted(label for label in labels if label in {"person", "face", "hand", "animal"}),
        "observed_labels": labels,
        "track_count": len(tracks),
    }
