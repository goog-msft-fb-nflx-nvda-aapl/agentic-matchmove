from __future__ import annotations

import math
from statistics import median


def make_plan(context: dict, config: dict) -> dict:
    video = context["video"]
    cgi_cfg = config.get("cgi", {})
    vlm = _extract_vlm_suggestion(context)

    # Config is always the explicit override; VLM fills anything left unset.
    asset = cgi_cfg.get("asset") or _asset_from_name(vlm.get("object", "")) or "procedural_robot"
    intent = cgi_cfg.get("story_intent") or _build_intent(vlm) or "Insert a small animated CGI object into the scene."
    color = cgi_cfg.get("color") or _color_from_context(context, asset)
    emission = float(cgi_cfg.get("emission_strength", 0.8))

    duration = min(
        float(video.get("duration_seconds", 0) or 0),
        float(config.get("max_video_seconds", 60)),
    )
    if duration <= 0:
        duration = max(1.0, len(context.get("frames", [])) * float(config.get("keyframe_stride_seconds", 1.0)))

    path = _choose_screen_path(context, cgi_cfg, vlm, duration)
    tracking = _tracking_config(context, config)
    audio = config.get("audio", {})

    return {
        "version": 1,
        "intent": intent,
        "asset": asset,
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
            "scale": float(cgi_cfg.get("scale", 0.5)),
            "rotation_turns": 1.0,
        },
        "cgi_features": _features_for_asset(asset),
        "appearance": {
            "color": color,
            "emission_strength": emission,
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


# ---------------------------------------------------------------------------
# VLM extraction helpers
# ---------------------------------------------------------------------------

def _extract_vlm_suggestion(context: dict) -> dict:
    """Pull the best VLM CGI suggestion from the merged context."""
    # first_cgi_concept (from scene_brief.json) is the richest source
    concept = context.get("first_cgi_concept", {})
    if concept.get("object"):
        return {
            "object": concept["object"],
            "action": concept.get("proposed_action", ""),
            "screen_path": concept.get("screen_path_draft"),
            "lighting_notes": context.get("scene_summary", {}).get("lighting_notes", ""),
        }
    # Fallback: parsed VLM response embedded by merge_perception_outputs
    suggested = (
        context
        .get("vlm_scene_understanding", {})
        .get("parsed_response", {})
        .get("suggested_cgi", {})
    )
    if suggested.get("object"):
        return {
            "object": suggested.get("object", ""),
            "action": suggested.get("action", ""),
            "screen_path": suggested.get("screen_path"),
            "lighting_notes": (
                context
                .get("vlm_scene_understanding", {})
                .get("parsed_response", {})
                .get("lighting_notes", "")
            ),
        }
    return {}


def _asset_from_name(object_name: str) -> str:
    name = object_name.lower()
    if not name:
        return ""
    if any(k in name for k in ("robot", "delivery")):
        return "procedural_robot"
    if any(k in name for k in ("lantern", "drone", "orb", "sphere", "float", "hover")):
        return "lantern_drone"
    if any(k in name for k in ("holo", "billboard", "sign", "panel")):
        return "hologram_panel"
    # Unknown object from VLM — default to robot (most complex geometry)
    return "procedural_robot"


def _build_intent(vlm: dict) -> str:
    obj = vlm.get("object", "")
    action = vlm.get("action", "")
    if obj and action:
        return f"A {obj} {action}."
    if obj:
        return f"A {obj} enters the scene, moves through it, and exits."
    return ""


def _features_for_asset(asset: str) -> list[str]:
    if asset == "lantern_drone":
        return ["lantern body", "glow ring", "hover oscillation", "pulsing emission", "translation", "rotation"]
    if asset == "hologram_panel":
        return ["flat panel", "emissive hologram", "scan lines", "translation", "fade oscillation"]
    return ["multi-part procedural robot", "eye geometry", "arm geometry", "translation", "rotation", "emissive material"]


def _color_from_context(context: dict, asset: str) -> list[float]:
    notes = (context.get("scene_summary", {}).get("lighting_notes", "") or "").lower()
    if asset == "lantern_drone":
        # Warm amber for dusk/warm scenes, cool white for indoor/overcast
        if any(k in notes for k in ("dusk", "sunset", "warm", "golden")):
            return [1.0, 0.72, 0.22]
        return [0.92, 0.96, 1.0]
    if asset == "hologram_panel":
        return [0.0, 0.85, 1.0]
    # Robot: blue by default, warmer if warm lighting
    if any(k in notes for k in ("warm", "dusk", "sunset")):
        return [0.25, 0.60, 1.0]
    return [0.1, 0.55, 1.0]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _choose_screen_path(context: dict, cgi_cfg: dict, vlm: dict, duration: float) -> list[list[float]]:
    # Priority: explicit config > VLM suggestion > derived from detections > fallback
    preferred = cgi_cfg.get("preferred_screen_path")
    if preferred:
        return _extend_path(preferred, duration)

    vlm_path = vlm.get("screen_path")
    if vlm_path and len(vlm_path) >= 2:
        return _extend_path(vlm_path, duration)

    return _extend_path(_path_from_detections(context), duration)


def _path_from_detections(context: dict) -> list[list[float]]:
    occupied_x, occupied_y = [], []
    width = max(1, int(context.get("video", {}).get("width", 1)))
    height = max(1, int(context.get("video", {}).get("height", 1)))
    for frame in context.get("frames", []):
        for inst in frame.get("instances", []):
            x1, y1, x2, y2 = inst.get("bbox_xyxy", [0, 0, 0, 0])
            occupied_x.append(((x1 + x2) * 0.5) / width)
            occupied_y.append(((y1 + y2) * 0.5) / height)

    if not occupied_x:
        return [[0.22, 0.78], [0.45, 0.66], [0.72, 0.73]]

    cx = median(occupied_x)
    y = min(0.82, max(0.58, median(occupied_y) + 0.20))
    if cx < 0.5:
        return [[0.65, y], [0.50, y - 0.08], [0.28, y]]
    return [[0.25, y], [0.48, y - 0.08], [0.72, y]]


def _extend_path(base: list[list[float]], duration: float) -> list[list[float]]:
    """For videos longer than ~8s, interpolate extra waypoints so motion stays interesting."""
    target_points = max(len(base), min(12, 3 + int(duration / 6)))
    if len(base) >= target_points:
        return base

    # Linearly interpolate between existing points, then add gentle sine drift
    extended: list[list[float]] = []
    segs = len(base) - 1
    per_seg = math.ceil((target_points - len(base)) / max(1, segs))
    for i in range(segs):
        x0, y0 = base[i]
        x1, y1 = base[i + 1]
        extended.append([x0, y0])
        for j in range(1, per_seg + 1):
            t = j / (per_seg + 1)
            # Add small perpendicular sine drift so the path isn't a straight line
            drift = 0.03 * math.sin(math.pi * t)
            extended.append([
                round(x0 + t * (x1 - x0) + drift, 3),
                round(y0 + t * (y1 - y0), 3),
            ])
    extended.append(base[-1])
    return extended


# ---------------------------------------------------------------------------
# Tracking / constraints helpers
# ---------------------------------------------------------------------------

def _tracking_config(context: dict, config: dict) -> dict:
    configured = config.get("tracking", {})
    context_tracking = context.get("tracking", {})
    if configured.get("source"):
        return configured
    if context_tracking.get("source"):
        return context_tracking
    return configured


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
