from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = []
    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--max-duration", type=float, default=0.0)
    return parser.parse_args(argv)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def setup_scene(plan: dict, samples: int, max_duration: float) -> None:
    video = plan["video"]
    fps = float(video.get("fps") or 30)
    duration = float(plan["animation"]["duration_seconds"])
    if max_duration > 0:
        duration = min(duration, max_duration)
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = max(2, int(round(duration * fps)))
    bpy.context.scene.render.fps = int(round(fps))
    bpy.context.scene.render.resolution_x = int(video.get("width") or 1920)
    bpy.context.scene.render.resolution_y = int(video.get("height") or 1080)
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.view_settings.look = "Medium High Contrast"
    try:
        cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
        cycles_prefs.compute_device_type = "CUDA"
        cycles_prefs.get_devices()
        for device in cycles_prefs.devices:
            device.use = True
        bpy.context.scene.cycles.device = "GPU"
    except Exception:
        pass


def make_camera(video_path: str) -> bpy.types.Object:
    bpy.ops.object.camera_add(location=(0, -6, 2.4), rotation=(math.radians(68), 0, 0))
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    camera.data.lens = 28
    camera.data.dof.use_dof = True
    camera.data.dof.focus_distance = 6.0

    clip = bpy.data.movieclips.load(video_path)
    camera.data.show_background_images = True
    bg = camera.data.background_images.new()
    bg.source = "MOVIE_CLIP"
    bg.clip = clip
    bg.display_depth = "BACK"
    bg.alpha = 1.0
    return camera


def apply_sfm_camera_motion(camera: bpy.types.Object, plan: dict) -> None:
    pose_path = plan.get("tracking", {}).get("camera_poses_path")
    if not pose_path or not Path(pose_path).exists():
        return
    payload = load_json(pose_path)
    poses = payload.get("poses", [])
    if len(poses) < 2:
        return

    centers = [Vector(pose["projection_center"]) for pose in poses if pose.get("projection_center")]
    if len(centers) < 2:
        return
    mean = sum(centers, Vector((0, 0, 0))) / len(centers)
    centered = [center - mean for center in centers]
    max_extent = max(max(abs(v.x), abs(v.y), abs(v.z)) for v in centered) or 1.0
    scale = 1.35 / max_extent

    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end
    span = max(1, frame_end - frame_start)
    base = Vector((0, -6, 2.4))
    target = Vector((0, 0, 0.8))
    for idx, offset in enumerate(centered):
        frame = frame_start + int(round((idx / max(1, len(centered) - 1)) * span))
        camera.location = base + Vector((offset.x * scale, -offset.z * scale, offset.y * scale))
        direction = target - camera.location
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)

    if camera.animation_data and camera.animation_data.action:
        for fc in camera.animation_data.action.fcurves:
            for key in fc.keyframe_points:
                key.interpolation = "BEZIER"


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def make_material(name: str, color: list[float], emission_strength: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color[:3], 1.0)
        bsdf.inputs["Emission Color"].default_value = (*color[:3], 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission_strength
        bsdf.inputs["Roughness"].default_value = 0.42
    return mat


# ---------------------------------------------------------------------------
# CGI asset builders
# ---------------------------------------------------------------------------

def make_robot(plan: dict) -> bpy.types.Object:
    """Multi-part procedural robot: body sphere + eyes + arms."""
    appearance = plan["appearance"]
    mat = make_material("robot_blue_glow", appearance["color"], appearance["emission_strength"])

    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.55, location=(0, 0, 0.9))
    body = bpy.context.object
    body.name = "CGI_robot_body"
    body.scale = (0.75, 0.55, 0.95)
    body.data.materials.append(mat)

    dark = make_material("robot_dark_joints", [0.03, 0.035, 0.04], 0.0)
    for x in (-0.32, 0.32):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.09,
                                              location=(x, -0.46, 1.02))
        eye = bpy.context.object
        eye.name = "CGI_robot_eye"
        eye.data.materials.append(dark)
        eye.parent = body

    for x in (-0.58, 0.58):
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=16, radius=0.055, depth=0.8,
            location=(x, 0, 0.62),
            rotation=(0, math.radians(18 if x < 0 else -18), 0),
        )
        arm = bpy.context.object
        arm.name = "CGI_robot_arm"
        arm.data.materials.append(mat)
        arm.parent = body

    # Antenna
    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.025, depth=0.55,
                                         location=(0.18, 0, 1.55))
    antenna = bpy.context.object
    antenna.name = "CGI_robot_antenna"
    antenna.data.materials.append(mat)
    antenna.parent = body

    bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=6, radius=0.06,
                                          location=(0.18, 0, 1.85))
    tip = bpy.context.object
    tip.name = "CGI_robot_antenna_tip"
    tip.data.materials.append(make_material("antenna_tip", [1.0, 0.3, 0.1], 3.0))
    tip.parent = body

    return body


def make_lantern_drone(plan: dict) -> bpy.types.Object:
    """Glowing lantern-style drone: body sphere + equatorial ring + top cap."""
    appearance = plan["appearance"]
    color = appearance["color"]
    emission = appearance["emission_strength"]
    mat = make_material("lantern_glow", color, emission)
    ring_color = [min(1.0, c * 1.6) for c in color[:3]]
    ring_mat = make_material("lantern_ring", ring_color, emission * 1.5)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.42, location=(0, 0, 1.0))
    body = bpy.context.object
    body.name = "CGI_lantern_body"
    body.scale = (1.0, 1.0, 1.25)
    body.data.materials.append(mat)

    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.50, minor_radius=0.04,
        major_segments=48, minor_segments=12,
        location=(0, 0, 1.0),
    )
    ring = bpy.context.object
    ring.name = "CGI_lantern_ring"
    ring.data.materials.append(ring_mat)
    ring.parent = body

    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.065, depth=0.16,
                                         location=(0, 0, 1.56))
    cap = bpy.context.object
    cap.name = "CGI_lantern_cap"
    cap.data.materials.append(mat)
    cap.parent = body

    # Dangling chain links (3 small tori below the body)
    chain_mat = make_material("lantern_chain", [0.12, 0.10, 0.08], 0.0)
    for i, z in enumerate([0.52, 0.42, 0.34]):
        bpy.ops.mesh.primitive_torus_add(
            major_radius=0.045, minor_radius=0.012,
            major_segments=16, minor_segments=6,
            location=(0, 0, z),
            rotation=(math.pi / 2 * (i % 2), 0, 0),
        )
        link = bpy.context.object
        link.name = f"CGI_lantern_chain_{i}"
        link.data.materials.append(chain_mat)
        link.parent = body

    return body


def make_hologram_panel(plan: dict) -> bpy.types.Object:
    """Floating holographic billboard: emissive plane with scan-line strips."""
    appearance = plan["appearance"]
    color = appearance["color"]
    emission = appearance["emission_strength"]
    mat = make_material("hologram_emit", color, emission * 2.2)
    mat.blend_method = "BLEND"

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 1.3))
    panel = bpy.context.object
    panel.name = "CGI_hologram_panel"
    panel.scale = (1.4, 0.01, 0.9)
    panel.rotation_euler = (math.radians(90), 0, 0)
    panel.data.materials.append(mat)

    scan_color = [c * 0.28 for c in color[:3]]
    scan_mat = make_material("hologram_scan", scan_color, emission * 0.6)
    for z_off in (-0.28, -0.09, 0.09, 0.28):
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, -0.007, 1.3 + z_off))
        strip = bpy.context.object
        strip.name = "CGI_hologram_strip"
        strip.scale = (1.35, 0.01, 0.032)
        strip.rotation_euler = (math.radians(90), 0, 0)
        strip.data.materials.append(scan_mat)
        strip.parent = panel

    # Frame border
    frame_mat = make_material("hologram_frame", color, emission * 0.9)
    for axis, sx, sz in [("h_top", 1.5, 0.04), ("h_bot", 1.5, 0.04), ("v_l", 0.04, 1.0), ("v_r", 0.04, 1.0)]:
        ox = 0.0
        oz = {"h_top": 0.48, "h_bot": -0.48, "v_l": 0.0, "v_r": 0.0}.get(axis, 0.0)
        ox = {"v_l": -0.72, "v_r": 0.72}.get(axis, 0.0)
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(ox, -0.007, 1.3 + oz))
        edge = bpy.context.object
        edge.name = f"CGI_hologram_frame_{axis}"
        edge.scale = (sx, 0.01, sz)
        edge.rotation_euler = (math.radians(90), 0, 0)
        edge.data.materials.append(frame_mat)
        edge.parent = panel

    return panel


def make_cgi_object(plan: dict) -> bpy.types.Object:
    """Dispatch to the right geometry builder based on plan['asset']."""
    asset = plan.get("asset", "procedural_robot")
    if asset == "lantern_drone":
        return make_lantern_drone(plan)
    if asset == "hologram_panel":
        return make_hologram_panel(plan)
    return make_robot(plan)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def screen_to_world(point: list[float], scale: float) -> tuple[float, float, float]:
    x_norm, y_norm = point
    x = (x_norm - 0.5) * 5.8
    y = -1.4 + (0.75 - y_norm) * 2.1
    z = -0.35 + scale * 0.55
    return (x, y, z)


def animate_object(obj: bpy.types.Object, plan: dict) -> None:
    anim = plan["animation"]
    path = anim["screen_path"]
    scale = float(anim["scale"])
    asset = plan.get("asset", "procedural_robot")
    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end
    span = max(1, frame_end - frame_start)

    obj.scale = (scale, scale, scale)

    for idx, point in enumerate(path):
        frame = frame_start + int(round((idx / max(1, len(path) - 1)) * span))
        wx, wy, wz = screen_to_world(point, scale)

        # Lantern drone hovers — add gentle sine oscillation on Z
        if asset == "lantern_drone":
            wz += 0.12 * math.sin(idx * math.pi * 0.62)

        obj.location = (wx, wy, wz)
        obj.rotation_euler = (
            0,
            0,
            idx * math.pi * 2 * float(anim.get("rotation_turns", 1.0)) / max(1, len(path) - 1),
        )
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            for key in fc.keyframe_points:
                key.interpolation = "BEZIER"


# ---------------------------------------------------------------------------
# Lighting / compositing / render
# ---------------------------------------------------------------------------

def add_lighting_and_shadow() -> None:
    bpy.ops.mesh.primitive_plane_add(size=14, location=(0, 0, 0))
    plane = bpy.context.object
    plane.name = "shadow_catcher_floor"
    mat = bpy.data.materials.new("matte_shadow_floor")
    mat.diffuse_color = (0.55, 0.55, 0.55, 1)
    plane.data.materials.append(mat)
    plane.is_shadow_catcher = True

    bpy.ops.object.light_add(type="AREA", location=(-3.5, -4.0, 5.5))
    key = bpy.context.object
    key.name = "soft_key_light"
    key.data.energy = 550
    key.data.size = 5.0

    bpy.ops.object.light_add(type="POINT", location=(2.5, -2.0, 2.4))
    fill = bpy.context.object
    fill.name = "robot_rim_light"
    fill.data.energy = 75


def setup_compositor(video_path: str) -> None:
    scene = bpy.context.scene
    scene.render.film_transparent = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    movie = tree.nodes.new("CompositorNodeMovieClip")
    movie.clip = bpy.data.movieclips.load(video_path)
    render = tree.nodes.new("CompositorNodeRLayers")
    alpha = tree.nodes.new("CompositorNodeAlphaOver")
    comp = tree.nodes.new("CompositorNodeComposite")

    tree.links.new(movie.outputs["Image"], alpha.inputs[1])
    tree.links.new(render.outputs["Image"], alpha.inputs[2])
    tree.links.new(alpha.outputs["Image"], comp.inputs["Image"])


def render_output(output: str) -> None:
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.audio_codec = "AAC"
    scene.render.filepath = output
    bpy.ops.render.render(animation=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    plan = load_json(args.plan)
    video_path = manifest["video"]["path"]

    clear_scene()
    setup_scene(plan, args.samples, args.max_duration)
    camera = make_camera(video_path)
    apply_sfm_camera_motion(camera, plan)
    add_lighting_and_shadow()
    cgi = make_cgi_object(plan)
    animate_object(cgi, plan)
    setup_compositor(video_path)
    render_output(args.output)


if __name__ == "__main__":
    main()
