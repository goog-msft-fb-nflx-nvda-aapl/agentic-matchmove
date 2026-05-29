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
    parser.add_argument("--samples", type=int, default=16)
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

    # EEVEE Next — vastly faster than Cycles for emissive/glowing objects
    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    try:
        eevee = bpy.context.scene.eevee
        eevee.taa_render_samples = max(8, samples)
        eevee.use_shadows = True
        eevee.use_gtao = True
        eevee.gtao_distance = 0.4
    except Exception:
        pass

    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.view_settings.look = "Medium High Contrast"


def make_camera(video_path: str) -> bpy.types.Object:
    bpy.ops.object.camera_add(location=(0, -6, 2.4), rotation=(math.radians(68), 0, 0))
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    camera.data.lens = 28

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
    centers = [Vector(p["projection_center"]) for p in poses if p.get("projection_center")]
    if len(centers) < 2:
        return
    mean = sum(centers, Vector((0, 0, 0))) / len(centers)
    centered = [c - mean for c in centers]
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
            for kp in fc.keyframe_points:
                kp.interpolation = "BEZIER"


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
        bsdf.inputs["Roughness"].default_value = 0.38
    return mat


# ---------------------------------------------------------------------------
# CGI asset builders
# ---------------------------------------------------------------------------

def make_robot(obj_def: dict, suffix: str = "") -> bpy.types.Object:
    app = obj_def["appearance"]
    mat = make_material(f"robot_glow{suffix}", app["color"], app["emission_strength"])
    dark = make_material(f"robot_dark{suffix}", [0.03, 0.035, 0.04], 0.0)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.55, location=(0, 0, 0.9))
    body = bpy.context.object
    body.name = f"CGI_robot_body{suffix}"
    body.scale = (0.75, 0.55, 0.95)
    body.data.materials.append(mat)

    for x in (-0.32, 0.32):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, radius=0.09,
                                              location=(x, -0.46, 1.02))
        eye = bpy.context.object
        eye.name = f"CGI_robot_eye{suffix}"
        eye.data.materials.append(dark)
        eye.parent = body

    for x in (-0.58, 0.58):
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.055, depth=0.8,
                                             location=(x, 0, 0.62),
                                             rotation=(0, math.radians(18 if x < 0 else -18), 0))
        arm = bpy.context.object
        arm.name = f"CGI_robot_arm{suffix}"
        arm.data.materials.append(mat)
        arm.parent = body

    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.025, depth=0.55,
                                         location=(0.18, 0, 1.55))
    antenna = bpy.context.object
    antenna.name = f"CGI_robot_antenna{suffix}"
    antenna.data.materials.append(mat)
    antenna.parent = body

    bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=6, radius=0.065,
                                          location=(0.18, 0, 1.85))
    tip = bpy.context.object
    tip.name = f"CGI_robot_tip{suffix}"
    tip.data.materials.append(make_material(f"tip{suffix}", [1.0, 0.3, 0.1], 3.5))
    tip.parent = body

    return body


def make_lantern_drone(obj_def: dict, suffix: str = "") -> bpy.types.Object:
    app = obj_def["appearance"]
    color = app["color"]
    emission = app["emission_strength"]
    mat = make_material(f"lantern_glow{suffix}", color, emission)
    ring_color = [min(1.0, c * 1.6) for c in color[:3]]
    ring_mat = make_material(f"lantern_ring{suffix}", ring_color, emission * 1.5)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=14, radius=0.42, location=(0, 0, 1.0))
    body = bpy.context.object
    body.name = f"CGI_lantern_body{suffix}"
    body.scale = (1.0, 1.0, 1.25)
    body.data.materials.append(mat)

    bpy.ops.mesh.primitive_torus_add(major_radius=0.50, minor_radius=0.04,
                                      major_segments=36, minor_segments=10,
                                      location=(0, 0, 1.0))
    ring = bpy.context.object
    ring.name = f"CGI_lantern_ring{suffix}"
    ring.data.materials.append(ring_mat)
    ring.parent = body

    bpy.ops.mesh.primitive_cylinder_add(vertices=10, radius=0.065, depth=0.16,
                                         location=(0, 0, 1.56))
    cap = bpy.context.object
    cap.name = f"CGI_lantern_cap{suffix}"
    cap.data.materials.append(mat)
    cap.parent = body

    chain_mat = make_material(f"lantern_chain{suffix}", [0.12, 0.10, 0.08], 0.0)
    for i, z in enumerate([0.52, 0.42, 0.34]):
        bpy.ops.mesh.primitive_torus_add(major_radius=0.045, minor_radius=0.012,
                                          major_segments=12, minor_segments=6,
                                          location=(0, 0, z),
                                          rotation=(math.pi / 2 * (i % 2), 0, 0))
        link = bpy.context.object
        link.name = f"CGI_lantern_chain{i}{suffix}"
        link.data.materials.append(chain_mat)
        link.parent = body

    return body


def make_hologram_panel(obj_def: dict, suffix: str = "") -> bpy.types.Object:
    app = obj_def["appearance"]
    color = app["color"]
    emission = app["emission_strength"]
    mat = make_material(f"hologram_emit{suffix}", color, emission * 2.2)

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 1.3))
    panel = bpy.context.object
    panel.name = f"CGI_hologram_panel{suffix}"
    panel.scale = (1.4, 0.01, 0.9)
    panel.rotation_euler = (math.radians(90), 0, 0)
    panel.data.materials.append(mat)

    scan_mat = make_material(f"hologram_scan{suffix}", [c * 0.25 for c in color[:3]], emission * 0.5)
    for z_off in (-0.28, -0.09, 0.09, 0.28):
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, -0.007, 1.3 + z_off))
        strip = bpy.context.object
        strip.name = f"CGI_scan_strip{suffix}"
        strip.scale = (1.35, 0.01, 0.032)
        strip.rotation_euler = (math.radians(90), 0, 0)
        strip.data.materials.append(scan_mat)
        strip.parent = panel

    return panel


def make_cgi_object(obj_def: dict, suffix: str = "") -> bpy.types.Object:
    asset = obj_def.get("asset", "procedural_robot")
    if asset == "lantern_drone":
        return make_lantern_drone(obj_def, suffix)
    if asset == "hologram_panel":
        return make_hologram_panel(obj_def, suffix)
    return make_robot(obj_def, suffix)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def screen_to_world(point: list[float], scale: float) -> tuple[float, float, float]:
    x_norm, y_norm = point
    x = (x_norm - 0.5) * 5.8
    y = -1.4 + (0.75 - y_norm) * 2.1
    z = -0.35 + scale * 0.55
    return (x, y, z)


def animate_object(obj: bpy.types.Object, obj_def: dict, frame_start: int, frame_end: int) -> None:
    anim = obj_def.get("animation", obj_def)  # support both nested and flat
    path = anim.get("screen_path", obj_def.get("screen_path", [[0.25, 0.75], [0.5, 0.65], [0.72, 0.70]]))
    scale = float(anim.get("scale", obj_def.get("scale", 0.4)))
    asset = obj_def.get("asset", "procedural_robot")
    rotation_turns = float(anim.get("rotation_turns", 1.0))
    span = max(1, frame_end - frame_start)

    obj.scale = (scale, scale, scale)
    for idx, point in enumerate(path):
        frame = frame_start + int(round((idx / max(1, len(path) - 1)) * span))
        wx, wy, wz = screen_to_world(point, scale)
        if asset == "lantern_drone":
            wz += 0.14 * math.sin(idx * math.pi * 0.55)
        obj.location = (wx, wy, wz)
        obj.rotation_euler = (
            0, 0,
            idx * math.pi * 2 * rotation_turns / max(1, len(path) - 1),
        )
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "BEZIER"


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
    key.data.energy = 600
    key.data.size = 5.0

    bpy.ops.object.light_add(type="POINT", location=(2.5, -2.0, 2.4))
    fill = bpy.context.object
    fill.name = "fill_light"
    fill.data.energy = 90


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

    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end

    objects_list = plan.get("objects")
    if objects_list:
        # Multi-object story mode
        for i, obj_def in enumerate(objects_list):
            cgi = make_cgi_object(obj_def, suffix=f"_{i}")
            animate_object(cgi, obj_def, frame_start, frame_end)
    else:
        # Legacy single-object mode
        cgi = make_cgi_object(plan)
        animate_object(cgi, plan, frame_start, frame_end)

    setup_compositor(video_path)
    render_output(args.output)


if __name__ == "__main__":
    main()
