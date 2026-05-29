from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    argv = []
    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def setup_scene(plan: dict) -> None:
    video = plan["video"]
    fps = float(video.get("fps") or 30)
    duration = float(plan["animation"]["duration_seconds"])
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = max(2, int(round(duration * fps)))
    bpy.context.scene.render.fps = int(round(fps))
    bpy.context.scene.render.resolution_x = int(video.get("width") or 1920)
    bpy.context.scene.render.resolution_y = int(video.get("height") or 1080)
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.view_settings.look = "Medium High Contrast"


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


def make_material(name: str, color: list[float], emission_strength: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
        bsdf.inputs["Emission Color"].default_value = (color[0], color[1], color[2], 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission_strength
        bsdf.inputs["Roughness"].default_value = 0.42
    return mat


def make_robot(plan: dict) -> bpy.types.Object:
    appearance = plan["appearance"]
    mat = make_material("robot_blue_glow", appearance["color"], appearance["emission_strength"])

    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.55, location=(0, 0, 0.9))
    body = bpy.context.object
    body.name = "CGI_robot_body"
    body.scale = (0.75, 0.55, 0.95)
    body.data.materials.append(mat)

    dark = make_material("robot_dark_joints", [0.03, 0.035, 0.04], 0.0)
    for x in (-0.32, 0.32):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.09, location=(x, -0.46, 1.02))
        eye = bpy.context.object
        eye.name = "CGI_robot_eye"
        eye.data.materials.append(dark)
        eye.parent = body

    for x in (-0.58, 0.58):
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.055, depth=0.8, location=(x, 0, 0.62), rotation=(0, math.radians(18 if x < 0 else -18), 0))
        arm = bpy.context.object
        arm.name = "CGI_robot_arm"
        arm.data.materials.append(mat)
        arm.parent = body

    return body


def screen_to_world(point: list[float], scale: float) -> tuple[float, float, float]:
    x_norm, y_norm = point
    x = (x_norm - 0.5) * 5.8
    y = -1.4 + (0.75 - y_norm) * 2.1
    z = 0.45 + scale * 0.55
    return (x, y, z)


def animate_object(obj: bpy.types.Object, plan: dict) -> None:
    anim = plan["animation"]
    path = anim["screen_path"]
    scale = float(anim["scale"])
    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end
    span = max(1, frame_end - frame_start)

    obj.scale = (scale, scale, scale)
    for idx, point in enumerate(path):
        frame = frame_start + int(round((idx / max(1, len(path) - 1)) * span))
        obj.location = screen_to_world(point, scale)
        obj.rotation_euler = (0, 0, idx * math.pi * 2 * float(anim.get("rotation_turns", 1.0)) / max(1, len(path) - 1))
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            for key in fc.keyframe_points:
                key.interpolation = "BEZIER"


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


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    plan = load_json(args.plan)
    video_path = manifest["video"]["path"]

    clear_scene()
    setup_scene(plan)
    make_camera(video_path)
    add_lighting_and_shadow()
    robot = make_robot(plan)
    animate_object(robot, plan)
    setup_compositor(video_path)
    render_output(args.output)


if __name__ == "__main__":
    main()
