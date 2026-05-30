"""
Blender CGI insertion script — geometry-grounded matchmove.

Camera:
  - Intrinsics (focal length, principal point) loaded from SfM ground_plane.json
  - Extrinsics (R, t) per frame loaded from camera_poses.json
  - Fixed camera used only when SfM data is absent

Placement:
  - screen_to_world() replaced by ray_plane_intersect():
    cast a ray from the SfM camera through the screen point,
    intersect with the estimated ground plane → real world XYZ
  - Fallback to hardcoded approximation when SfM unavailable

Occlusion:
  - SAM2 foreground masks (person, tree, railing) loaded as per-frame
    image sequences in the compositor → CGI passes behind foreground

Render:
  - EEVEE Next for fast turnaround
  - Bloom glare for emissive glow
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    argv = []
    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--max-duration", type=float, default=0.0)
    return p.parse_args(argv)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# SfM data loading
# ---------------------------------------------------------------------------

def load_sfm_data(plan: dict) -> dict | None:
    """
    Load camera_poses.json and ground_plane.json from the tracking directory.
    Returns a dict with 'poses', 'intrinsics', 'ground_plane', or None.
    """
    pose_path = plan.get("tracking", {}).get("camera_poses_path", "")
    if not pose_path or not Path(pose_path).exists():
        return None
    raw = load_json(pose_path)
    poses = raw.get("poses", [])
    if not poses:
        return None

    gp_path = Path(pose_path).parent / "ground_plane.json"
    ground_plane = load_json(gp_path) if gp_path.exists() else None

    # Build frame-index → pose lookup by image name
    pose_by_frame: dict[int, dict] = {}
    for pose in poses:
        name = pose.get("name", "")
        # name format: "0042_frame_000630.jpg" → frame index 630
        try:
            frame_idx = int(name.split("_frame_")[1].split(".")[0])
        except Exception:
            continue
        pose_by_frame[frame_idx] = pose

    intrinsics = None
    if ground_plane:
        intrinsics = ground_plane.get("camera_intrinsics")

    return {
        "poses": poses,
        "pose_by_frame": pose_by_frame,
        "ground_plane": ground_plane,
        "intrinsics": intrinsics,
    }


def _nearest_pose(sfm: dict, frame_idx: int) -> dict | None:
    """Return the SfM pose closest to frame_idx."""
    pbf = sfm.get("pose_by_frame", {})
    if not pbf:
        return None
    if frame_idx in pbf:
        return pbf[frame_idx]
    closest = min(pbf.keys(), key=lambda k: abs(k - frame_idx))
    return pbf[closest]


# ---------------------------------------------------------------------------
# Ray-plane intersection (replaces screen_to_world)
# ---------------------------------------------------------------------------

def ray_plane_intersect(
    u: float, v: float,
    R: np.ndarray, t: np.ndarray,
    K: np.ndarray,
    plane_normal: np.ndarray, plane_d: float,
    W: int, H: int,
) -> np.ndarray | None:
    """
    Cast a ray from the camera through pixel (u*W, v*H) and intersect it
    with the ground plane.

    u, v: normalised screen coords [0,1]
    R, t: cam_from_world rotation and translation (3×3, 3)
    K:    3×3 camera intrinsic matrix
    plane_normal, plane_d: n·x = d in world coords

    Returns world XYZ on the ground plane, or None if ray is parallel.
    """
    px = u * W
    py = v * H
    # Ray direction in camera space (unnormalised)
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([px, py, 1.0])
    # Transform to world space
    R_wc = R.T                    # world-from-camera rotation
    ray_world = R_wc @ ray_cam    # direction in world
    cam_origin = -R_wc @ t        # camera centre in world

    denom = float(plane_normal @ ray_world)
    if abs(denom) < 1e-6:
        return None               # ray parallel to plane
    lam = (plane_d - float(plane_normal @ cam_origin)) / denom
    if lam < 0:
        lam = abs(lam)            # flip if behind camera
    return cam_origin + lam * ray_world


def screen_to_world_sfm(
    u: float, v: float, sfm: dict, frame_idx: int, scale: float
) -> tuple[float, float, float]:
    """World position from screen coords using SfM data. Falls back to hardcoded."""
    gp = sfm.get("ground_plane") if sfm else None
    intr = sfm.get("intrinsics") if sfm else None
    pose = _nearest_pose(sfm, frame_idx) if sfm else None

    if gp and intr and pose:
        R_data = pose.get("R")
        t_data = pose.get("t")
        if R_data and t_data:
            R = np.array(R_data)
            t = np.array(t_data)
            f = intr["focal_length"]
            cx, cy = intr["cx"], intr["cy"]
            W, H = intr["width"], intr["height"]
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            n = np.array(gp["normal"])
            d = float(gp["d"])
            pt = ray_plane_intersect(u, v, R, t, K, n, d, W, H)
            if pt is not None:
                return float(pt[0]), float(pt[1]), float(pt[2])

    # Hardcoded fallback (no SfM)
    x = (u - 0.5) * 5.8
    y = -1.4 + (0.75 - v) * 2.1
    z = -0.35 + scale * 0.55
    return x, y, z


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

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
    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    try:
        eevee = bpy.context.scene.eevee
        eevee.taa_render_samples = max(8, samples)
        eevee.use_shadows = True
        eevee.use_gtao = True
    except Exception:
        pass
    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.view_settings.look = "Medium High Contrast"


# ---------------------------------------------------------------------------
# Camera setup — SfM-driven
# ---------------------------------------------------------------------------

def make_camera(video_path: str, sfm: dict | None) -> bpy.types.Object:
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    camera.name = "SfM_Camera"

    # Set intrinsics from SfM if available
    intr = (sfm or {}).get("intrinsics")
    if intr:
        f = intr["focal_length"]
        W, H = intr["width"], intr["height"]
        # Blender uses sensor width + focal length in mm
        # sensor_width / focal_length = 2 * tan(hfov/2) → sensor_width = 36mm default
        sensor_w = 36.0
        focal_mm = sensor_w * f / W
        camera.data.lens = focal_mm
        camera.data.sensor_width = sensor_w
        # Shift principal point if not centred
        cx, cy = intr["cx"], intr["cy"]
        camera.data.shift_x = (cx - W / 2) / W
        camera.data.shift_y = -(cy - H / 2) / H
    else:
        camera.data.lens = 28
        camera.data.sensor_width = 36.0

    # Load background video clip
    try:
        clip = bpy.data.movieclips.load(video_path)
        camera.data.show_background_images = True
        bg = camera.data.background_images.new()
        bg.source = "MOVIE_CLIP"
        bg.clip = clip
        bg.display_depth = "BACK"
        bg.alpha = 1.0
    except Exception as e:
        print(f"Warning: could not load background clip: {e}")

    return camera


def animate_camera_sfm(camera: bpy.types.Object, sfm: dict, plan: dict) -> bool:
    """
    Keyframe camera position and rotation from SfM extrinsics.
    Returns True if any poses were applied.
    """
    pbf = sfm.get("pose_by_frame", {})
    if not pbf:
        return False

    fps = float(plan["video"].get("fps") or 30)
    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end
    applied = 0

    for frame_idx, pose in sorted(pbf.items()):
        blender_frame = frame_start + int(round(frame_idx / fps * fps))
        blender_frame = max(frame_start, min(frame_end, frame_idx + 1))

        R_data = pose.get("R")
        t_data = pose.get("t")
        if not R_data or not t_data:
            continue

        R = np.array(R_data)
        t = np.array(t_data)
        # Camera centre in world = -R^T t
        cam_pos = (-R.T @ t).tolist()
        # Rotation: convert cam_from_world to Blender world_from_cam
        # Blender camera looks along -Z, SfM camera along +Z
        R_bl = R.T                       # world_from_cam
        # Flip Y and Z axes: SfM Z forward → Blender -Z forward
        flip = np.diag([1, -1, -1])
        R_bl = R_bl @ flip
        mat = Matrix(R_bl.tolist())
        euler = mat.to_euler('XYZ')

        camera.location = Vector(cam_pos)
        camera.rotation_euler = euler
        camera.keyframe_insert(data_path="location", frame=blender_frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=blender_frame)
        applied += 1

    if applied > 0:
        if camera.animation_data and camera.animation_data.action:
            for fc in camera.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
        print(f"Camera animated with {applied} SfM poses")
        return True

    # No R/t in poses — fallback to projection_center only
    centers = [np.array(p["projection_center"])
               for p in sfm["poses"] if p.get("projection_center")]
    if len(centers) < 2:
        return False

    mean = np.mean(centers, axis=0)
    centered = [c - mean for c in centers]
    extent = max(max(abs(v)) for c in centered for v in c) or 1.0
    scale = 2.0 / extent
    target = Vector((0, 0, 0))
    frame_span = frame_end - frame_start

    for i, offset in enumerate(centered):
        frame = frame_start + int(round(i / max(1, len(centered) - 1) * frame_span))
        pos = Vector((float(offset[0]) * scale,
                      float(-offset[2]) * scale,
                      float(offset[1]) * scale))
        camera.location = pos
        direction = target - pos
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)
        applied += 1

    print(f"Camera animated with {applied} projection-centre keyframes (no R/t)")
    return applied > 0


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def make_material(name: str, color: list[float], emission: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    emit = tree.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (*color[:3], 1.0)
    emit.inputs["Strength"].default_value = max(emission, 1.5)
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_material_dark(name: str, color: list[float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color[:3], 1.0)
        bsdf.inputs["Roughness"].default_value = 0.9
    return mat


# ---------------------------------------------------------------------------
# CGI asset builders
# ---------------------------------------------------------------------------

def make_robot(obj_def: dict, suffix: str = "") -> bpy.types.Object:
    app = obj_def["appearance"]
    mat = make_material(f"robot_glow{suffix}", app["color"], app["emission_strength"])
    dark = make_material_dark(f"robot_dark{suffix}", [0.03, 0.035, 0.04])

    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.55, location=(0, 0, 0.9))
    body = bpy.context.object
    body.name = f"CGI_robot_body{suffix}"
    body.scale = (0.75, 0.55, 0.95)
    body.data.materials.append(mat)

    for x in (-0.32, 0.32):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, radius=0.09, location=(x, -0.46, 1.02))
        eye = bpy.context.object; eye.name = f"CGI_robot_eye{suffix}"
        eye.data.materials.append(dark); eye.parent = body

    for x in (-0.58, 0.58):
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.055, depth=0.8,
                                             location=(x, 0, 0.62),
                                             rotation=(0, math.radians(18 if x < 0 else -18), 0))
        arm = bpy.context.object; arm.name = f"CGI_robot_arm{suffix}"
        arm.data.materials.append(mat); arm.parent = body

    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.025, depth=0.55, location=(0.18, 0, 1.55))
    ant = bpy.context.object; ant.name = f"CGI_robot_antenna{suffix}"
    ant.data.materials.append(mat); ant.parent = body

    bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=6, radius=0.065, location=(0.18, 0, 1.85))
    tip = bpy.context.object; tip.name = f"CGI_robot_tip{suffix}"
    tip.data.materials.append(make_material(f"tip{suffix}", [1.0, 0.3, 0.1], 3.5))
    tip.parent = body
    return body


def make_lantern_drone(obj_def: dict, suffix: str = "") -> bpy.types.Object:
    app = obj_def["appearance"]
    color = app["color"]; emission = app["emission_strength"]
    mat = make_material(f"lantern_glow{suffix}", color, emission)
    ring_mat = make_material(f"lantern_ring{suffix}", [min(1.0, c * 1.6) for c in color], emission * 1.5)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=14, radius=0.42, location=(0, 0, 1.0))
    body = bpy.context.object; body.name = f"CGI_lantern_body{suffix}"; body.scale = (1.0, 1.0, 1.25)
    body.data.materials.append(mat)

    bpy.ops.mesh.primitive_torus_add(major_radius=0.50, minor_radius=0.04,
                                      major_segments=36, minor_segments=10, location=(0, 0, 1.0))
    ring = bpy.context.object; ring.name = f"CGI_lantern_ring{suffix}"
    ring.data.materials.append(ring_mat); ring.parent = body

    bpy.ops.mesh.primitive_cylinder_add(vertices=10, radius=0.065, depth=0.16, location=(0, 0, 1.56))
    cap = bpy.context.object; cap.name = f"CGI_lantern_cap{suffix}"
    cap.data.materials.append(mat); cap.parent = body
    return body


def make_kitsune_fox(obj_def: dict, suffix: str = "") -> bpy.types.Object:
    app = obj_def["appearance"]
    color = app["color"]; emission = app["emission_strength"]
    mat = make_material(f"fox_glow{suffix}", color, emission * 2.5)
    ear_mat = make_material(f"fox_ear{suffix}", [min(1.0, c * 1.4) for c in color], emission * 3.5)

    def wf(obj: bpy.types.Object, t: float = 0.025) -> None:
        mod = obj.modifiers.new("Wireframe", "WIREFRAME")
        mod.thickness = t; mod.use_even_offset = True

    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=10, radius=0.5, location=(0, 0, 0.65))
    body = bpy.context.object; body.name = f"CGI_fox_body{suffix}"; body.scale = (0.65, 0.45, 0.80)
    body.data.materials.append(mat); wf(body, 0.022)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=14, ring_count=10, radius=0.34, location=(0, -0.48, 1.22))
    head = bpy.context.object; head.name = f"CGI_fox_head{suffix}"; head.scale = (0.9, 1.05, 0.85)
    head.data.materials.append(mat); wf(head, 0.020); head.parent = body

    for ex in (-0.20, 0.20):
        bpy.ops.mesh.primitive_cone_add(vertices=6, radius1=0.10, radius2=0.01, depth=0.36,
                                         location=(ex, -0.38, 1.48))
        ear = bpy.context.object; ear.name = f"CGI_fox_ear{suffix}"
        ear.rotation_euler = (math.radians(-12), 0, math.radians(-8 if ex < 0 else 8))
        ear.data.materials.append(ear_mat); wf(ear, 0.018); ear.parent = body

    for i, (tx, ty, tz, tr) in enumerate([(0, 0.45, 0.30, 0.20), (0, 0.65, 0.55, 0.22),
                                            (0, 0.72, 0.85, 0.21), (0, 0.60, 1.10, 0.18)]):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=10, ring_count=8, radius=tr, location=(tx, ty, tz))
        seg = bpy.context.object; seg.name = f"CGI_fox_tail{i}{suffix}"
        seg.data.materials.append(mat); wf(seg, 0.018); seg.parent = body
    return body


def make_hologram_panel(obj_def: dict, suffix: str = "") -> bpy.types.Object:
    app = obj_def["appearance"]
    color = app["color"]; emission = app["emission_strength"]
    mat = make_material(f"hologram_emit{suffix}", color, emission * 3.0)

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 1.3))
    panel = bpy.context.object; panel.name = f"CGI_hologram_panel{suffix}"
    panel.scale = (1.8, 0.02, 1.1); panel.rotation_euler = (math.radians(90), 0, 0)
    panel.data.materials.append(mat)
    return panel


def import_glb(glb_path: str, obj_def: dict, suffix: str = "") -> bpy.types.Object | None:
    try:
        before = set(bpy.data.objects.keys())
        bpy.ops.import_scene.gltf(filepath=glb_path)
        new_objs = [bpy.data.objects[n] for n in bpy.data.objects.keys() if n not in before]
        if not new_objs:
            return None
        roots = [o for o in new_objs if o.parent not in new_objs]
        root = roots[0] if roots else new_objs[0]
        root.name = f"CGI_asset_{suffix}"

        all_verts = []
        for o in new_objs:
            if o.type == "MESH":
                for v in o.data.vertices:
                    all_verts.append(o.matrix_world @ v.co)
        if all_verts:
            xs = [v.x for v in all_verts]; ys = [v.y for v in all_verts]; zs = [v.z for v in all_verts]
            extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) or 1.0
            root.scale = (1.0 / extent,) * 3
            bpy.ops.object.select_all(action="DESELECT")
            for o in new_objs: o.select_set(True)
            bpy.context.view_layer.objects.active = root
            bpy.ops.object.transform_apply(scale=True)
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
            root.location = (0, 0, 0)

        color = obj_def.get("appearance", {}).get("color", [0.9, 0.8, 0.3])
        emission = obj_def.get("appearance", {}).get("emission_strength", 2.0)
        tint_mat = make_material(f"asset_tint{suffix}", color, emission)
        for o in new_objs:
            if o.type == "MESH":
                o.data.materials.clear()
                o.data.materials.append(tint_mat)
        return root
    except Exception as e:
        print(f"GLB import failed: {e}")
        return None


_GEOMETRY_REGISTRY: dict = {}


def make_cgi_object(obj_def: dict, suffix: str = "") -> bpy.types.Object:
    glb = obj_def.get("glb_path", "")
    if glb and Path(glb).exists():
        obj = import_glb(glb, obj_def, suffix)
        if obj is not None:
            return obj

    fn_name = obj_def.get("geometry_function", "")
    if fn_name and fn_name in _GEOMETRY_REGISTRY:
        return _GEOMETRY_REGISTRY[fn_name](obj_def, suffix)

    query = obj_def.get("visual_description") or f"{obj_def.get('label','')} {obj_def.get('story_role','')}"
    fn_name = _semantic_match(query)
    return _GEOMETRY_REGISTRY.get(fn_name, make_robot)(obj_def, suffix)


def _semantic_match(query: str) -> str:
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from geometry_catalog import CATALOG
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("all-MiniLM-L6-v2")
        q = model.encode(query, convert_to_tensor=True)
        best_fn, best_score = "make_robot", -1.0
        for fn_name, info in CATALOG.items():
            score = float(util.cos_sim(q, model.encode(info["description"], convert_to_tensor=True)))
            if score > best_score:
                best_score, best_fn = score, fn_name
        return best_fn
    except Exception:
        return "make_robot"


# ---------------------------------------------------------------------------
# Animation — geometry-grounded
# ---------------------------------------------------------------------------

def animate_object(
    obj: bpy.types.Object, obj_def: dict,
    frame_start: int, frame_end: int,
    sfm: dict | None, fps: float,
) -> None:
    anim = obj_def.get("animation", obj_def)
    path = anim.get("screen_path", [[0.25, 0.75], [0.5, 0.65], [0.72, 0.70]])
    scale = float(anim.get("scale", obj_def.get("scale", 0.9)))
    asset = obj_def.get("asset", "")
    rotation_turns = float(anim.get("rotation_turns", 1.0))
    span = max(1, frame_end - frame_start)

    obj.scale = (scale, scale, scale)

    for idx, point in enumerate(path):
        frame = frame_start + int(round((idx / max(1, len(path) - 1)) * span))
        u, v = point[0], point[1]

        # Choose which SfM camera pose to use for this animation keyframe
        world_frame_idx = int(round((frame - frame_start) / fps)) if fps > 0 else 0

        wx, wy, wz = screen_to_world_sfm(u, v, sfm, world_frame_idx, scale)

        if asset == "lantern_drone" or obj_def.get("placement") == "air":
            wz += 0.14 * math.sin(idx * math.pi * 0.55)

        obj.location = (wx, wy, wz)
        obj.rotation_euler = (0, 0,
                               idx * math.pi * 2 * rotation_turns / max(1, len(path) - 1))
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "BEZIER"


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

def add_lighting() -> None:
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 10))
    sun = bpy.context.object; sun.name = "sun_light"
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(45), 0, math.radians(30))


# ---------------------------------------------------------------------------
# Compositor — background video + SAM2 occlusion masks
# ---------------------------------------------------------------------------

def setup_compositor(video_path: str, mask_dir: str | None = None) -> None:
    scene = bpy.context.scene
    scene.render.film_transparent = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    movie = tree.nodes.new("CompositorNodeMovieClip")
    movie.clip = bpy.data.movieclips.load(video_path)

    render = tree.nodes.new("CompositorNodeRLayers")

    glare = tree.nodes.new("CompositorNodeGlare")
    glare.glare_type = "BLOOM"; glare.quality = "MEDIUM"
    glare.threshold = 0.5; glare.size = 7

    alpha = tree.nodes.new("CompositorNodeAlphaOver")
    comp = tree.nodes.new("CompositorNodeComposite")

    tree.links.new(render.outputs["Image"], glare.inputs["Image"])

    # SAM2 occlusion: composite foreground masks on top of CGI
    # so CGI passes BEHIND people, trees, railings
    cgi_out = glare.outputs["Image"]
    if mask_dir and Path(mask_dir).exists():
        cgi_out = _add_occlusion_masks(tree, cgi_out, render.outputs["Image"], mask_dir)

    tree.links.new(movie.outputs["Image"], alpha.inputs[1])   # background
    tree.links.new(cgi_out, alpha.inputs[2])                  # CGI (with occlusion)
    tree.links.new(alpha.outputs["Image"], comp.inputs["Image"])


def _add_occlusion_masks(
    tree: bpy.types.NodeTree,
    cgi_node_output,
    render_output,
    mask_dir: str,
) -> object:
    """
    For each foreground mask (person, tree, railing), punch a hole in the CGI
    layer so that foreground objects appear in front of the CGI.

    Approach: load each mask as an image sequence, use AlphaOver to restore
    the background video pixel where the mask is active.
    """
    mask_path = Path(mask_dir)
    # Group masks by label (use first mask file per label as sequence base)
    label_bases: dict[str, Path] = {}
    foreground_labels = {"person", "railing", "tree", "fence", "arch", "pole"}

    for p in sorted(mask_path.glob("frame_000_*.png")):
        label = p.stem.split("_", 2)[-1].replace("_", " ").lower()
        if any(fl in label for fl in foreground_labels):
            seq_name = p.stem[:9]  # "frame_000"
            if seq_name not in label_bases:
                label_bases[seq_name] = p

    if not label_bases:
        return cgi_node_output

    # Load background movie for restoration
    bg_restore = tree.nodes.new("CompositorNodeMovieClip")
    bg_restore.clip = list(bpy.data.movieclips)[0]

    current_out = cgi_node_output
    for seq_name, mask_base in list(label_bases.items())[:4]:  # max 4 masks
        # Load mask as image sequence
        try:
            img = bpy.data.images.load(str(mask_base))
            img.source = "SEQUENCE"
            img_node = tree.nodes.new("CompositorNodeImage")
            img_node.image = img
            img_node.frame_duration = bpy.context.scene.frame_end
            img_node.use_auto_refresh = True

            # Use mask to restore background behind CGI
            mix = tree.nodes.new("CompositorNodeAlphaOver")
            tree.links.new(current_out, mix.inputs[1])       # CGI (below)
            tree.links.new(bg_restore.outputs["Image"], mix.inputs[2])  # BG (above)
            tree.links.new(img_node.outputs["Image"], mix.inputs["Fac"])
            current_out = mix.outputs["Image"]
        except Exception as e:
            print(f"Mask load failed ({mask_base}): {e}")

    return current_out


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

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

    sfm = load_sfm_data(plan)
    if sfm:
        print(f"SfM loaded: {len(sfm['pose_by_frame'])} poses, "
              f"ground_plane={'yes' if sfm.get('ground_plane') else 'no'}")
    else:
        print("No SfM data — using hardcoded camera")

    fps = float(plan["video"].get("fps") or 30)

    clear_scene()
    setup_scene(plan, args.samples, args.max_duration)
    camera = make_camera(video_path, sfm)

    if sfm:
        animate_camera_sfm(camera, sfm, plan)

    add_lighting()

    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end

    objects_list = plan.get("objects")
    if objects_list:
        for i, obj_def in enumerate(objects_list):
            cgi = make_cgi_object(obj_def, suffix=f"_{i}")
            animate_object(cgi, obj_def, frame_start, frame_end, sfm, fps)
    else:
        cgi = make_cgi_object(plan)
        animate_object(cgi, plan, frame_start, frame_end, sfm, fps)

    mask_dir = str(Path(args.manifest).parent / "sam2_masks")
    setup_compositor(video_path, mask_dir if Path(mask_dir).exists() else None)
    render_output(args.output)


# Register geometry builders
_GEOMETRY_REGISTRY.update({
    "make_kitsune_fox": make_kitsune_fox,
    "make_lantern_drone": make_lantern_drone,
    "make_robot": make_robot,
    "make_hologram_panel": make_hologram_panel,
})

if __name__ == "__main__":
    main()
