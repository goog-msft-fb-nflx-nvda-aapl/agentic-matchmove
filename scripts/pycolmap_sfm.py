#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pycolmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a conservative pycolmap SfM pass on extracted keyframes")
    parser.add_argument("--workdir", default="work")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--video", default="")
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential")
    return parser.parse_args()


def _device(name: str):
    return {"cpu": pycolmap.Device.cpu, "cuda": pycolmap.Device.cuda, "auto": pycolmap.Device.auto}[name]


def _prepare_images_from_keyframes(frame_dir: Path, out_dir: Path, max_frames: int, max_side: int) -> list[dict]:
    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"No keyframes under {frame_dir}")
    stride = max(1, len(frames) // max_frames)
    selected = frames[::stride][:max_frames]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for idx, src in enumerate(selected):
        img = cv2.imread(str(src))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        dst = out_dir / f"{idx:04d}_{src.name}"
        cv2.imwrite(str(dst), img)
        frame_index = int(src.stem.replace("frame_", ""))
        manifest.append(
            {
                "sfm_image_name": dst.name,
                "source_frame": str(src),
                "frame_index": frame_index,
                "scale": scale,
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
            }
        )
    return manifest


def _prepare_images_from_video(video_path: Path, out_dir: Path, max_frames: int, max_side: int, sample_fps: float) -> list[dict]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    stride = max(1, int(round(fps / max(0.1, sample_fps))))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    candidate_indices = list(range(0, total, stride))
    if len(candidate_indices) > max_frames:
        pick = np.linspace(0, len(candidate_indices) - 1, max_frames).round().astype(int)
        candidate_indices = [candidate_indices[int(i)] for i in pick]
    candidate_set = set(candidate_indices)

    manifest = []
    frame_index = 0
    out_index = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if frame_index not in candidate_set:
            frame_index += 1
            continue
        h, w = img.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        dst = out_dir / f"{out_index:04d}_frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(dst), img)
        manifest.append(
            {
                "sfm_image_name": dst.name,
                "source_video": str(video_path),
                "frame_index": frame_index,
                "timestamp": frame_index / fps,
                "scale": scale,
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
            }
        )
        out_index += 1
        frame_index += 1
    cap.release()
    return manifest


def _pose_to_dict(image) -> dict:
    pose = image.cam_from_world
    num_points2d = image.num_points2D() if callable(image.num_points2D) else image.num_points2D
    num_points3d = image.num_points3D() if callable(image.num_points3D) else image.num_points3D
    out = {
        "image_id": int(image.image_id),
        "name": image.name,
        "projection_center": [float(v) for v in image.projection_center()],
        "num_points2D": int(num_points2d),
        "num_points3D": int(num_points3d),
    }
    if hasattr(pose, "matrix"):
        mat = pose.matrix()
        out["cam_from_world_3x4"] = np.asarray(mat).astype(float).tolist()
    else:
        out["cam_from_world"] = image.todict().get("cam_from_world")
    return out


def main() -> int:
    args = parse_args()
    work = Path(args.workdir)
    tracking = work / "tracking"
    images = tracking / "images"
    database = tracking / "database.db"
    sparse = tracking / "sparse"

    tracking.mkdir(parents=True, exist_ok=True)
    if database.exists():
        database.unlink()
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True, exist_ok=True)

    if args.video:
        manifest = _prepare_images_from_video(Path(args.video), images, args.max_frames, args.max_side, args.sample_fps)
    else:
        manifest = _prepare_images_from_keyframes(work / "frames", images, args.max_frames, args.max_side)
    (tracking / "image_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    reader_options = pycolmap.ImageReaderOptions(camera_model="SIMPLE_RADIAL")
    extraction_options = pycolmap.FeatureExtractionOptions(
        max_image_size=args.max_side,
        num_threads=args.threads,
        use_gpu=args.device != "cpu",
    )
    matching_options = pycolmap.FeatureMatchingOptions(
        num_threads=args.threads,
        use_gpu=args.device != "cpu",
    )
    sequential_options = pycolmap.SequentialPairingOptions(num_threads=args.threads, overlap=12)
    mapping_options = pycolmap.IncrementalPipelineOptions()
    mapping_options.num_threads = args.threads
    mapping_options.multiple_models = True
    mapping_options.min_model_size = max(3, min(10, len(manifest) // 4))
    mapping_options.ba_use_gpu = args.device != "cpu"
    mapping_options.mapper.abs_pose_min_num_inliers = 20
    mapping_options.mapper.init_min_num_inliers = 50

    pycolmap.extract_features(
        database,
        images,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=_device(args.device),
    )
    if args.matcher == "sequential":
        pycolmap.match_sequential(
            database,
            matching_options=matching_options,
            pairing_options=sequential_options,
            device=_device(args.device),
        )
    else:
        pycolmap.match_exhaustive(database, matching_options=matching_options, device=_device(args.device))
    reconstructions = pycolmap.incremental_mapping(database, images, sparse, options=mapping_options)

    if not reconstructions:
        report = {
            "status": "failed",
            "reason": "pycolmap returned no reconstructions",
            "image_count": len(manifest),
        }
        (tracking / "sfm_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(tracking / "sfm_report.json")
        return 2

    best_id, best = max(reconstructions.items(), key=lambda item: item[1].num_reg_images())
    best_out = sparse / "best"
    best_out.mkdir(parents=True, exist_ok=True)
    best.write(best_out)

    poses = []
    for image_id in sorted(best.reg_image_ids()):
        image = best.image(image_id)
        poses.append(_pose_to_dict(image))

    report = {
        "status": "ok",
        "source": "pycolmap",
        "best_model_id": int(best_id),
        "image_count": len(manifest),
        "registered_images": int(best.num_reg_images()),
        "points3D": int(best.num_points3D()),
        "mean_reprojection_error": float(best.compute_mean_reprojection_error()),
        "camera_poses_path": str(tracking / "camera_poses.json"),
        "sparse_model_path": str(best_out),
        "summary": best.summary(),
    }
    (tracking / "camera_poses.json").write_text(json.dumps({"poses": poses}, indent=2) + "\n", encoding="utf-8")
    (tracking / "sfm_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(tracking / "sfm_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
