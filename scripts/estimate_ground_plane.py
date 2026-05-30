#!/usr/bin/env python3
"""
Estimate ground plane from SfM point cloud.

Strategy: project all 3D points through every registered camera, find which
points land inside SAM2 ground/walkway mask regions, then fit a plane to
those ground-labelled 3D points using RANSAC.

Output: work/tracking/ground_plane.json
  {
    "normal": [nx, ny, nz],   # plane unit normal (pointing toward cameras)
    "point":  [px, py, pz],   # a point on the plane
    "d": float,               # signed distance: normal . X = d for X on plane
    "inlier_count": int,
    "method": "ransac_ground_masked | ransac_all | fallback_y"
  }
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pycolmap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", default="work")
    p.add_argument("--sparse-model", default="")
    p.add_argument("--ransac-iters", type=int, default=2000)
    p.add_argument("--inlier-thresh", type=float, default=0.08,
                   help="Distance threshold (world units) for RANSAC inliers")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Ground mask lookup
# ---------------------------------------------------------------------------

GROUND_LABELS = {"ground", "walkway", "ground_walkway", "ground walkway",
                 "floor", "road", "pavement", "sidewalk"}

def _load_ground_masks(mask_dir: Path) -> list[np.ndarray]:
    """Load all SAM2 mask images whose filename contains a ground label."""
    masks = []
    for p in sorted(mask_dir.glob("*.png")):
        label = p.stem.split("_", 2)[-1].replace("_", " ").lower()
        if any(g in label for g in GROUND_LABELS):
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                masks.append(m > 128)
    return masks


def _point_in_ground(u: float, v: float, W: int, H: int,
                     masks: list[np.ndarray]) -> bool:
    """Check whether screen coordinate (u,v) in [0,1]² hits any ground mask."""
    px, py = int(u * W), int(v * H)
    px = max(0, min(W - 1, px))
    py = max(0, min(H - 1, py))
    return any(m[py, px] for m in masks)


# ---------------------------------------------------------------------------
# RANSAC plane fit
# ---------------------------------------------------------------------------

def _fit_plane_svd(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit plane to Nx3 points via SVD. Returns (normal, d) where n·x = d."""
    centroid = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - centroid)
    normal = Vt[-1]           # smallest singular vector = normal
    d = float(normal @ centroid)
    return normal, d


def ransac_plane(pts: np.ndarray, n_iters: int, thresh: float
                 ) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC plane fitting. Returns (normal, d, inlier_mask)."""
    best_mask = np.zeros(len(pts), dtype=bool)
    rng = random.Random(42)
    for _ in range(n_iters):
        idx = rng.sample(range(len(pts)), 3)
        sample = pts[idx]
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        n = np.cross(v1, v2)
        if np.linalg.norm(n) < 1e-8:
            continue
        n = n / np.linalg.norm(n)
        d = float(n @ sample[0])
        dists = np.abs(pts @ n - d)
        mask = dists < thresh
        if mask.sum() > best_mask.sum():
            best_mask = mask
    # Refit on inliers
    if best_mask.sum() >= 3:
        normal, d = _fit_plane_svd(pts[best_mask])
    else:
        normal, d = _fit_plane_svd(pts)
        best_mask = np.ones(len(pts), dtype=bool)
    return normal, d, best_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    work = Path(args.workdir)
    sparse_dir = Path(args.sparse_model) if args.sparse_model else work / "tracking" / "sparse" / "best"
    out_path = work / "tracking" / "ground_plane.json"

    model = pycolmap.Reconstruction(str(sparse_dir))
    cam = list(model.cameras.values())[0]
    W, H = cam.width, cam.height
    f = cam.focal_length
    cx, cy = cam.principal_point_x, cam.principal_point_y
    # K matrix
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    all_pts = np.array([pt.xyz for pt in model.points3D.values()])
    print(f"Total 3D points: {len(all_pts)}")

    # Try to label points as ground via mask projection
    mask_dir = work / "sam2_masks"
    ground_masks = _load_ground_masks(mask_dir) if mask_dir.exists() else []
    print(f"Ground masks loaded: {len(ground_masks)}")

    ground_pts = []
    method = "ransac_all"

    if ground_masks:
        imgs = sorted(model.images.values(), key=lambda x: x.name)
        for img in imgs:
            pose = img.cam_from_world()
            R = pose.rotation.matrix()
            t = pose.translation
            # Project all 3D points through this camera
            pts_cam = (R @ all_pts.T).T + t  # Nx3
            valid = pts_cam[:, 2] > 0
            pts_img = (K @ pts_cam[valid].T).T
            pts_img /= pts_img[:, 2:3]
            u = pts_img[:, 0] / W
            v = pts_img[:, 1] / H
            in_frame = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
            for i, (ui, vi) in enumerate(zip(u[in_frame], v[in_frame])):
                if _point_in_ground(ui, vi, W, H, ground_masks):
                    orig_idx = np.where(valid)[0][np.where(in_frame)[0][i]]
                    ground_pts.append(all_pts[orig_idx])

        ground_pts = np.unique(np.array(ground_pts), axis=0) if ground_pts else np.array([])
        print(f"Ground-masked points: {len(ground_pts)}")

        if len(ground_pts) >= 10:
            method = "ransac_ground_masked"
        else:
            ground_pts = all_pts

    pts_for_fit = ground_pts if len(ground_pts) >= 10 else all_pts

    print(f"Fitting plane via RANSAC on {len(pts_for_fit)} points...")
    normal, d, inliers = ransac_plane(pts_for_fit, args.ransac_iters, args.inlier_thresh)

    # Orient normal toward camera cluster (cameras should be above the ground)
    cam_centers = np.array([-model.images[img_id].cam_from_world().rotation.matrix().T @
                             model.images[img_id].cam_from_world().translation
                             for img_id in model.images])
    cam_mean = cam_centers.mean(0)
    if float(normal @ (cam_mean - (normal * d))) < 0:
        normal = -normal
        d = -d

    pt_on_plane = normal * d
    print(f"Ground plane: normal={np.round(normal,4).tolist()}, d={d:.4f}")
    print(f"Inliers: {inliers.sum()} / {len(pts_for_fit)}")

    result = {
        "normal": normal.tolist(),
        "point": pt_on_plane.tolist(),
        "d": float(d),
        "inlier_count": int(inliers.sum()),
        "total_points": len(pts_for_fit),
        "method": method,
        "camera_intrinsics": {
            "focal_length": float(f),
            "cx": float(cx), "cy": float(cy),
            "width": int(W), "height": int(H),
        },
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
