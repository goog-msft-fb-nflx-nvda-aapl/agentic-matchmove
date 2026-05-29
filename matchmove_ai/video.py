from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .schema import VideoInfo


def probe_video(video_path: str | Path) -> VideoInfo:
    path = Path(video_path)
    try:
        import cv2  # type: ignore
    except Exception:
        return _probe_with_ffprobe(path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return VideoInfo(str(path), fps, width, height, frame_count, duration)


def extract_keyframes(video_path: str | Path, out_dir: str | Path, stride_seconds: float) -> list[tuple[int, float, Path]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore
    except Exception:
        return _extract_keyframes_ffmpeg(video_path, out, stride_seconds)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    stride = max(1, int(round(fps * stride_seconds)))
    frames: list[tuple[int, float, Path]] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            frame_path = out / f"frame_{index:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frames.append((index, index / fps, frame_path))
        index += 1
    cap.release()
    return frames


def _probe_with_ffprobe(path: Path) -> VideoInfo:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    payload = json.loads(proc.stdout)
    stream = payload["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    num, den = stream.get("r_frame_rate", "30/1").split("/")
    fps = float(num) / float(den)
    duration = float(stream.get("duration") or 0.0)
    frame_count = int(float(stream.get("nb_frames") or int(round(duration * fps))))
    return VideoInfo(str(path), fps, width, height, frame_count, duration)


def _extract_keyframes_ffmpeg(video_path: str | Path, out: Path, stride_seconds: float) -> list[tuple[int, float, Path]]:
    video = probe_video(video_path)
    fps_expr = 1.0 / max(stride_seconds, 0.001)
    pattern = out / "frame_%06d.jpg"
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps={fps_expr}", str(pattern)]
    subprocess.run(cmd, check=True)
    frames: list[tuple[int, float, Path]] = []
    for idx, frame_path in enumerate(sorted(out.glob("frame_*.jpg"))):
        timestamp = idx * stride_seconds
        frame_index = int(round(timestamp * video.fps))
        frames.append((frame_index, timestamp, frame_path))
    return frames
