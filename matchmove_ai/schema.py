from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float


@dataclass
class Instance:
    track_id: str
    label: str
    confidence: float
    bbox_xyxy: list[float]
    mask_path: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameContext:
    frame_index: int
    timestamp: float
    image_path: str
    caption: str = ""
    instances: list[Instance] = field(default_factory=list)


@dataclass
class PerceptionContext:
    video: VideoInfo
    frames: list[FrameContext]
    notes: list[str] = field(default_factory=list)


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value

