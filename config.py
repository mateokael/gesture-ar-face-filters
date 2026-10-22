"""Configuration dataclasses and enums for the gesture-controlled AR filter app.

All tunables live here. Thresholds are scale-relative (fractions of hand span
or interocular distance) so distance from the camera does not change behavior.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class Gesture(Enum):
    """Recognized hand gestures."""

    NONE = "none"
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"
    ROCK = "rock"
    THREE_FINGERS = "three_fingers"
    OPEN_PALM = "open_palm"


class FilterType(Enum):
    """Available AR filter overlays."""

    NONE = "none"
    WEB_SLINGER = "web_slinger"
    SUNGLASSES = "sunglasses"
    ARMOR_VISOR = "armor_visor"
    PATRIOTIC_MASK = "patriotic_mask"


@dataclass
class GestureConfig:
    """Thresholds for gesture detection and hysteresis.

    All extension thresholds are ratios of fingertip-to-wrist distance divided
    by hand_span (wrist to middle MCP), making them scale-invariant.
    """

    finger_extended_threshold: float = 0.6
    """Finger is considered extended when tip-to-MCP / hand_span exceeds this."""

    finger_curled_threshold: float = 0.3
    """Finger is considered curled when tip-to-MCP / hand_span falls below this."""

    thumb_extended_threshold: float = 0.45
    """Thumb is considered extended when its extension ratio exceeds this."""

    thumb_curled_threshold: float = 0.30
    """Thumb is considered curled when its extension ratio falls below this."""

    enter_frames: int = 5
    """Consecutive frames a gesture must be detected before it activates."""

    release_frames: int = 8
    """Consecutive frames with no gesture before the candidate clears."""

    gesture_filter_map: Dict[Gesture, FilterType] = field(default_factory=lambda: {
        Gesture.NONE: FilterType.NONE,
        Gesture.PEACE: FilterType.WEB_SLINGER,
        Gesture.THUMBS_UP: FilterType.SUNGLASSES,
        Gesture.ROCK: FilterType.ARMOR_VISOR,
        Gesture.THREE_FINGERS: FilterType.PATRIOTIC_MASK,
        Gesture.OPEN_PALM: FilterType.NONE,
    })
    """Maps each recognized gesture to the filter it activates."""


@dataclass
class FilterPlacementConfig:
    """Positioning ratios for a single filter type.

    All values are multiples of the interocular distance, used as the
    scale reference from the face mesh.
    """

    width_ratio: float = 1.0
    height_ratio: float = 1.0
    y_offset_ratio: float = 0.0


@dataclass
class FilterConfig:
    """Positioning parameters for all filter types.

    Each filter's size and vertical offset are expressed as multiples of the
    interocular distance so they scale naturally with face size in frame.
    """

    web_slinger: FilterPlacementConfig = field(
        default_factory=lambda: FilterPlacementConfig(
            width_ratio=3.2,
            height_ratio=3.5,
            y_offset_ratio=0.3,
        )
    )

    sunglasses: FilterPlacementConfig = field(
        default_factory=lambda: FilterPlacementConfig(
            width_ratio=2.8,
            height_ratio=0.9,
            y_offset_ratio=0.0,
        )
    )

    armor_visor: FilterPlacementConfig = field(
        default_factory=lambda: FilterPlacementConfig(
            width_ratio=3.0,
            height_ratio=1.8,
            y_offset_ratio=0.3,
        )
    )

    patriotic_mask: FilterPlacementConfig = field(
        default_factory=lambda: FilterPlacementConfig(
            width_ratio=3.2,
            height_ratio=2.5,
            y_offset_ratio=-0.3,
        )
    )


@dataclass
class AppConfig:
    """Top-level application configuration."""

    gesture: GestureConfig = field(default_factory=GestureConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)

    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480

    assets_dir: str = "assets"

    flask_host: str = "127.0.0.1"
    flask_port: int = 5000
