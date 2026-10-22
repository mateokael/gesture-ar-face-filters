"""Pure gesture-detection and filter-placement logic.

No camera, no display, no Flask. All functions operate on normalized landmark
coordinates (0..1) and return state dicts or dataclasses. This module is fully
testable with synthetic fixtures.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from .config import (
        AppConfig,
        FilterConfig,
        FilterType,
        Gesture,
        GestureConfig,
    )
except ImportError:
    from config import (
        AppConfig,
        FilterConfig,
        FilterType,
        Gesture,
        GestureConfig,
    )

# ---------------------------------------------------------------------------
# MediaPipe hand landmark indices
# ---------------------------------------------------------------------------

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Face mesh indices used for filter placement
FACE_LEFT_EYE = 33
FACE_RIGHT_EYE = 263
FACE_BETWEEN_EYES = 168
FACE_FOREHEAD = 10
FACE_NOSE_TIP = 1

# Type alias: a landmark is (x, y, z) normalized to 0..1
Landmark = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _distance(p1: Landmark, p2: Landmark) -> float:
    """3D Euclidean distance between two landmarks."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _hand_span(landmarks: List[Landmark]) -> float:
    """Distance from wrist to middle finger MCP, floored at 1e-6.

    This serves as the scale reference for all finger extension ratios,
    making gesture detection invariant to distance from camera.
    """
    span = _distance(landmarks[WRIST], landmarks[MIDDLE_MCP])
    return max(span, 1e-6)


def _finger_extension(
    landmarks: List[Landmark],
    tip_idx: int,
    mcp_idx: int,
    hand_span: float,
) -> float:
    """Ratio of fingertip-to-MCP distance over hand_span.

    Using tip-to-MCP (not tip-to-wrist) gives much better separation
    between curled and extended states, especially for ring and pinky
    fingers where even a curled finger is still somewhat far from the wrist.

    Args:
        landmarks: 21 hand landmarks in normalized coords.
        tip_idx: Index of the fingertip landmark.
        mcp_idx: Index of the finger's MCP joint.
        hand_span: Pre-computed wrist-to-middle-MCP distance.

    Returns:
        Extension ratio (typically 0.1–0.8).
    """
    tip_to_mcp = _distance(landmarks[tip_idx], landmarks[mcp_idx])
    return tip_to_mcp / hand_span


def _thumb_extension(landmarks: List[Landmark], hand_span: float) -> float:
    """Thumb extension ratio: thumb-tip-to-wrist distance over hand_span.

    The thumb's geometry differs from other fingers (CMC rotation), but
    the tip-to-wrist heuristic works well enough for open/closed detection.
    """
    tip_to_wrist = _distance(landmarks[THUMB_TIP], landmarks[WRIST])
    return tip_to_wrist / hand_span


# ---------------------------------------------------------------------------
# Gesture detection
# ---------------------------------------------------------------------------


def detect_gesture(landmarks: List[Landmark], config: GestureConfig) -> Gesture:
    """Classify a hand gesture from 21 normalized landmarks.

    Uses scale-relative thresholds (fractions of hand_span) so detection
    is invariant to camera distance.

    Args:
        landmarks: Exactly 21 (x, y, z) tuples in MediaPipe hand order.
        config: Gesture detection thresholds.

    Returns:
        The detected Gesture enum value.
    """
    if len(landmarks) < 21:
        return Gesture.NONE

    span = _hand_span(landmarks)

    # Compute extension ratios for each finger
    index_ext = _finger_extension(landmarks, INDEX_TIP, INDEX_MCP, span)
    middle_ext = _finger_extension(landmarks, MIDDLE_TIP, MIDDLE_MCP, span)
    ring_ext = _finger_extension(landmarks, RING_TIP, RING_MCP, span)
    pinky_ext = _finger_extension(landmarks, PINKY_TIP, PINKY_MCP, span)
    thumb_ext = _thumb_extension(landmarks, span)

    # Classification helpers
    def finger_extended(ext: float) -> bool:
        return ext > config.finger_extended_threshold

    def finger_curled(ext: float) -> bool:
        return ext < config.finger_curled_threshold

    def thumb_extended() -> bool:
        return thumb_ext > config.thumb_extended_threshold

    def thumb_curled() -> bool:
        return thumb_ext < config.thumb_curled_threshold

    # Peace: index + middle extended, ring + pinky curled
    if (
        finger_extended(index_ext)
        and finger_extended(middle_ext)
        and finger_curled(ring_ext)
        and finger_curled(pinky_ext)
    ):
        return Gesture.PEACE

    # Thumbs up: thumb extended, all four fingers curled
    if (
        thumb_extended()
        and finger_curled(index_ext)
        and finger_curled(middle_ext)
        and finger_curled(ring_ext)
        and finger_curled(pinky_ext)
    ):
        return Gesture.THUMBS_UP

    # Rock/horns: index + pinky extended, middle + ring curled
    if (
        finger_extended(index_ext)
        and finger_curled(middle_ext)
        and finger_curled(ring_ext)
        and finger_extended(pinky_ext)
    ):
        return Gesture.ROCK

    # Three fingers: index + middle + ring extended, pinky curled
    if (
        finger_extended(index_ext)
        and finger_extended(middle_ext)
        and finger_extended(ring_ext)
        and finger_curled(pinky_ext)
    ):
        return Gesture.THREE_FINGERS

    # Open palm: all four fingers extended (thumb state ignored)
    if (
        finger_extended(index_ext)
        and finger_extended(middle_ext)
        and finger_extended(ring_ext)
        and finger_extended(pinky_ext)
    ):
        return Gesture.OPEN_PALM

    return Gesture.NONE


# ---------------------------------------------------------------------------
# Filter placement
# ---------------------------------------------------------------------------


@dataclass
class FilterPlacement:
    """Computed position and size for an AR filter overlay.

    All values are in normalized coordinates (0..1).
    """

    filter_type: FilterType
    center_x: float
    center_y: float
    width: float
    height: float


def compute_filter_placement(
    face_landmarks: List[Landmark],
    filter_type: FilterType,
    config: FilterConfig,
) -> Optional[FilterPlacement]:
    """Compute where to draw a filter overlay on the face.

    Uses the interocular distance as the scale reference so the filter
    tracks face size naturally.

    Args:
        face_landmarks: Face mesh landmarks (at least 468 points, normalized).
        filter_type: Which filter to place.
        config: Filter positioning ratios.

    Returns:
        A FilterPlacement with normalized coords, or None if the filter
        type is NONE or landmarks are insufficient.
    """
    if filter_type == FilterType.NONE:
        return None

    # Need at least enough landmarks for our indices
    max_idx = max(FACE_LEFT_EYE, FACE_RIGHT_EYE, FACE_BETWEEN_EYES,
                  FACE_FOREHEAD, FACE_NOSE_TIP)
    if len(face_landmarks) <= max_idx:
        return None

    left_eye = face_landmarks[FACE_LEFT_EYE]
    right_eye = face_landmarks[FACE_RIGHT_EYE]
    between_eyes = face_landmarks[FACE_BETWEEN_EYES]
    forehead = face_landmarks[FACE_FOREHEAD]
    nose_tip = face_landmarks[FACE_NOSE_TIP]

    # Interocular distance as our scale unit
    iod = _distance(left_eye, right_eye)
    if iod < 1e-6:
        return None

    # Select anchor point and sizing based on filter type
    # All superhero masks are anchored at between_eyes for full-face coverage
    if filter_type == FilterType.WEB_SLINGER:
        placement_cfg = config.web_slinger
        anchor = between_eyes
    elif filter_type == FilterType.SUNGLASSES:
        placement_cfg = config.sunglasses
        anchor = between_eyes
    elif filter_type == FilterType.ARMOR_VISOR:
        placement_cfg = config.armor_visor
        anchor = between_eyes
    elif filter_type == FilterType.PATRIOTIC_MASK:
        placement_cfg = config.patriotic_mask
        anchor = between_eyes
    else:
        return None

    width = iod * placement_cfg.width_ratio
    height = iod * placement_cfg.height_ratio
    center_x = anchor[0]
    center_y = anchor[1] + iod * placement_cfg.y_offset_ratio

    return FilterPlacement(
        filter_type=filter_type,
        center_x=center_x,
        center_y=center_y,
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# Stateful gesture-to-filter controller
# ---------------------------------------------------------------------------


class GestureFilterState:
    """Hysteresis-based state machine mapping gestures to active filters.

    A gesture must persist for ``enter_frames`` consecutive frames before
    activating its mapped filter. Once a filter is active, it stays until
    a *different* gesture activates a different filter — the open palm gesture
    (mapped to FilterType.NONE) explicitly clears the filter.

    If no gesture is detected for ``release_frames`` consecutive frames,
    the candidate resets but the current filter remains.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.current_filter: FilterType = FilterType.NONE

        self._candidate_gesture: Gesture = Gesture.NONE
        self._candidate_frames: int = 0
        self._release_frames: int = 0

    def update(
        self,
        hand_landmarks: Optional[List[Landmark]],
        face_landmarks: Optional[List[Landmark]],
        timestamp: float,
    ) -> Dict[str, object]:
        """Process one frame of landmarks and return the current state.

        Args:
            hand_landmarks: 21 hand landmarks or None if no hand detected.
            face_landmarks: Face mesh landmarks or None if no face detected.
            timestamp: Frame timestamp in seconds (for future use).

        Returns:
            Dict with keys:
                - gesture: The confirmed active Gesture.
                - filter_type: The currently active FilterType.
                - placement: FilterPlacement or None.
                - raw_gesture: The instantaneous (unfiltered) detected gesture.
        """
        gesture_cfg = self.config.gesture
        filter_cfg = self.config.filter

        # Detect raw gesture from hand landmarks
        if hand_landmarks and len(hand_landmarks) >= 21:
            raw_gesture = detect_gesture(hand_landmarks, gesture_cfg)
        else:
            raw_gesture = Gesture.NONE

        # Hysteresis logic
        if raw_gesture != Gesture.NONE:
            self._release_frames = 0

            if raw_gesture == self._candidate_gesture:
                self._candidate_frames += 1
            else:
                # New candidate detected
                self._candidate_gesture = raw_gesture
                self._candidate_frames = 1

            # Activate if candidate persists long enough
            if self._candidate_frames >= gesture_cfg.enter_frames:
                new_filter = gesture_cfg.gesture_filter_map.get(
                    self._candidate_gesture, FilterType.NONE
                )
                self.current_filter = new_filter
        else:
            # No gesture detected this frame
            self._release_frames += 1

            if self._release_frames >= gesture_cfg.release_frames:
                # Clear candidate but keep current filter
                self._candidate_gesture = Gesture.NONE
                self._candidate_frames = 0

        # Compute filter placement if we have a face
        placement: Optional[FilterPlacement] = None
        if self.current_filter != FilterType.NONE and face_landmarks:
            placement = compute_filter_placement(
                face_landmarks, self.current_filter, filter_cfg
            )

        return {
            "gesture": self._candidate_gesture
            if self._candidate_frames >= gesture_cfg.enter_frames
            else Gesture.NONE,
            "filter_type": self.current_filter,
            "placement": placement,
            "raw_gesture": raw_gesture,
        }
