"""Pytest tests for core.py — gesture detection, filter placement, and hysteresis.

All tests use synthetic landmark data and require NO webcam or camera access.
"""

import sys
import os
from typing import List, Tuple

# Add the projects directory to path so the package can be imported
_project_dir = os.path.dirname(os.path.abspath(__file__))
_projects_dir = os.path.dirname(_project_dir)
sys.path.insert(0, _projects_dir)

from mateokael.config import AppConfig, FilterType, Gesture, GestureConfig
from mateokael.core import (
    WRIST,
    THUMB_TIP,
    INDEX_TIP,
    INDEX_MCP,
    MIDDLE_TIP,
    MIDDLE_MCP,
    RING_TIP,
    RING_MCP,
    PINKY_TIP,
    PINKY_MCP,
    FACE_LEFT_EYE,
    FACE_RIGHT_EYE,
    FACE_BETWEEN_EYES,
    FACE_FOREHEAD,
    FACE_NOSE_TIP,
    FilterPlacement,
    GestureFilterState,
    compute_filter_placement,
    detect_gesture,
)

Landmark = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Synthetic hand landmark helpers
# ---------------------------------------------------------------------------


def _base_hand_landmarks() -> List[Landmark]:
    """Create 21 hand landmarks at a default neutral position.

    Wrist at (0.5, 0.8, 0.0), middle MCP at (0.5, 0.55, 0.0) giving a
    hand_span of 0.25. Fingertips default to near their MCP (curled).

    With the tip-to-MCP metric:
      hand_span = 0.25
      extended threshold = 0.6 * 0.25 = 0.15 (tip-to-MCP must exceed)
      curled threshold = 0.3 * 0.25 = 0.075 (tip-to-MCP must be below)
    """
    landmarks: List[Landmark] = [(0.5, 0.5, 0.0)] * 21

    # Wrist
    landmarks[WRIST] = (0.5, 0.8, 0.0)

    # MCP joints (knuckle row) — spread horizontally around x=0.5
    landmarks[INDEX_MCP] = (0.42, 0.55, 0.0)
    landmarks[MIDDLE_MCP] = (0.5, 0.55, 0.0)  # hand_span reference
    landmarks[RING_MCP] = (0.58, 0.55, 0.0)
    landmarks[PINKY_MCP] = (0.65, 0.55, 0.0)

    # Thumb base joints
    landmarks[1] = (0.35, 0.75, 0.0)  # THUMB_CMC
    landmarks[2] = (0.32, 0.70, 0.0)  # THUMB_MCP
    landmarks[3] = (0.30, 0.65, 0.0)  # THUMB_IP

    # PIP/DIP for fingers (between MCP and tip)
    landmarks[6] = (0.42, 0.52, 0.0)  # INDEX_PIP
    landmarks[7] = (0.42, 0.50, 0.0)  # INDEX_DIP
    landmarks[10] = (0.5, 0.52, 0.0)  # MIDDLE_PIP
    landmarks[11] = (0.5, 0.50, 0.0)  # MIDDLE_DIP
    landmarks[14] = (0.58, 0.52, 0.0)  # RING_PIP
    landmarks[15] = (0.58, 0.50, 0.0)  # RING_DIP
    landmarks[18] = (0.65, 0.52, 0.0)  # PINKY_PIP
    landmarks[19] = (0.65, 0.50, 0.0)  # PINKY_DIP

    # Default tips near their MCP (curled position)
    # tip-to-MCP ~0.05 which is < curled threshold 0.075
    landmarks[THUMB_TIP] = (0.34, 0.68, 0.0)
    landmarks[INDEX_TIP] = (0.42, 0.58, 0.0)
    landmarks[MIDDLE_TIP] = (0.50, 0.58, 0.0)
    landmarks[RING_TIP] = (0.58, 0.58, 0.0)
    landmarks[PINKY_TIP] = (0.65, 0.58, 0.0)

    return landmarks


def make_peace_landmarks() -> List[Landmark]:
    """Peace sign: index + middle fingers extended, ring + pinky curled.

    hand_span = 0.25
    Extended: tip-to-MCP / hand_span > 0.6 → tip-to-MCP > 0.15
    Curled: tip-to-MCP / hand_span < 0.3 → tip-to-MCP < 0.075
    """
    landmarks = _base_hand_landmarks()

    # Index tip far from MCP (extended): tip-to-MCP = 0.20, ratio = 0.80
    landmarks[INDEX_TIP] = (0.42, 0.35, 0.0)

    # Middle tip far from MCP (extended): tip-to-MCP = 0.20, ratio = 0.80
    landmarks[MIDDLE_TIP] = (0.50, 0.35, 0.0)

    # Ring tip near its MCP (curled): tip-to-MCP = 0.05, ratio = 0.20
    landmarks[RING_TIP] = (0.58, 0.60, 0.0)

    # Pinky tip near its MCP (curled): tip-to-MCP = 0.05, ratio = 0.20
    landmarks[PINKY_TIP] = (0.65, 0.60, 0.0)

    return landmarks


def make_thumbs_up_landmarks() -> List[Landmark]:
    """Thumbs up: thumb extended far from wrist, all four fingers curled.

    thumb uses tip-to-wrist / hand_span:
      thumb_extended_threshold = 0.45, so tip-to-wrist / 0.25 > 0.45 → dist > 0.1125
    fingers use tip-to-MCP / hand_span:
      curled threshold = 0.3, so tip-to-MCP / 0.25 < 0.3 → dist < 0.075
    """
    landmarks = _base_hand_landmarks()

    # Thumb tip far from wrist (extended): distance ~0.45 > 0.1125
    landmarks[THUMB_TIP] = (0.30, 0.40, 0.0)

    # All fingers curled: tips near their MCPs
    landmarks[INDEX_TIP] = (0.42, 0.58, 0.0)   # dist to MCP ~0.03
    landmarks[MIDDLE_TIP] = (0.50, 0.58, 0.0)  # dist to MCP ~0.03
    landmarks[RING_TIP] = (0.58, 0.58, 0.0)    # dist to MCP ~0.03
    landmarks[PINKY_TIP] = (0.65, 0.58, 0.0)   # dist to MCP ~0.03

    return landmarks


def make_rock_landmarks() -> List[Landmark]:
    """Rock/horns sign: index + pinky extended, middle + ring curled.

    fingers use tip-to-MCP / hand_span:
      extended threshold = 0.6, so tip-to-MCP / 0.25 > 0.6 → dist > 0.15
      curled threshold = 0.3, so tip-to-MCP / 0.25 < 0.3 → dist < 0.075
    """
    landmarks = _base_hand_landmarks()

    # Index extended: tip far from MCP, dist = 0.20, ratio = 0.80
    landmarks[INDEX_TIP] = (0.42, 0.35, 0.0)

    # Middle curled: tip near MCP, dist ~0.03
    landmarks[MIDDLE_TIP] = (0.50, 0.58, 0.0)

    # Ring curled: tip near MCP, dist ~0.03
    landmarks[RING_TIP] = (0.58, 0.58, 0.0)

    # Pinky extended: tip far from MCP, dist = 0.20, ratio = 0.80
    landmarks[PINKY_TIP] = (0.65, 0.35, 0.0)

    return landmarks


def make_three_fingers_landmarks() -> List[Landmark]:
    """Three-finger salute: index + middle + ring extended, pinky curled.

    fingers use tip-to-MCP / hand_span:
      extended threshold = 0.6, so tip-to-MCP / 0.25 > 0.6 → dist > 0.15
      curled threshold = 0.3, so tip-to-MCP / 0.25 < 0.3 → dist < 0.075
    """
    landmarks = _base_hand_landmarks()

    # Index extended: tip far from MCP, dist = 0.20, ratio = 0.80
    landmarks[INDEX_TIP] = (0.42, 0.35, 0.0)

    # Middle extended: tip far from MCP, dist = 0.20, ratio = 0.80
    landmarks[MIDDLE_TIP] = (0.50, 0.35, 0.0)

    # Ring extended: tip far from MCP, dist = 0.20, ratio = 0.80
    landmarks[RING_TIP] = (0.58, 0.35, 0.0)

    # Pinky curled: tip near MCP, dist ~0.03
    landmarks[PINKY_TIP] = (0.65, 0.58, 0.0)

    return landmarks


def make_open_palm_landmarks() -> List[Landmark]:
    """Open palm: all four fingers extended far from their MCPs.

    finger_extended: tip-to-MCP / hand_span > 0.6 → tip-to-MCP > 0.15
    """
    landmarks = _base_hand_landmarks()

    # All fingertips far from their MCPs (extended): tip-to-MCP = 0.20, ratio = 0.80
    landmarks[INDEX_TIP] = (0.42, 0.35, 0.0)
    landmarks[MIDDLE_TIP] = (0.50, 0.35, 0.0)
    landmarks[RING_TIP] = (0.58, 0.35, 0.0)
    landmarks[PINKY_TIP] = (0.65, 0.35, 0.0)

    # Thumb extended too
    landmarks[THUMB_TIP] = (0.30, 0.40, 0.0)

    return landmarks


def make_no_gesture_landmarks() -> List[Landmark]:
    """Ambiguous gesture: only index extended (pointing), doesn't match any pattern.

    This should return Gesture.NONE since no defined gesture matches one-finger-only.
    """
    landmarks = _base_hand_landmarks()

    # Only index extended: tip-to-MCP = 0.20, ratio = 0.80
    landmarks[INDEX_TIP] = (0.42, 0.35, 0.0)

    # Middle, ring, pinky curled: tips near MCPs
    landmarks[MIDDLE_TIP] = (0.50, 0.58, 0.0)
    landmarks[RING_TIP] = (0.58, 0.58, 0.0)
    landmarks[PINKY_TIP] = (0.65, 0.58, 0.0)

    # Thumb neutral (not clearly extended for thumbs_up, not clearly curled)
    # tip-to-wrist: distance((0.40, 0.72), (0.5, 0.8)) = sqrt(0.01+0.0064) ~= 0.128
    # ratio = 0.128/0.25 = 0.51 — between thresholds (not extended, not curled)
    landmarks[THUMB_TIP] = (0.40, 0.72, 0.0)

    return landmarks


# ---------------------------------------------------------------------------
# Synthetic face landmark helper
# ---------------------------------------------------------------------------


def make_face_landmarks(
    left_eye: Landmark = (0.35, 0.4, 0.0),
    right_eye: Landmark = (0.65, 0.4, 0.0),
    between_eyes: Landmark = (0.5, 0.4, 0.0),
    forehead: Landmark = (0.5, 0.3, 0.0),
    nose_tip: Landmark = (0.5, 0.55, 0.0),
) -> List[Landmark]:
    """Build 468 face mesh landmarks with controllable key positions.

    Most landmarks are set to (0.5, 0.5, 0.0); the key indices used by
    compute_filter_placement are set to the provided values.

    Args:
        left_eye: Position of left eye (index 33).
        right_eye: Position of right eye (index 263).
        between_eyes: Position between eyes (index 168).
        forehead: Position of forehead (index 10).
        nose_tip: Position of nose tip (index 1).

    Returns:
        List of 468 (x, y, z) tuples.
    """
    landmarks: List[Landmark] = [(0.5, 0.5, 0.0)] * 468

    landmarks[FACE_LEFT_EYE] = left_eye
    landmarks[FACE_RIGHT_EYE] = right_eye
    landmarks[FACE_BETWEEN_EYES] = between_eyes
    landmarks[FACE_FOREHEAD] = forehead
    landmarks[FACE_NOSE_TIP] = nose_tip

    return landmarks


# ---------------------------------------------------------------------------
# Gesture detection tests
# ---------------------------------------------------------------------------


class TestDetectGesture:
    """Tests for detect_gesture() with synthetic hand landmarks."""

    def setup_method(self) -> None:
        self.config = AppConfig().gesture

    def test_detect_peace(self) -> None:
        """Peace landmarks (index+middle extended, ring+pinky curled) -> Gesture.PEACE."""
        landmarks = make_peace_landmarks()
        result = detect_gesture(landmarks, self.config)
        assert result == Gesture.PEACE

    def test_detect_thumbs_up(self) -> None:
        """Thumbs up landmarks (thumb extended, all fingers curled) -> Gesture.THUMBS_UP."""
        landmarks = make_thumbs_up_landmarks()
        result = detect_gesture(landmarks, self.config)
        assert result == Gesture.THUMBS_UP

    def test_detect_rock(self) -> None:
        """Rock landmarks (index+pinky extended, middle+ring curled) -> Gesture.ROCK."""
        landmarks = make_rock_landmarks()
        result = detect_gesture(landmarks, self.config)
        assert result == Gesture.ROCK

    def test_detect_three_fingers(self) -> None:
        """Three fingers landmarks (index+middle+ring extended, pinky curled) -> Gesture.THREE_FINGERS."""
        landmarks = make_three_fingers_landmarks()
        result = detect_gesture(landmarks, self.config)
        assert result == Gesture.THREE_FINGERS

    def test_detect_open_palm(self) -> None:
        """Open palm landmarks (all four fingers extended) -> Gesture.OPEN_PALM."""
        landmarks = make_open_palm_landmarks()
        result = detect_gesture(landmarks, self.config)
        assert result == Gesture.OPEN_PALM

    def test_detect_none(self) -> None:
        """Ambiguous landmarks (only index extended) -> Gesture.NONE."""
        landmarks = make_no_gesture_landmarks()
        result = detect_gesture(landmarks, self.config)
        assert result == Gesture.NONE


# ---------------------------------------------------------------------------
# Hysteresis tests
# ---------------------------------------------------------------------------


class TestHysteresis:
    """Tests for GestureFilterState hysteresis behavior."""

    def setup_method(self) -> None:
        self.config = AppConfig()
        self.state = GestureFilterState(self.config)
        self.face = make_face_landmarks()

    def test_hysteresis_enter(self) -> None:
        """Gesture must persist for enter_frames before activating the filter.

        Default enter_frames = 5. Feed peace for 4 frames: filter stays NONE.
        On the 5th frame the filter activates.
        """
        peace = make_peace_landmarks()
        enter_frames = self.config.gesture.enter_frames

        # Feed gesture for (enter_frames - 1) frames — should NOT activate yet
        for i in range(enter_frames - 1):
            result = self.state.update(peace, self.face, float(i))
            assert result["filter_type"] == FilterType.NONE, (
                f"Filter activated too early on frame {i}"
            )

        # On the enter_frames-th frame, the filter should activate
        result = self.state.update(peace, self.face, float(enter_frames))
        expected_filter = self.config.gesture.gesture_filter_map[Gesture.PEACE]
        assert result["filter_type"] == expected_filter
        assert result["gesture"] == Gesture.PEACE

    def test_hysteresis_no_flicker(self) -> None:
        """A single frame of a different gesture does not change the active filter.

        Activate peace (WEB_SLINGER), then inject one frame of thumbs_up.
        The filter should remain WEB_SLINGER because one frame is not enough
        to reach enter_frames for the new gesture.
        """
        peace = make_peace_landmarks()
        thumbs_up = make_thumbs_up_landmarks()
        enter_frames = self.config.gesture.enter_frames

        # Activate peace filter
        for i in range(enter_frames):
            self.state.update(peace, self.face, float(i))

        # Confirm peace is active
        result = self.state.update(peace, self.face, float(enter_frames))
        peace_filter = self.config.gesture.gesture_filter_map[Gesture.PEACE]
        assert result["filter_type"] == peace_filter

        # Inject ONE frame of thumbs_up — should not change filter
        result = self.state.update(thumbs_up, self.face, float(enter_frames + 1))
        assert result["filter_type"] == peace_filter, (
            "Filter flickered on a single different-gesture frame"
        )

        # Go back to peace — filter should still be the peace filter
        result = self.state.update(peace, self.face, float(enter_frames + 2))
        assert result["filter_type"] == peace_filter


# ---------------------------------------------------------------------------
# Filter placement tests
# ---------------------------------------------------------------------------


class TestFilterPlacement:
    """Tests for compute_filter_placement() with synthetic face landmarks."""

    def setup_method(self) -> None:
        self.config = AppConfig().filter
        self.face = make_face_landmarks()

    def test_filter_placement_sunglasses(self) -> None:
        """Sunglasses filter produces a valid non-None FilterPlacement."""
        placement = compute_filter_placement(
            self.face, FilterType.SUNGLASSES, self.config
        )
        assert placement is not None
        assert isinstance(placement, FilterPlacement)
        assert placement.filter_type == FilterType.SUNGLASSES
        assert placement.width > 0
        assert placement.height > 0
        # Center should be near between_eyes (0.5, 0.4)
        assert 0.0 < placement.center_x < 1.0
        assert 0.0 < placement.center_y < 1.0

    def test_filter_placement_none(self) -> None:
        """FilterType.NONE returns None placement (no filter to draw)."""
        placement = compute_filter_placement(
            self.face, FilterType.NONE, self.config
        )
        assert placement is None

    def test_no_face_no_placement(self) -> None:
        """If face_landmarks is None (no face detected), placement is None.

        compute_filter_placement expects a list, so we pass an empty list
        to simulate no usable face data.
        """
        # Empty list — not enough landmarks
        placement = compute_filter_placement([], FilterType.SUNGLASSES, self.config)
        assert placement is None

    def test_filter_placement_web_slinger(self) -> None:
        """Web-slinger mask is positioned around between_eyes."""
        placement = compute_filter_placement(
            self.face, FilterType.WEB_SLINGER, self.config
        )
        assert placement is not None
        assert placement.filter_type == FilterType.WEB_SLINGER
        # Should be centered near between_eyes (0.5, 0.4)
        assert 0.3 < placement.center_x < 0.7
        assert placement.width > 0
        assert placement.height > 0

    def test_filter_placement_armor_visor(self) -> None:
        """Armor visor is positioned at eye level."""
        placement = compute_filter_placement(
            self.face, FilterType.ARMOR_VISOR, self.config
        )
        assert placement is not None
        assert placement.filter_type == FilterType.ARMOR_VISOR
        # Anchored at between_eyes (0.5, 0.4) with positive offset pushing it lower
        assert 0.3 < placement.center_y < 0.6

    def test_filter_placement_patriotic_mask(self) -> None:
        """Patriotic mask is positioned around between_eyes."""
        placement = compute_filter_placement(
            self.face, FilterType.PATRIOTIC_MASK, self.config
        )
        assert placement is not None
        assert placement.filter_type == FilterType.PATRIOTIC_MASK
        assert 0.3 < placement.center_x < 0.7
