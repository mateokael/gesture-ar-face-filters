"""Desktop webcam entry point for the gesture-controlled AR filter app.

Opens the camera, runs MediaPipe hand + face tracking, detects gestures via
the pure logic in core.py, and overlays PNG filters on the face in real time.

Usage:
    python main.py [--camera 0]
"""

import argparse
import os
import sys
import time
from typing import Dict, Optional

import cv2
import mediapipe as mp
import numpy as np

try:
    # When run as part of the package (python -m projects.mateokael.main)
    from .config import AppConfig, FilterType
    from .core import GestureFilterState
except ImportError:
    # When run directly (python main.py) from the project directory
    from config import AppConfig, FilterType
    from core import GestureFilterState


# ---------------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------------


def load_assets(assets_dir: str) -> Dict[FilterType, Optional[np.ndarray]]:
    """Load PNG overlay images (with alpha channel) from the assets directory.

    Expected filenames:
        web_slinger.png, sunglasses.png, armor_visor.png, patriotic_mask.png

    Args:
        assets_dir: Path to the folder containing PNG assets.

    Returns:
        Dict mapping each FilterType to its BGRA image array, or None if
        the file is missing.
    """
    asset_map: Dict[FilterType, Optional[np.ndarray]] = {
        FilterType.NONE: None,
    }

    filenames = {
        FilterType.WEB_SLINGER: "web_slinger.png",
        FilterType.SUNGLASSES: "sunglasses.png",
        FilterType.ARMOR_VISOR: "armor_visor.png",
        FilterType.PATRIOTIC_MASK: "patriotic_mask.png",
    }

    for filter_type, filename in filenames.items():
        filepath = os.path.join(assets_dir, filename)
        if os.path.isfile(filepath):
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] == 4:
                asset_map[filter_type] = img
            else:
                print(f"[WARN] {filepath} is not a valid BGRA PNG, skipping.")
                asset_map[filter_type] = None
        else:
            print(f"[WARN] Asset not found: {filepath}")
            asset_map[filter_type] = None

    return asset_map


# ---------------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------------


def overlay_png(
    frame: np.ndarray,
    png_rgba: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Alpha-blend a BGRA PNG onto the frame at the given position and size.

    Handles clipping when the overlay extends beyond frame boundaries.
    Modifies `frame` in place.

    Args:
        frame: BGR frame to draw on (H x W x 3).
        png_rgba: Source BGRA image (any size, will be resized).
        x: Top-left x coordinate on the frame.
        y: Top-left y coordinate on the frame.
        w: Desired width of the overlay on the frame.
        h: Desired height of the overlay on the frame.
    """
    if w <= 0 or h <= 0:
        return

    # Resize the overlay to the desired dimensions
    resized = cv2.resize(png_rgba, (w, h), interpolation=cv2.INTER_AREA)

    frame_h, frame_w = frame.shape[:2]

    # Compute clipping bounds
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, frame_w)
    y2 = min(y + h, frame_h)

    if x1 >= x2 or y1 >= y2:
        return

    # Corresponding region in the resized overlay
    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    # Extract overlay region and alpha
    overlay_region = resized[oy1:oy2, ox1:ox2]
    alpha = overlay_region[:, :, 3:4].astype(np.float32) / 255.0
    bgr = overlay_region[:, :, :3].astype(np.float32)

    # Blend
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    blended = bgr * alpha + roi * (1.0 - alpha)
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)


# ---------------------------------------------------------------------------
# HUD drawing
# ---------------------------------------------------------------------------


def draw_hud(frame: np.ndarray, gesture_name: str, filter_name: str) -> None:
    """Draw a small translucent HUD in the top-left corner.

    Shows the current detected gesture and active filter.

    Args:
        frame: BGR frame to draw on (modified in place).
        gesture_name: Display name of the current gesture.
        filter_name: Display name of the active filter.
    """
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Text
    cv2.putText(
        frame,
        f"Gesture: {gesture_name}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Filter: {filter_name}",
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 200),
        1,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the desktop webcam gesture-controlled AR filter application."""
    parser = argparse.ArgumentParser(
        description="Gesture-controlled AR filter (desktop webcam)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    args = parser.parse_args()

    # Configuration
    config = AppConfig(camera_index=args.camera)

    # Resolve assets directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(script_dir, config.assets_dir)

    # Load overlay assets
    assets = load_assets(assets_dir)

    # Initialize state machine
    state = GestureFilterState(config)

    # Initialize MediaPipe (legacy mp.solutions API)
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Open webcam
    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {config.camera_index}.")
        print("        Try a different index with --camera 1")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    print(f"[INFO] Camera {config.camera_index} opened. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame, exiting.")
                break

            # Mirror the frame for intuitive interaction
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            timestamp = time.time()

            # Convert BGR -> RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            # Process hand landmarks
            hand_results = hands.process(rgb_frame)
            hand_landmarks = None
            if hand_results.multi_hand_landmarks:
                hand_lm = hand_results.multi_hand_landmarks[0]
                hand_landmarks = [
                    (lm.x, lm.y, lm.z) for lm in hand_lm.landmark
                ]

            # Process face landmarks
            face_results = face_mesh.process(rgb_frame)
            face_landmarks = None
            if face_results.multi_face_landmarks:
                face_lm = face_results.multi_face_landmarks[0]
                face_landmarks = [
                    (lm.x, lm.y, lm.z) for lm in face_lm.landmark
                ]

            # Update state machine
            result = state.update(hand_landmarks, face_landmarks, timestamp)

            # Overlay filter if placement is computed
            placement = result["placement"]
            if placement is not None:
                png_asset = assets.get(placement.filter_type)
                if png_asset is not None:
                    # Convert normalized coords to pixel coords
                    cx = int(placement.center_x * frame_w)
                    cy = int(placement.center_y * frame_h)
                    pw = int(placement.width * frame_w)
                    ph = int(placement.height * frame_h)

                    # Top-left from center
                    px = cx - pw // 2
                    py = cy - ph // 2

                    overlay_png(frame, png_asset, px, py, pw, ph)

            # Draw HUD
            gesture_name = result["gesture"].value.replace("_", " ").title()
            filter_name = result["filter_type"].value.replace("_", " ").title()
            draw_hud(frame, gesture_name, filter_name)

            # Display
            cv2.imshow("Gesture AR Filter", frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        face_mesh.close()
        print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    main()
