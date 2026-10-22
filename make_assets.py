"""Generate placeholder PNG overlay assets for the face filter app.

Creates RGBA images with transparency using only NumPy and cv2.
Run standalone: python make_assets.py
"""

import os

import cv2
import numpy as np


def make_web_slinger_mask(path: str) -> None:
    """Create a 200x220 RGBA web-slinger style mask.

    Red mask with white eye shapes and a web pattern radiating from the nose.
    """
    h, w = 220, 200
    img = np.zeros((h, w, 4), dtype=np.uint8)

    # Red base mask shape (oval covering face)
    cv2.ellipse(img, (100, 110), (90, 105), 0, 0, 360, (0, 0, 200, 255), -1)

    # Darker red border
    cv2.ellipse(img, (100, 110), (90, 105), 0, 0, 360, (0, 0, 150, 255), 3)

    # Web lines radiating from nose center
    center = (100, 120)
    web_color = (50, 50, 50, 255)
    # Radial lines
    for angle in range(0, 360, 30):
        end_x = int(center[0] + 95 * np.cos(np.radians(angle)))
        end_y = int(center[1] + 110 * np.sin(np.radians(angle)))
        cv2.line(img, center, (end_x, end_y), web_color, 1)

    # Concentric web arcs
    for r in range(25, 100, 20):
        cv2.ellipse(img, center, (r, int(r * 1.1)), 0, 0, 360, web_color, 1)

    # Large white angular eye shapes
    left_eye_pts = np.array([[55, 80], [75, 65], [90, 80], [75, 100]], np.int32)
    right_eye_pts = np.array([[145, 80], [125, 65], [110, 80], [125, 100]], np.int32)
    cv2.fillPoly(img, [left_eye_pts], (255, 255, 255, 255))
    cv2.fillPoly(img, [right_eye_pts], (255, 255, 255, 255))

    # Black outline around eyes
    cv2.polylines(img, [left_eye_pts], True, (0, 0, 0, 255), 2)
    cv2.polylines(img, [right_eye_pts], True, (0, 0, 0, 255), 2)

    cv2.imwrite(path, img)
    print(f"  Created: {path} ({w}x{h} RGBA)")


def make_sunglasses(path: str) -> None:
    """Create a 200x70 RGBA sunglasses: two dark circles + bridge line."""
    img = np.zeros((70, 200, 4), dtype=np.uint8)

    # Left lens
    cv2.circle(img, (60, 35), 25, (30, 30, 30, 255), -1)
    # Right lens
    cv2.circle(img, (140, 35), 25, (30, 30, 30, 255), -1)
    # Bridge connecting lenses
    cv2.line(img, (85, 35), (115, 35), (50, 50, 50, 255), 3)
    # Thin frame outline
    cv2.circle(img, (60, 35), 25, (60, 60, 60, 255), 2)
    cv2.circle(img, (140, 35), 25, (60, 60, 60, 255), 2)

    cv2.imwrite(path, img)
    print(f"  Created: {path} (200x70 RGBA)")


def make_armor_visor(path: str) -> None:
    """Create a 200x130 RGBA metal armor visor.

    Gold/red metallic visor with a T-shaped opening and rivets.
    """
    h, w = 130, 200
    img = np.zeros((h, w, 4), dtype=np.uint8)

    # Gold metallic helmet shape (rounded rectangle)
    cv2.ellipse(img, (100, 65), (90, 60), 0, 0, 360, (0, 180, 220, 255), -1)

    # Darker gold border
    cv2.ellipse(img, (100, 65), (90, 60), 0, 0, 360, (0, 140, 180, 255), 3)

    # Red accent stripes on sides
    cv2.ellipse(img, (100, 65), (90, 60), 0, 150, 180, (0, 0, 180, 255), 8)
    cv2.ellipse(img, (100, 65), (90, 60), 0, 0, 30, (0, 0, 180, 255), 8)

    # T-shaped visor slit (dark opening)
    # Horizontal bar for eyes
    cv2.rectangle(img, (30, 45), (170, 65), (40, 40, 40, 255), -1)
    # Vertical bar for nose
    cv2.rectangle(img, (90, 45), (110, 100), (40, 40, 40, 255), -1)

    # Bright edge on the visor slit
    cv2.rectangle(img, (30, 45), (170, 65), (100, 230, 255, 255), 2)
    cv2.rectangle(img, (90, 45), (110, 100), (100, 230, 255, 255), 2)

    # Rivets (small bright circles)
    for x in [25, 50, 150, 175]:
        cv2.circle(img, (x, 30), 4, (150, 220, 240, 255), -1)
        cv2.circle(img, (x, 30), 4, (100, 180, 200, 255), 1)

    cv2.imwrite(path, img)
    print(f"  Created: {path} ({w}x{h} RGBA)")


def make_patriotic_mask(path: str) -> None:
    """Create a 200x180 RGBA patriotic hero mask.

    Blue mask with a white star on the forehead and red/white accents.
    Wing-shaped eye openings.
    """
    h, w = 180, 200
    img = np.zeros((h, w, 4), dtype=np.uint8)

    # Blue base mask (covers upper face and forehead)
    # Rounded shape wider at top
    pts = np.array([
        [100, 5], [160, 20], [185, 60], [180, 110],
        [160, 150], [100, 170], [40, 150], [20, 110],
        [15, 60], [40, 20]
    ], np.int32)
    cv2.fillPoly(img, [pts], (180, 50, 20, 255))  # Blue in BGR

    # White star on forehead
    star_center = (100, 45)
    star_pts = []
    for i in range(5):
        # Outer points
        angle = np.radians(-90 + i * 72)
        star_pts.append((
            int(star_center[0] + 22 * np.cos(angle)),
            int(star_center[1] + 22 * np.sin(angle))
        ))
        # Inner points
        angle = np.radians(-90 + i * 72 + 36)
        star_pts.append((
            int(star_center[0] + 10 * np.cos(angle)),
            int(star_center[1] + 10 * np.sin(angle))
        ))
    star_arr = np.array(star_pts, np.int32)
    cv2.fillPoly(img, [star_arr], (255, 255, 255, 255))

    # Wing-shaped eye openings (dark cutouts)
    left_eye = np.array([[45, 80], [65, 70], [85, 80], [65, 95]], np.int32)
    right_eye = np.array([[155, 80], [135, 70], [115, 80], [135, 95]], np.int32)
    cv2.fillPoly(img, [left_eye], (0, 0, 0, 0))   # Transparent cutouts
    cv2.fillPoly(img, [right_eye], (0, 0, 0, 0))

    # White outline around eyes
    cv2.polylines(img, [left_eye], True, (255, 255, 255, 255), 2)
    cv2.polylines(img, [right_eye], True, (255, 255, 255, 255), 2)

    # Red accent stripe across the brow
    cv2.line(img, (30, 65), (170, 65), (0, 0, 200, 255), 3)

    # Wing tips at temples (small triangles)
    left_wing = np.array([[15, 70], [5, 55], [25, 60]], np.int32)
    right_wing = np.array([[185, 70], [195, 55], [175, 60]], np.int32)
    cv2.fillPoly(img, [left_wing], (180, 50, 20, 255))
    cv2.fillPoly(img, [right_wing], (180, 50, 20, 255))

    cv2.imwrite(path, img)
    print(f"  Created: {path} ({w}x{h} RGBA)")


def main() -> None:
    """Generate all placeholder assets."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(script_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    print("Generating superhero mask placeholder assets...")
    make_web_slinger_mask(os.path.join(assets_dir, "web_slinger.png"))
    make_sunglasses(os.path.join(assets_dir, "sunglasses.png"))
    make_armor_visor(os.path.join(assets_dir, "armor_visor.png"))
    make_patriotic_mask(os.path.join(assets_dir, "patriotic_mask.png"))
    print(f"Done! Assets saved to: {assets_dir}")


if __name__ == "__main__":
    main()
