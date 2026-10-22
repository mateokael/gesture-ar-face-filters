# Gesture AR Face Filters

A real-time gesture-controlled AR face filter app built with **MediaPipe**, **OpenCV**, and **Flask**. Show a hand gesture to your webcam and watch a face filter snap onto your face instantly — no buttons, no clicks.

Runs in two modes: a **desktop app** (OpenCV window) and a **web app** (browser + WebSocket).

---

## Demo

| Gesture | Filter Applied |
|---|---|
| ✌️ Peace | 🕷️ Web-Slinger Mask |
| 👍 Thumbs Up | 🕶️ Sunglasses |
| 🤘 Rock / Horns | ⚙️ Armor Visor |
| 3️⃣ Three Fingers | 🛡️ Patriotic Mask |
| 🖐️ Open Palm | ❌ Clears filter |

---

## How It Works

1. **MediaPipe Hands** tracks 21 hand landmarks per frame
2. **Gesture detection** (`core.py`) classifies the gesture using scale-relative finger extension ratios — works at any distance from the camera
3. A **hysteresis state machine** holds the gesture for 5 frames before activating (prevents flickering)
4. **MediaPipe Face Mesh** provides 468 face landmarks used to anchor and scale the filter overlay
5. A PNG overlay is alpha-blended onto the frame at the computed position

---

## Project Structure

```
gesture-ar-face-filters/
├── config.py            # All tunables: thresholds, filter ratios, app settings
├── core.py              # Pure gesture detection + filter placement logic (no I/O)
├── main.py              # Desktop: OpenCV webcam loop with PNG overlay
├── web.py               # Web: Flask + WebSocket server
├── make_assets.py       # Generates placeholder PNG filter assets
├── test_core.py         # Unit tests (no webcam needed)
├── __init__.py          # Package exports
├── assets/              # Filter PNG images (BGRA, transparent background)
│   ├── web_slinger.png
│   ├── sunglasses.png
│   ├── armor_visor.png
│   └── patriotic_mask.png
└── templates/
    └── index.html       # Browser UI with client-side MediaPipe
```

---

## Setup

**Python 3.10+ recommended.**

```bash
# Clone the repo
git clone https://github.com/mateokael/gesture-ar-face-filters.git
cd gesture-ar-face-filters

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Generate placeholder PNG filter assets
python make_assets.py
```

---

## Run — Desktop Mode

Uses your webcam directly with an OpenCV window.

```bash
python main.py              # default camera (index 0)
python main.py --camera 1   # alternate camera
```

Press **`q`** to quit.

---

## Run — Web Mode

MediaPipe runs in the **browser**. Landmarks are streamed to the Flask server over WebSocket. No server-side camera access needed.

```bash
python web.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## Run Tests

Tests use synthetic landmark data — no webcam required.

```bash
pytest test_core.py -v
```

---

## Architecture

```
┌──────────────────┐   landmarks   ┌─────────────────┐   state   ┌──────────────────┐
│  Camera /        │ ────────────▶ │    core.py       │ ────────▶ │  Renderer        │
│  Browser (JS)    │               │  (pure logic)    │           │  (overlay / WS)  │
└──────────────────┘               └─────────────────┘           └──────────────────┘
```

- **`core.py`** — Pure functions, zero I/O. Testable without a webcam. Handles gesture classification, filter placement math, and hysteresis state management.
- **`main.py`** — Thin desktop adapter. Captures frames → feeds landmarks to core → draws result.
- **`web.py`** — Thin web adapter. Receives landmarks from browser via WebSocket → feeds to core → returns JSON placement.
- **`config.py`** — Single source of truth for all tunables. Nothing is hardcoded in the logic.

---

## Gesture Detection Logic

Gestures are classified using **scale-relative finger extension ratios**:

```
extension_ratio = distance(fingertip, MCP_joint) / hand_span
```

Where `hand_span = distance(wrist, middle_MCP)`. This makes detection invariant to how close or far your hand is from the camera.

| Ratio > 0.6 | Finger is extended |
|---|---|
| Ratio < 0.3 | Finger is curled |

---

## Dependencies

| Package | Purpose |
|---|---|
| `mediapipe` | Hand and face landmark detection |
| `opencv-python` | Webcam capture and frame rendering |
| `flask` | Web server |
| `flask-sock` | WebSocket support |
| `numpy` | Alpha blending math |
| `pytest` | Unit testing |

---

## Built With

This project was built as part of the [kiro-ph-computer-vision](https://github.com/mateokael/kiro-ph-computer-vision) workshop — a hands-on computer vision learning series using Kiro AI.
