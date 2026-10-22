"""Flask web layer for the gesture-controlled AR filter app.

Architecture: the browser runs MediaPipe hand/face detection in JavaScript,
sends normalized landmarks to the server over a WebSocket, and the server
runs the pure core.py logic (gesture detection + filter placement) and sends
back the current filter state. The browser then draws overlays on a canvas.
"""

import json
import time

from flask import Flask, render_template
from flask_sock import Sock

try:
    from .config import AppConfig, FilterType, Gesture
    from .core import FilterPlacement, GestureFilterState
except ImportError:
    from config import AppConfig, FilterType, Gesture
    from core import FilterPlacement, GestureFilterState

config = AppConfig()

app = Flask(__name__, static_folder="assets", static_url_path="/static/assets")
sock = Sock(app)


@app.route("/")
def index() -> str:
    """Serve the main dashboard page."""
    return render_template("index.html")


@sock.route("/ws")
def websocket(ws) -> None:
    """WebSocket handler: one GestureFilterState per connection.

    Receives JSON:
        {
            "hand_landmarks": [[x, y, z], ...] or null,
            "face_landmarks": [[x, y, z], ...] or null
        }

    Sends JSON:
        {
            "gesture": "peace",
            "filter_type": "web_slinger",
            "raw_gesture": "peace",
            "placement": {
                "filter_type": "web_slinger",
                "center_x": 0.5,
                "center_y": 0.3,
                "width": 0.4,
                "height": 0.2
            } or null
        }
    """
    state = GestureFilterState(config)

    while True:
        data = ws.receive()
        if data is None:
            break

        try:
            msg = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            continue

        # Parse hand landmarks: list of [x, y, z] tuples or null
        raw_hand = msg.get("hand_landmarks")
        hand_landmarks = None
        if raw_hand and isinstance(raw_hand, list):
            hand_landmarks = [(lm[0], lm[1], lm[2]) for lm in raw_hand if len(lm) >= 3]

        # Parse face landmarks: list of [x, y, z] tuples or null
        raw_face = msg.get("face_landmarks")
        face_landmarks = None
        if raw_face and isinstance(raw_face, list):
            face_landmarks = [(lm[0], lm[1], lm[2]) for lm in raw_face if len(lm) >= 3]

        timestamp = time.time()
        result = state.update(hand_landmarks, face_landmarks, timestamp)

        # Serialize the response
        placement = result["placement"]
        placement_dict = None
        if placement is not None and isinstance(placement, FilterPlacement):
            placement_dict = {
                "filter_type": placement.filter_type.value,
                "center_x": placement.center_x,
                "center_y": placement.center_y,
                "width": placement.width,
                "height": placement.height,
            }

        response = {
            "gesture": result["gesture"].value if isinstance(result["gesture"], Gesture) else str(result["gesture"]),
            "filter_type": result["filter_type"].value if isinstance(result["filter_type"], FilterType) else str(result["filter_type"]),
            "raw_gesture": result["raw_gesture"].value if isinstance(result["raw_gesture"], Gesture) else str(result["raw_gesture"]),
            "placement": placement_dict,
        }

        ws.send(json.dumps(response))


if __name__ == "__main__":
    app.run(host=config.flask_host, port=config.flask_port)
