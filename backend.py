"""Local API for connecting the TrafficMind dashboard to the user's RF model."""

from __future__ import annotations

import json
import os
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Timer
from urllib.parse import urlsplit

import joblib
import pandas as pd


DASHBOARD_DIR = Path(__file__).resolve().parent
ML_DIR = Path(os.environ.get("TRAFFIC_ML_DIR", r"D:\JU_tech"))
MODEL_FILE = ML_DIR / "traffic_rf_model.pkl"
HOST = "127.0.0.1"
PORT = 8765
FEATURE_COLUMNS = [
    "queue_length",
    "avg_speed_kmh",
    "occupancy_pct",
    "arrival_rate_veh_per_step",
    "approach_wait_sec",
    "lanes_open",
]


def load_ml_assets():
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"Could not find the trained model file: {MODEL_FILE}")
    return joblib.load(MODEL_FILE)


MODEL = load_ml_assets()


def estimate_observation(queue_length, arrival_rate_veh_per_step, approach_wait_sec, lanes_open):
    """Recreate the model's six input features from the simulated junction state."""
    return {
        "queue_length": queue_length,
        "avg_speed_kmh": max(5.0, min(60.0, 50.0 - 0.8 * queue_length)),
        "occupancy_pct": max(0.0, min(100.0, queue_length * 2.0)),
        "arrival_rate_veh_per_step": arrival_rate_veh_per_step,
        "approach_wait_sec": approach_wait_sec,
        "lanes_open": lanes_open,
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/health":
            self._send_json(
                200,
                {
                    "status": "ready",
                    "model": MODEL_FILE.name,
                    "features": FEATURE_COLUMNS,
                },
            )
            return
        self.path = path
        super().do_GET()

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/predict":
            self._send_json(404, {"error": "Unknown API endpoint."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length < 1 or content_length > 64_000:
                self._send_json(400, {"error": "Invalid request size."})
                return
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            observations = payload.get("observations")
            if not isinstance(observations, list) or not observations:
                self._send_json(400, {"error": "Provide one or more traffic observations."})
                return

            features = []
            for item in observations:
                features.append(
                    estimate_observation(
                        queue_length=float(item["queue_length"]),
                        arrival_rate_veh_per_step=float(item["arrival_rate_veh_per_step"]),
                        approach_wait_sec=float(item["approach_wait_sec"]),
                        lanes_open=int(item["lanes_open"]),
                    )
                )
            frame = pd.DataFrame(features)[FEATURE_COLUMNS]
            predictions = MODEL.predict(frame)
            self._send_json(
                200,
                {"green_seconds": [float(value) for value in predictions]},
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": f"Invalid traffic observation: {error}"})
        except Exception as error:  # Keep an inference error visible to the dashboard.
            self._send_json(500, {"error": f"Random Forest prediction failed: {error}"})


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print("TrafficMind dashboard and Random Forest API are ready.")
    print(f"Dashboard: http://{HOST}:{PORT}/traffic_dashboard.html")
    print(f"Random Forest: {MODEL_FILE}")
    print("Press Ctrl+C to stop the local server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping TrafficMind backend.")
    finally:
        server.server_close()


if __name__ == "__main__":
    Timer(1.0, lambda: webbrowser.open(f"http://{HOST}:{PORT}/traffic_dashboard.html")).start()
    main()
