"""
iot_alert.py
------------
IoT Alert Module for Elephant Detection Traffic System.
Simulates sending alerts to physical traffic light hardware via HTTP/MQTT.
Replace the placeholder URLs with your actual IoT device endpoints.
"""

import requests
import time
import json


# ─────────────────────────────────────────────
#  Configuration – replace with real values
# ─────────────────────────────────────────────
IOT_DEVICE_URL = "http://YOUR_IOT_DEVICE_IP/api/traffic"   # e.g. ESP32 / Raspberry Pi
MQTT_BROKER    = "mqtt://YOUR_BROKER_IP"                    # optional MQTT broker
API_KEY        = "YOUR_IOT_API_KEY"                         # optional auth token

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}


# ─────────────────────────────────────────────
#  Traffic-light colour codes
# ─────────────────────────────────────────────
LIGHT_CODES = {
    "GREEN":  {"color": "GREEN",  "hex": "#00C851", "signal": 0},
    "YELLOW": {"color": "YELLOW", "hex": "#FFD700", "signal": 1},
    "RED":    {"color": "RED",    "hex": "#FF4444", "signal": 2},
}


def send_traffic_light_command(status: str, distance: float, location: str = "Junction-A") -> dict:
    """
    Send a traffic-light command to the IoT device.

    Parameters
    ----------
    status   : "GREEN" | "YELLOW" | "RED"
    distance : estimated elephant distance in metres
    location : identifier of the junction / road segment

    Returns
    -------
    dict with the payload sent and the HTTP response (or simulation result)
    """
    payload = {
        "location":   location,
        "status":     status,
        "signal":     LIGHT_CODES[status]["signal"],
        "distance_m": round(distance, 2),
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "alert":      status in ("RED", "YELLOW"),
        "message":    _build_message(status, distance),
    }

    result = {"payload": payload, "response": None, "simulated": False}

    try:
        response = requests.post(
            IOT_DEVICE_URL,
            headers=HEADERS,
            json=payload,
            timeout=3,
        )
        result["response"] = {
            "status_code": response.status_code,
            "body": response.text,
        }
    except Exception as exc:
        # Device not reachable → fall back to simulation mode
        result["simulated"] = True
        result["response"]  = {"status_code": 0, "body": f"Simulated (no device): {exc}"}

    return result


def get_alert_sound_path(status: str) -> str:
    """Return path to an alert sound file based on status (optional feature)."""
    sounds = {
        "RED":    "assets/alert_red.wav",
        "YELLOW": "assets/alert_yellow.wav",
        "GREEN":  "assets/clear.wav",
    }
    return sounds.get(status, "")


def _build_message(status: str, distance: float) -> str:
    if status == "RED":
        return f"🚨 DANGER: Elephant is only {distance:.1f} m away! Traffic STOPPED."
    if status == "YELLOW":
        return f"⚠️ WARNING: Elephant detected at {distance:.1f} m. Slow down!"
    return f"✅ Road clear. Nearest elephant > {distance:.1f} m."


# ─────────────────────────────────────────────
#  Quick self-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    for s, d in [("GREEN", 60), ("YELLOW", 35), ("RED", 15)]:
        r = send_traffic_light_command(s, d)
        print(json.dumps(r, indent=2))
