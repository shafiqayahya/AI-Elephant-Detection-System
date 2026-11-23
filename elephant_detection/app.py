"""
app.py  –  Elephant Detection Dashboard
========================================
Streamlit dashboard for real-time elephant detection via webcam.
Uses Roboflow for inference and estimates distance to trigger a
colour-coded IoT traffic-light alert.

  Distance  >  40 m  →  🟢  GREEN  – road clear
  Distance  20–40 m  →  🟡  YELLOW – elephant approaching
  Distance  <  20 m  →  🔴  RED    – elephant crossing / stop!

Run locally:
    streamlit run app.py

Run in Google Colab:
    See elephant_detection_colab.ipynb
"""

import time
import json
import math
import threading
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from inference_sdk import InferenceHTTPClient   # pip install inference-sdk

from iot_alert import send_traffic_light_command, LIGHT_CODES

# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🐘 Elephant Detection System",
    page_icon="🐘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── General ── */
body { background-color: #0e1117; }
.block-container { padding-top: 1rem; }

/* ── Traffic Light Box ── */
.traffic-box {
    border-radius: 16px;
    padding: 24px 16px;
    text-align: center;
    font-size: 3.5rem;
    font-weight: 900;
    letter-spacing: 2px;
    box-shadow: 0 0 30px rgba(0,0,0,0.5);
    transition: all 0.4s ease;
}
.traffic-green  { background: #1a3a1a; border: 4px solid #00C851; color: #00C851; }
.traffic-yellow { background: #3a3a00; border: 4px solid #FFD700; color: #FFD700; }
.traffic-red    { background: #3a0000; border: 4px solid #FF4444; color: #FF4444; }

/* ── Metric card ── */
.metric-card {
    background: #1e2130;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin: 4px 0;
}
.metric-val { font-size: 2rem; font-weight: 700; color: #e0e0e0; }
.metric-lbl { font-size: 0.8rem; color: #888; text-transform: uppercase; }

/* ── Alert banner ── */
.alert-red    { background:#FF4444; color:white; padding:12px; border-radius:10px; font-weight:700; text-align:center; }
.alert-yellow { background:#FFD700; color:black; padding:12px; border-radius:10px; font-weight:700; text-align:center; }
.alert-green  { background:#00C851; color:white; padding:12px; border-radius:10px; font-weight:700; text-align:center; }

/* ── Log table ── */
.log-entry { font-size: 0.78rem; font-family: monospace; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS – Distance Estimation
# ═══════════════════════════════════════════════════════════════
KNOWN_ELEPHANT_HEIGHT_M = 2.8          # average African elephant ~ 2.8 m tall
FOCAL_LENGTH_PX         = 700          # calibrate to your camera; 700 is a safe default
                                       # formula: F = (P * D) / H  where P=pixels, D=known_dist, H=known_height
YELLOW_THRESHOLD_M      = 40.0         # metres – turn yellow
RED_THRESHOLD_M         = 20.0         # metres – turn red / stop


# ═══════════════════════════════════════════════════════════════
#  SESSION STATE DEFAULTS
# ═══════════════════════════════════════════════════════════════
defaults = {
    "detection_log":     [],
    "iot_log":           [],
    "last_status":       "GREEN",
    "last_distance":     float("inf"),
    "total_detections":  0,
    "red_alerts":        0,
    "yellow_alerts":     0,
    "running":           False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def estimate_distance(bbox_height_px: float, frame_height_px: int) -> float:
    """
    Estimate real-world distance (metres) from bounding-box height.
    Uses the pinhole camera model:
        distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / bbox_height_pixels
    Falls back to a frame-fraction heuristic when bbox_height_px is 0.
    """
    if bbox_height_px <= 0:
        return 999.0
    distance_m = (KNOWN_ELEPHANT_HEIGHT_M * FOCAL_LENGTH_PX) / bbox_height_px
    return round(distance_m, 2)


def get_traffic_status(distance_m: float):
    """Return (status_str, css_class, emoji, message) for a given distance."""
    if distance_m <= RED_THRESHOLD_M:
        return ("RED",    "traffic-red",    "🔴",
                f"🚨 DANGER — Elephant only {distance_m:.1f} m away! TRAFFIC STOPPED.")
    if distance_m <= YELLOW_THRESHOLD_M:
        return ("YELLOW", "traffic-yellow", "🟡",
                f"⚠️ WARNING — Elephant at {distance_m:.1f} m. Slow down!")
    return ("GREEN",  "traffic-green",  "🟢",
            f"✅ Road clear. Nearest elephant {distance_m:.1f} m away.")


def draw_boxes(frame: np.ndarray, predictions: list) -> tuple[np.ndarray, float]:
    """Draw bounding boxes on the frame; return annotated frame + closest distance."""
    closest = float("inf")
    h_frame  = frame.shape[0]

    for pred in predictions:
        x      = int(pred["x"] - pred["width"]  / 2)
        y      = int(pred["y"] - pred["height"] / 2)
        w      = int(pred["width"])
        h      = int(pred["height"])
        conf   = pred.get("confidence", 0)
        label  = pred.get("class", "elephant")

        dist   = estimate_distance(h, h_frame)
        if dist < closest:
            closest = dist

        # Choose box colour based on distance
        status, _, _, _ = get_traffic_status(dist)
        colour = {"GREEN": (0, 200, 80), "YELLOW": (0, 200, 255), "RED": (60, 60, 255)}[status]

        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 3)

        label_text = f"{label} | {conf:.0%} | {dist:.1f} m"
        (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - lh - 12), (x + lw + 8, y), colour, -1)
        cv2.putText(frame, label_text, (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Distance ring overlay
        cx, cy = int(pred["x"]), int(pred["y"])
        cv2.circle(frame, (cx, cy), 6, colour, -1)

    return frame, closest


def add_log_entry(distance: float, status: str, confidence: float, count: int):
    """Append an entry to the detection log (capped at 50)."""
    entry = {
        "time":       datetime.now().strftime("%H:%M:%S"),
        "distance_m": f"{distance:.1f}",
        "status":     status,
        "confidence": f"{confidence:.0%}",
        "elephants":  count,
    }
    st.session_state.detection_log.insert(0, entry)
    st.session_state.detection_log = st.session_state.detection_log[:50]


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR – Configuration
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Roboflow_logo.png/320px-Roboflow_logo.png",
             width=160)
    st.title("⚙️ Configuration")

    st.subheader("🔑 Roboflow Settings")
    rf_api_key   = st.text_input("API Key",        type="password",
                                  placeholder="rf_xxxxxxxxxxxxxxxx")
    rf_workspace = st.text_input("Workspace ID",   placeholder="my-workspace")
    rf_project   = st.text_input("Project ID",     placeholder="elephant-detection")
    rf_version   = st.number_input("Model Version", min_value=1, value=1, step=1)

    st.markdown("---")
    st.subheader("📷 Camera")
    cam_index = st.number_input("Webcam Index (0 = default)", min_value=0, value=0, step=1)
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.45, 0.05)

    st.markdown("---")
    st.subheader("📏 Distance Thresholds")
    yellow_dist = st.slider("🟡 Yellow Alert (m)", 10, 100, 40, 5)
    red_dist    = st.slider("🔴 Red Alert (m)",    5,  50,  20, 5)

    st.markdown("---")
    st.subheader("🌐 IoT Device")
    iot_url     = st.text_input("IoT Endpoint URL",  placeholder="http://192.168.1.100/api/traffic")
    iot_enabled = st.toggle("Enable IoT Alerts", value=True)

    st.markdown("---")
    st.subheader("📊 Session Stats")
    st.metric("Total Detections",  st.session_state.total_detections)
    st.metric("🟡 Yellow Alerts",  st.session_state.yellow_alerts)
    st.metric("🔴 Red Alerts",     st.session_state.red_alerts)

    if st.button("🗑️ Reset Stats"):
        st.session_state.total_detections = 0
        st.session_state.yellow_alerts    = 0
        st.session_state.red_alerts       = 0
        st.session_state.detection_log    = []
        st.session_state.iot_log          = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════
st.title("🐘 AI Elephant Detection — Smart Traffic System")
st.caption("Powered by Roboflow · Real-time webcam · IoT traffic light integration")

# ── Row 1: Camera feed + Traffic light ──────────────────────────
col_cam, col_light = st.columns([3, 2], gap="large")

with col_cam:
    st.subheader("📷 Live Camera Feed")
    cam_placeholder   = st.empty()
    status_placeholder = st.empty()

with col_light:
    st.subheader("🚦 Traffic Light Status")
    light_placeholder = st.empty()
    dist_placeholder  = st.empty()
    iot_placeholder   = st.empty()

# ── Row 2: Metrics ──────────────────────────────────────────────
st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
m1_ph = m1.empty()
m2_ph = m2.empty()
m3_ph = m3.empty()
m4_ph = m4.empty()

# ── Row 3: Detection log + IoT log ──────────────────────────────
st.markdown("---")
log_col, iot_col = st.columns(2, gap="large")
with log_col:
    st.subheader("📋 Detection Log")
    log_placeholder = st.empty()
with iot_col:
    st.subheader("📡 IoT Alert Log")
    iot_log_placeholder = st.empty()

# ── Control buttons ─────────────────────────────────────────────
st.markdown("---")
btn_start, btn_stop, btn_snapshot = st.columns(3)

start_btn    = btn_start.button("▶️ Start Detection",   type="primary", use_container_width=True)
stop_btn     = btn_stop.button("⏹️ Stop Detection",                     use_container_width=True)
snapshot_btn = btn_snapshot.button("📸 Take Snapshot",                  use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  RENDER HELPERS  (called every frame)
# ═══════════════════════════════════════════════════════════════

def render_traffic_light(status: str, distance: float):
    _, css, emoji, msg = get_traffic_status(distance)
    light_placeholder.markdown(f"""
    <div class="traffic-box {css}">
        {emoji}<br/>{status}
    </div>
    """, unsafe_allow_html=True)

    alert_css = {"RED": "alert-red", "YELLOW": "alert-yellow", "GREEN": "alert-green"}[status]
    status_placeholder.markdown(f'<div class="{alert_css}">{msg}</div>', unsafe_allow_html=True)

    dist_txt = f"{distance:.1f} m" if distance < 999 else "—"
    dist_placeholder.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{dist_txt}</div>
        <div class="metric-lbl">Estimated Distance</div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(count: int):
    for ph, val, lbl in [
        (m1_ph, st.session_state.total_detections, "Total Detections"),
        (m2_ph, st.session_state.yellow_alerts,    "Yellow Alerts"),
        (m3_ph, st.session_state.red_alerts,       "Red Alerts"),
        (m4_ph, count,                             "Elephants in Frame"),
    ]:
        ph.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)


def render_logs():
    if st.session_state.detection_log:
        import pandas as pd
        df = pd.DataFrame(st.session_state.detection_log)
        log_placeholder.dataframe(df, use_container_width=True, height=250)
    else:
        log_placeholder.info("No detections yet.")

    if st.session_state.iot_log:
        iot_log_placeholder.json(st.session_state.iot_log[:5])
    else:
        iot_log_placeholder.info("No IoT alerts yet.")


# ═══════════════════════════════════════════════════════════════
#  DEFAULT (idle) display
# ═══════════════════════════════════════════════════════════════
render_traffic_light("GREEN", 999.0)
render_metrics(0)
render_logs()

cam_placeholder.info("⬅️ Configure Roboflow credentials in the sidebar, then press ▶️ Start Detection.")


# ═══════════════════════════════════════════════════════════════
#  DETECTION LOOP
# ═══════════════════════════════════════════════════════════════
if start_btn:
    if not rf_api_key or not rf_workspace or not rf_project:
        st.error("❌ Please fill in Roboflow API Key, Workspace ID, and Project ID in the sidebar.")
        st.stop()

    # Build Roboflow inference client
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=rf_api_key,
    )
    MODEL_ID = f"{rf_project}/{rf_version}"

    # Update runtime thresholds from sidebar
    global YELLOW_THRESHOLD_M, RED_THRESHOLD_M
    YELLOW_THRESHOLD_M = yellow_dist
    RED_THRESHOLD_M    = red_dist

    cap = cv2.VideoCapture(int(cam_index))
    if not cap.isOpened():
        st.error(f"❌ Cannot open webcam (index {cam_index}). Check connection.")
        st.stop()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_status   = "GREEN"
    frame_counter = 0
    infer_every   = 3          # run inference every N frames (performance)
    last_preds    = []
    last_closest  = 999.0

    stop_flag = st.session_state  # use session_state as a simple flag carrier

    while True:
        if stop_btn:
            break

        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Lost camera feed.")
            break

        frame_counter += 1
        run_inference = (frame_counter % infer_every == 0)

        if run_inference:
            try:
                # Convert to PIL for Roboflow SDK
                img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img  = Image.fromarray(img_rgb)

                result   = client.infer(pil_img, model_id=MODEL_ID)
                preds    = [p for p in result.get("predictions", [])
                            if p.get("confidence", 0) >= conf_threshold]
                last_preds = preds

                if preds:
                    frame_annotated, closest = draw_boxes(frame.copy(), preds)
                    last_closest = closest
                    avg_conf     = sum(p["confidence"] for p in preds) / len(preds)

                    status, _, _, _ = get_traffic_status(closest)

                    # Update counters
                    st.session_state.total_detections += len(preds)
                    if status == "RED"    and prev_status != "RED":
                        st.session_state.red_alerts += 1
                    if status == "YELLOW" and prev_status not in ("RED", "YELLOW"):
                        st.session_state.yellow_alerts += 1

                    # Log entry
                    add_log_entry(closest, status, avg_conf, len(preds))

                    # IoT alert (only on status change)
                    if status != prev_status and iot_enabled:
                        iot_result = send_traffic_light_command(
                            status, closest,
                            location=iot_url if iot_url else "Junction-A"
                        )
                        st.session_state.iot_log.insert(0, iot_result["payload"])
                        st.session_state.iot_log = st.session_state.iot_log[:10]

                    prev_status = status
                    frame = frame_annotated
                else:
                    last_closest = 999.0
                    status       = "GREEN"
                    if prev_status != "GREEN":
                        if iot_enabled:
                            send_traffic_light_command("GREEN", 999.0)
                        prev_status = "GREEN"

            except Exception as exc:
                cv2.putText(frame, f"Inference error: {exc}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Overlay: distance bar
        if last_closest < 999:
            bar_max   = 60
            bar_fill  = max(0, min(1, (bar_max - last_closest) / bar_max))
            bar_w     = int(frame.shape[1] * bar_fill)
            bar_color = (0, 200, 80)
            if last_closest <= YELLOW_THRESHOLD_M:
                bar_color = (0, 200, 255)
            if last_closest <= RED_THRESHOLD_M:
                bar_color = (50, 50, 255)
            cv2.rectangle(frame, (0, frame.shape[0] - 14), (bar_w, frame.shape[0]), bar_color, -1)
            cv2.putText(frame, f"Dist: {last_closest:.1f} m", (8, frame.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Overlay: timestamp
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Display frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update UI widgets
        render_traffic_light(prev_status, last_closest)
        render_metrics(len(last_preds))
        render_logs()

        # IoT placeholder
        if st.session_state.iot_log:
            iot_placeholder.success(f"📡 Last IoT signal: **{st.session_state.iot_log[0]['status']}**"
                                    f" at {st.session_state.iot_log[0]['timestamp']}")

        time.sleep(0.03)   # ~30 fps cap

    cap.release()
    st.info("Detection stopped.")


# ─── SNAPSHOT ───────────────────────────────────────────────────
if snapshot_btn:
    cap_snap = cv2.VideoCapture(int(cam_index))
    if cap_snap.isOpened():
        ret, frame = cap_snap.read()
        if ret:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"snapshot_{ts}.jpg"
            cv2.imwrite(path, frame)
            cam_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                  caption=f"Snapshot saved: {path}",
                                  use_container_width=True)
            st.success(f"📸 Snapshot saved → {path}")
        cap_snap.release()


if stop_btn:
    st.session_state.running = False
    st.warning("⏹️ Detection stopped by user.")
