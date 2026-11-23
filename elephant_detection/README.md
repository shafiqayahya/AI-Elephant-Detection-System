# 🐘 AI Elephant Detection — Smart Traffic Light System

> Real-time elephant detection dashboard using **Roboflow** + **Streamlit**  
> Triggers IoT traffic light signals based on estimated distance

---

## 🗺️ System Overview

```
📷 Webcam  →  🤖 Roboflow Inference  →  📏 Distance Estimation
                                                ↓
                             > 40 m  →  🟢 GREEN  — Road clear
                           20–40 m  →  🟡 YELLOW — Elephant approaching
                             < 20 m  →  🔴 RED    — Elephant crossing / STOP
                                                ↓
                                       📡 IoT HTTP Alert
                                      (ESP32 / Raspberry Pi)
```

---

## 📁 File Structure

```
elephant_detection/
├── app.py                          # Main Streamlit dashboard
├── iot_alert.py                    # IoT alert helper (HTTP + simulation)
├── requirements.txt                # Python dependencies
├── elephant_detection_colab.ipynb  # Google Colab notebook
└── README.md                       # This file
```

---

## ⚙️ Setup Instructions

### 1 — Get a Roboflow API Key
1. Sign up at [roboflow.com](https://roboflow.com)
2. Search **Roboflow Universe** for an elephant detection model  
   (e.g. search *"elephant"* → fork a project)
3. Train or download a model version
4. Go to **Settings → API Key** and copy your key

### 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### 3 — Run the Streamlit Dashboard
```bash
streamlit run app.py
```

### 4 — Configure in the Sidebar
| Field | Description |
|---|---|
| API Key | Your Roboflow API key |
| Workspace ID | Your Roboflow workspace slug |
| Project ID | Your Roboflow project slug |
| Model Version | Version number (usually 1) |
| Webcam Index | 0 = default camera |
| Yellow Alert | Distance threshold for yellow light (default 40 m) |
| Red Alert | Distance threshold for red light (default 20 m) |
| IoT Endpoint | URL of your IoT device (optional) |

---

## 📡 IoT Integration

The system sends HTTP POST requests to your IoT device with this JSON payload:

```json
{
  "location":   "Junction-A",
  "status":     "RED",
  "signal":     2,
  "distance_m": 15.3,
  "timestamp":  "2026-04-28T14:30:00Z",
  "alert":      true,
  "message":    "DANGER: Elephant 15.3 m away!"
}
```

**Signal codes:**
- `0` = GREEN
- `1` = YELLOW  
- `2` = RED

### Sample Arduino/ESP32 code snippet
```cpp
// In your ESP32 HTTP server handler:
if (doc["signal"] == 2) {
  digitalWrite(RED_PIN, HIGH);
  digitalWrite(YELLOW_PIN, LOW);
  digitalWrite(GREEN_PIN, LOW);
} else if (doc["signal"] == 1) {
  digitalWrite(RED_PIN, LOW);
  digitalWrite(YELLOW_PIN, HIGH);
  digitalWrite(GREEN_PIN, LOW);
} else {
  digitalWrite(RED_PIN, LOW);
  digitalWrite(YELLOW_PIN, LOW);
  digitalWrite(GREEN_PIN, HIGH);
}
```

---

## 📏 Distance Estimation

Uses the **pinhole camera model**:

```
distance = (KNOWN_ELEPHANT_HEIGHT × FOCAL_LENGTH) / BBOX_HEIGHT_PIXELS
```

**Default values:**
- Elephant height: **2.8 m** (average)
- Focal length: **700 px** (calibrate to your camera)

### How to calibrate focal length:
1. Stand at a **known distance D** from an elephant (or object of known height)
2. Measure its bounding box height **P** pixels
3. Calculate: `FOCAL_LENGTH = (P × D) / KNOWN_HEIGHT`

---

## 🖥️ Google Colab

Use `elephant_detection_colab.ipynb` to:
- Run inference on uploaded images
- Process video files frame-by-frame
- Launch the Streamlit dashboard via **ngrok** tunnel

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| Camera not found | Change webcam index in sidebar |
| Inference too slow | Increase `infer_every` frames |
| Wrong distance | Calibrate focal length |
| IoT not responding | Check endpoint URL & device IP |
| Low accuracy | Retrain with more elephant images |

---

## 📚 Technologies Used

| Tool | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Dashboard UI |
| [Roboflow](https://roboflow.com) | Elephant detection model |
| [OpenCV](https://opencv.org) | Camera capture & image processing |
| [inference-sdk](https://github.com/roboflow/inference) | Roboflow inference client |
| ESP32 / Raspberry Pi | IoT traffic light hardware |
| ngrok | Colab tunnel |
