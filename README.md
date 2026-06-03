# Driver Alertness Detection System

A real-time driver monitoring system that detects drowsiness and distraction through a webcam. The system runs in the browser — open the dashboard, grant camera access, and detection starts immediately.

**Custom Dataset:** The models in this project were trained on a custom, curated dataset of 60,000+ images which I have open-sourced and published on Kaggle.

[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/de080105/driver-drowsiness-detection-dataset)

---

## What it does

- Monitors the driver's face in real-time using a standard webcam
- Detects three signs of impairment: **eye closure**, **yawning**, and **head turning away**
- Combines all three into a single drowsiness score and assigns one of three states: `ALERT`, `DROWSY`, or `DISTRACTED`
- Plays an alarm and displays a warning when the driver is not alert
- Logs all events with timestamps in a live session dashboard

---

## How detection works

Each webcam frame goes through three independent pipelines:

**Eye state**
- MediaPipe Face Mesh crops the left and right eye regions from the frame
- Each crop is grayscale-converted, stacked into a 96x96 RGB image, and classified by a MobileNetV2 model
- The two eye predictions are averaged — below 0.5 means eyes are closed

**Yawn detection**
- The mouth region is cropped using MediaPipe landmark indices
- Fed into a second MobileNetV2 model trained on labeled yawning images

**Head pose**
- Six stable landmarks (nose tip, eye corners, mouth corners, chin) are passed to OpenCV's `solvePnP`
- This produces yaw and pitch angles via Rodrigues decomposition
- Angles are smoothed with an exponential moving average to reduce jitter at low frame rates
- If yaw or pitch exceeds the threshold, the driver is marked as looking away

**Alert logic**
- The system uses consecutive frame counters, not single-frame decisions — a driver must be in a bad state for multiple frames before the level changes
- Drowsiness score = `(closed_eye_frames × 8) + (yawn_frames × 2)`
- Score above 24 → `DROWSY`; sustained head turning → `DISTRACTED`

---

## Why WebSocket instead of HTTP

- Each frame is sent over a persistent WebSocket connection (`/ws`) instead of a new HTTP request
- Eliminates repeated connection handshake overhead at every frame
- The browser sends a frame, waits for the result, then sends the next — this naturally rate-limits to the server's inference speed (~2 fps on CPU)

---

## Project structure

```
├── Models/                              Pre-trained .h5 weights (eye, yawn, head models)
├── src/
│   ├── 1_Custom_CNN/                    Custom CNN training scripts and webcam detection
│   └── 2_Transfer_Learning_MobileNetV2/ MobileNetV2 training and comparison scripts
├── tests/                               Experimental and utility scripts from development
├── docs/
│   ├── evaluation_Metrics/              Confusion matrices and ROC curves
│   └── flowcharts/                      Architecture diagrams
├── web_deployment/
│   ├── Backend/
│   │   ├── main.py                      FastAPI app — registers routes, serves frontend
│   │   ├── models.py                    Loads .h5 models once at server startup
│   │   ├── inference.py                 Full inference pipeline (pure function, no global state)
│   │   ├── state.py                     Per-connection SessionState class + session registry
│   │   └── routes/
│   │       ├── http_routes.py           REST endpoints: /history, /events, /reset, /health
│   │       └── ws_routes.py             WebSocket endpoint: /ws
│   └── Frontend/
│       ├── index.html
│       ├── style.css
│       └── js/
│           ├── main.js                  Entry point — session lifecycle, alarm
│           ├── websocket.js             WS connection manager with auto-reconnect
│           ├── camera.js               getUserMedia and frame capture
│           ├── ui.js                   All DOM updates
│           └── chart.js               Chart.js setup and updates
├── Dockerfile
└── requirements.txt
```

---

## Models

| Model | Architecture | Training Data | Task |
|-------|-------------|---------------|------|
| `eye_model_mobilenet_tuned.h5` | MobileNetV2 (fine-tuned) | ~54,000 eye images | Open / Closed |
| `yawn_model_mobilenet_tuned.h5` | MobileNetV2 (fine-tuned) | ~6,000 mouth images | Yawn / No Yawn |

- The `src/` folder also contains custom CNN versions and frozen MobileNetV2 variants from earlier experiments
- The fine-tuned MobileNetV2 models gave the best validation accuracy and are used in the web deployment

**Dataset on Kaggle:** [Driver Alertness Combined Dataset](https://www.kaggle.com/datasets/de080105/driver-drowsiness-detection-dataset)

---

## Getting started

**Requirements:** Python 3.10+, a webcam

**1. Clone the repo**
```bash
git clone https://github.com/deepakk789/driver-alertness-detection.git
cd driver-alertness-detection
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
```

**3. Install dependencies**
```bash
pip install -r web_deployment/Backend/requirements.txt
```

**4. Start the server**
```bash
cd web_deployment/Backend
uvicorn main:app --reload --port 8000
```

**5. Open the dashboard**

Go to `http://localhost:8000`, click **Start Detection**, and allow camera access.

---

## Run local webcam detection (no server required)

Want to test the models directly on your machine without running the web server? Two standalone scripts are available that run detection using your system webcam and display the output in a pop-up OpenCV window.

**Requirements:** Python 3.10+, a webcam, all packages from `requirements.txt`

**Option 1 — Custom CNN** (trained from scratch)
```bash
python src/1_Custom_CNN/webcam_detection_CustomCNN.py
```
- Uses `eye_model.h5` and `yawn_model.h5`
- Lighter model, faster on low-end hardware

**Option 2 — MobileNetV2** (transfer learning, higher accuracy)
```bash
python src/2_Transfer_Learning_MobileNetV2/webcam_detection_MobileNetV2.py
```
- Uses `eye_model_mobilenet_tuned.h5` and `yawn_model_mobilenet_tuned.h5`
- Same model used in the live web deployment
- Better accuracy, especially for eye detection

> **Note:** Run both scripts from the **project root directory**, not from inside the `src/` folder. Press `ESC` to exit the webcam window.

---

## Multi-user support

- Each browser tab or device that connects receives an isolated session
- The server assigns a UUID to every WebSocket connection — all counters and history are tracked separately per session
- Sessions are cleaned up automatically on disconnect
- Active sessions can be viewed at `/sessions`; server status at `/health`

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `WS` | `/ws` | Main inference — receives base64 frames, returns alertness results |
| `GET` | `/history` | Last 120 alertness scores for the live chart |
| `GET` | `/events` | All distraction and drowsiness events in the session |
| `POST` | `/reset` | Reset counters and history for a session |
| `GET` | `/health` | Confirms server is up and models are loaded |
| `GET` | `/sessions` | Lists all active session IDs |

All data endpoints accept an optional `?session_id=` query parameter. If only one session is active, it can be omitted.

Interactive API docs: `http://localhost:8000/docs`

---

## Docker

```bash
docker build -t driver-alertness .
docker run -p 8000:8000 driver-alertness
```

---

## Notes

- Inference runs synchronously on CPU. At 2 fps this is sufficient. For higher throughput, inference can be offloaded to a thread pool using `asyncio.run_in_executor`
- The alarm requires at least one browser interaction before audio can autoplay — clicking Start satisfies this
- MediaPipe returns 468 face landmarks per frame; only a targeted subset is used for cropping and head pose estimation
