<div align="center">

# 🛡️ VMD Shield-AI

### Real-Time AI Surveillance with 3D Spatial Intelligence

**VMD Shield-AI** is a computer vision system that turns any ordinary RGB camera into a smart surveillance node — detecting violence, falls, intrusions, loitering, and abandoned objects in real time, with no LiDAR or depth hardware required.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Proprietary-red?style=flat-square)](#-licensing)

</div>

---

## 📺 Demo

**Real-time VMD Pipeline** — Pose Estimation (left) + Monocular Depth Map (right)

> _Video not playing? Watch the full demo **[on YouTube →](#)**_

---

## 💡 How It Works — The VMD Pipeline

The system runs three AI models in sync on every frame. Together they give the software a sense of **who is where, how fast they're moving, and how far apart they are in 3D space** — using just a regular camera.

| Layer | Model | What It Does |
|---|---|---|
| 🎥 **Video** | YOLOv11 Pose | Detects people, tracks them across frames, maps body keypoints |
| 🌊 **Motion** | Farneback Optical Flow | Measures movement speed and direction — catches sudden chaos |
| 📐 **Depth** | MiDaS DPT Transformer | Estimates how far each pixel is from the camera without depth hardware |

These three streams are fused per-frame. For example, two people standing close in 2D pixel space might actually be on different floors — the depth layer filters that out, reducing false alerts dramatically.

---

## ✨ Key Features

**🔒 Automatic Face Blurring**
Every detected face is blurred in real time using pose keypoints. The system never stores or displays identifiable faces — behavioral analysis continues on the anonymized feed.

**📦 No Special Hardware Needed**
Depth is estimated from a standard RGB camera using a neural model. No LiDAR, no ToF sensor, no stereo rig — any IP camera or webcam works.

**🖥️ Multi-Camera GUI Dashboard**
A PyQt6 desktop app lets you add cameras by RTSP URL, monitor all feeds in a live grid, and review the full alert history — all from one interface.

**💾 Evidence Clips Saved Automatically**
When an alert is confirmed, the system saves a 10-second MP4 clip (including pre-event footage) to an `evidence/` folder for review.

**📊 SQLite Logging**
Every alert is stored in a local SQLite database with camera name, alert type, severity, and timestamp. No external database or cloud required.

---

## 🗂️ Project Structure

```
vmd-shield-ai/
│
├── main.py                   # Headless entry point (CLI mode)
├── gui_app.py                # Desktop GUI dashboard (PyQt6)
├── requirements.txt          # Python dependencies
├── .gitignore
│
├── vision_engine.py          # YOLOv11 — person detection & tracking
├── motion_engine.py          # Optical flow — movement analysis
├── depth_engine.py           # MiDaS — monocular depth estimation
│
├── behavior_engine.py        # Wires all detection modules together
├── behavior_modules.py       # Individual alert detectors (see below)
├── behavior_accumulator.py   # Debounce layer — avoids one-frame false alerts
│
├── audio_engine.py           # YAMNet — sound classification (screams, gunshots)
├── evidence_manager.py       # Saves pre/post-event video clips
├── utils.py                  # Drawing skeletons, overlays, face blur
│
└── sentinel.db               # Auto-created SQLite database (gitignored)
```

---

## ⚙️ Installation

### Requirements

- Python **3.10 or newer**
- A GPU is strongly recommended — NVIDIA (CUDA) or Apple Silicon (MPS)
- CPU-only works but will be slow for live streams

---

### Step 1 — Clone the repo

```bash
git clone https://github.com/your-username/vmd-shield-ai.git
cd vmd-shield-ai
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3 — Install PyTorch for your hardware

> Do this **before** `requirements.txt` — PyTorch needs the right build for your system.

**NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Apple Silicon (M1 / M2 / M3):**
```bash
pip install torch torchvision
```

**CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 4 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Download model weights

Weights are never committed to the repo. Run these once to pre-download them:

```bash
# YOLOv11 pose model
python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"

# MiDaS depth model
python -c "import torch; torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)"
```

> YAMNet (audio) weights download automatically via TensorFlow Hub on first run — no action needed.

---

## ▶️ Running the System

### Option A — GUI Dashboard _(recommended)_

```bash
python gui_app.py
```

Opens a full desktop interface with:
- **Live View** — all camera feeds in a grid with overlaid detections
- **Cameras** — add or remove cameras by RTSP URL
- **Event Log** — searchable table of every alert that fired

### Option B — Headless / CLI Mode

```bash
python main.py
```

Before running, set your video source at the top of `main.py`:

```python
video_path = 'rtsp://admin:password@192.168.1.100:554/stream1'  # IP camera
video_path = 0             # Webcam
video_path = 'clip.mp4'   # Local video file
```

**Keyboard shortcuts while running:**

| Key | Action |
|---|---|
| `Q` | Quit |
| `D` | Toggle depth map overlay |

---

## 📡 Adding a Camera (GUI)

1. Open the **Cameras** page from the sidebar
2. Click **＋ Add Camera**
3. Fill in the form:

| Field | Example |
|---|---|
| Camera Name | `Parking Lot North` |
| RTSP URL | `rtsp://admin:pass@192.168.1.55:554/stream` |
| Location | `Site B, Ground Floor` |
| Zone Type | `Parking` |

4. Click **Add Camera** — the feed starts immediately in **Live View**

All camera data is saved to `sentinel.db` and reloaded automatically on the next launch.

---

## 🚨 Detection Modules

| Module | What Triggers It | Severity |
|---|---|---|
| **Violence Detector** | Two people in close proximity with high chaotic motion on the same depth plane | 🔴 Alert |
| **Fall Detector** | Person becomes horizontal or head drops to ankle level for more than 1.5 seconds | 🔴 Critical |
| **Intrusion Detector** | Any person crosses a virtual tripwire drawn across the frame | 🔴 Alert |
| **Loitering Detector** | Person stays in frame longer than the dwell time limit (default: 5 minutes) | 🟡 Suspicious |
| **Abandoned Object Detector** | An object stays unattended for more than 15 seconds with no nearby person on the same depth plane | 🟡 Warning |

### Adjusting sensitivity

All thresholds are configured in `behavior_engine.py`:

```python
ViolenceDetector(
    motion_thresh=45,    # Lower = more sensitive to movement
    dist_thresh=280,     # Pixel distance to consider people "close"
    depth_thresh=100     # Depth difference to consider "same plane"
)
LoiteringDetector(limit_sec=300)           # 5 minutes
IntrusionDetector(tripwire_y_ratio=0.7)    # Tripwire at 70% down the frame
AbandonedObjectDetector(limit_sec=15, dist_thresh=120)
FallDetector(time_to_confirm=1.5)          # Seconds sustained before alert fires
```

---

## 📼 Evidence Clips

An alert must fire **3 or more times within 10 seconds** before a clip is saved — this prevents single-frame noise from filling your disk.

Clips are saved to the `evidence/` folder with timestamps:

```
evidence/
├── ALERT_PHYSICAL_ALTERCATION_20250601_142310.mp4
├── CRITICAL_FALL_DETECTED_20250601_143005.mp4
└── WARNING_UNATTENDED_OBJECT_20250601_150122.mp4
```

Each clip includes **10 seconds of pre-event footage** from a rolling buffer, so you always see what led up to the incident.

Configure buffer size in `main.py`:

```python
evidence_mgr = VideoEvidenceManager(
    buffer_sec=10,       # Seconds of pre-event footage to retain
    fps=20,
    output_dir="evidence"
)
```

---

## 🗃️ Database Schema

The SQLite database (`sentinel.db`) is created automatically and has two tables:

```sql
-- Stores all registered cameras
CREATE TABLE cameras (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL,
    rtsp_url  TEXT NOT NULL,
    location  TEXT NOT NULL,
    zone      TEXT DEFAULT 'General',
    active    INTEGER DEFAULT 1,
    added_at  TEXT DEFAULT (datetime('now'))
);

-- Stores every alert event
CREATE TABLE events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   INTEGER,
    camera_name TEXT,
    alert_text  TEXT,
    severity    TEXT DEFAULT 'ALERT',
    timestamp   TEXT DEFAULT (datetime('now'))
);
```

Query it directly anytime:

```bash
# See last 20 alerts
sqlite3 sentinel.db "SELECT timestamp, camera_name, alert_text FROM events ORDER BY id DESC LIMIT 20;"

# See all registered cameras
sqlite3 sentinel.db "SELECT id, name, location, rtsp_url FROM cameras;"
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| Object Detection & Pose | Ultralytics YOLOv11 + ByteTrack |
| Depth Estimation | MiDaS DPT (via PyTorch Hub) |
| Motion Analysis | OpenCV Farneback Optical Flow |
| Audio Classification | TensorFlow + YAMNet |
| GUI | PyQt6 |
| Database | SQLite3 |
| Evidence Recording | OpenCV VideoWriter |

---

## 🤝 Collaboration & Licensing

VMD Shield-AI is **proprietary and closed-source**, currently transitioning from a research prototype to a production-grade enterprise product.

Open to partnerships in:

- **Edge deployment** — NVIDIA Jetson (TensorRT), Google Coral TPU
- **Custom model training** — domain-specific behavioral classifiers
- **Enterprise integration** — connecting to existing VMS / PSIM platforms

To discuss collaboration, open an issue or reach out directly.

---

## ⚠️ Legal & Ethical Use

This software is intended **only for authorized security and safety monitoring**. Operators are responsible for complying with all applicable laws including GDPR, CCPA, and local surveillance regulations.

Face blurring is **enabled by default** and should remain active in any public-facing deployment.

---

<div align="center">

*VMD Shield-AI — Turning passive cameras into active intelligence.*

</div>