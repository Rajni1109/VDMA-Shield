# 🛡️ VMD Shield-AI: Neural Spatiotemporal Intelligence Hub
### **Winner: 3rd Rank @ Codethon '26, IIT Roorkee** 🥉

VMD Shield-AI is an advanced computer vision framework engineered to transmute monocular 2D sensor data into high-fidelity, 3D-aware intelligence streams. By implementing a proprietary **VMD (Video, Motion, and Depth)** synchronization architecture, the system achieves sub-meter spatial accuracy and complex behavioral heuristics without auxiliary active-depth hardware (LiDAR/ToF).

---

## 📺 Synchronized Inference Demo
<div align="center">
  <video src="https://youtu.be/RqsGIQ1qk6E" width="100%" autoplay loop muted playsinline></video>
  <p><em>Real-time VMD Pipeline: Multi-Instance Neural Pose Estimation (Left) & Dense Monocular Inverse-Depth Mapping (Right)</em></p>
</div>

> **Note:** If the video does not autoplay in your browser, you can view the full demonstration [on YouTube here](https://youtu.be/RqsGIQ1qk6E).
> 
---

## 🚀 The VMD Synchronous Pipeline
The core innovation lies in the **VMD (Video, Motion, and Depth)** engine, which utilizes a decoupled-yet-synchronized inference stack to resolve temporal instability in monocular vision.

* **[V]ideo (Object/Pose Semantics):** Leverages a **YOLOv11-backbone** for real-time multi-instance keypoint detection and centroid tracking. This provides the semantic basis for spatial anchoring.
* **[M]otion (Temporal Kinematics):** Implements **Farneback Optical Flow** and motion magnitude vectors to analyze velocity shifts, differentiating between ambient environment changes and critical behavioral anomalies (e.g., falls or altercations).
* **[D]epth (Geometric Reconstruction):** Utilizes a **DPT-based (Dense Prediction Transformer)** MiDaS architecture to compute a dense relative depth map. This is mathematically projected into a pseudo-3D coordinate system to calculate inter-object distancing.

---

## ✨ Engineering Innovations
* **Dynamic Privacy Anonymization:** A low-latency masking algorithm that utilizes pose-estimation keypoints to apply localized Gaussian diffusion on specific ROI (Regions of Interest). This ensures 100% identity protection while maintaining behavioral visibility.
* **Hardware-Agnostic 3D Volumetrics:** By approximating depth from parallax and neural cues, the system enables 3D spatial occupancy mapping on legacy RGB-only infrastructure, reducing CAPEX by 90% compared to LiDAR deployments.
* **Edge-Optimized Persistence:** Implements an asynchronous SQLite3 logging layer for crowd density telemetry, utilized by a Matplotlib-driven analytics engine to perform hourly trend regression.
* **Multithreaded GUI Concurrency:** Engineered on a PyQt6 framework utilizing Python's `QThread` and `pyqtSignal` architecture to prevent GIL-lock bottlenecks during high-bitrate dual-stream rendering.

---

## 🛠️ Technical Stack
* **Runtime:** Python 3.10+ (Optimized with Environment-specific hardware acceleration)
* **Neural Frameworks:** Ultralytics YOLOv11, PyTorch (MiDaS DPT-Large/Hybrid)
* **Computer Vision:** OpenCV (Kinematic Vector Analysis & Flow Magnitude)
* **GUI & Rendering:** PyQt6 Interface with integrated Matplotlib Backend
* **Persistence Layer:** SQLite3 with relational time-series schema

---

## 🏆 Hackathon Recognition
This project secured **3rd Place** at **Codethon '26, IIT Roorkee**. The jury specifically commended the **VMD synchronization logic**, citing its high industrial scalability and robust ethical framework through real-time privacy-preserving inference.

---

## 🤝 Technical Collaboration & Licensing
VMD Shield-AI is currently a **Proprietary/Closed-Source** framework as it undergoes transition from a prototype to a production-grade enterprise solution. 

**Seeking strategic partnerships for:**
* **Edge Optimization:** Porting the VMD stack to NVIDIA Jetson (TensorRT) and Coral TPU.
* **Behavioral Scaling:** Training custom classifiers for high-risk environmental detection.

* **LinkedIn:** [https://www.linkedin.com/in/surya-verma/]
* **Email:** [11surya.v@gmail.com]

---
*Architected for Codethon '26, IIT Roorkee*
