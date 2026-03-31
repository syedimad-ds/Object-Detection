![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-FF6F00?logo=tensorflow&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue)
![WebGL](https://img.shields.io/badge/WebGL-Enabled-green)
![Status](https://img.shields.io/badge/Status-Active-success)

# 🚀 Real-Time Edge AI: Object Detection Web App  
**Powered by YOLOv8s & TensorFlow.js**

[![Live Demo](https://img.shields.io/badge/Demo-Live%20on%20GitHub%20Pages-brightgreen)](https://syedimad-ds.github.io/Object-Detection/)

A lightweight, privacy-focused web application that performs real-time multi-object detection directly within the browser. There is no server-side processing; the entire AI engine runs locally on the user's device (client-side).

---

## 🛠️ Tech Stack
- **AI Model:** Ultralytics YOLOv8s (Small Version)  
- **Framework:** TensorFlow.js (TF.js)  
- **Frontend:** HTML5, CSS3 (Modern Dark Theme), JavaScript (ES6+)  
- **Deployment:** GitHub Pages  

---

## 📂 Repository Structure

```text
Object-Detection/
│
├── index.html                  # Main UI layout (Video & Canvas container)
├── style.css                   # Premium Dark Theme & UI styling
├── script.js                   # Core AI Engine (Inference, WebGL NMS, Letterboxing)
├── Yolov8n_json.ipynb          # Jupyter Notebook for exporting Nano model to TF.js
├── Yolov8s_json.ipynb          # Jupyter Notebook for exporting Small model to TF.js
├── README.md                   # Project documentation
│
├── yolov8n_web_model/          # Exported TF.js Model for Mobile (320x320)
│   ├── model.json              
│   └── [shard .bin files...]
│
└── yolov8s_web_model/          # Exported TF.js Model for Desktop (640x640)
    ├── model.json              
    └── [shard .bin files...]
```
---

# 📄 File Descriptions

| File / Folder | Description |
|--------------|------------|
| `index.html` | Handles webcam feed, UI layout, and HD Mode toggle |
| `style.css` | Cyberpunk-themed styling and responsive UI |
| `script.js` | Handles dynamic preprocessing, GPU inference, and canvas rendering |
| `Yolov8n_json.ipynb` | Python script to convert and export YOLOv8-Nano model to TF.js format |
| `Yolov8s_json.ipynb` | Python script to convert and export YOLOv8-Small model to TF.js format |
| `yolov8n_web_model/` | Contains weights for lightweight Nano model (Mobile Default) |
| `yolov8s_web_model/` | Contains weights for high-accuracy Small model (Desktop / HD Mode) |

---

# ✨ Key Features & Extreme Optimizations

- **Adaptive Dynamic Resolution**
  - Scales to `320×320` (YOLOv8n) on mobile for FPS
  - Scales to `640×640` (YOLOv8s) on desktop for accuracy

- **GPU-Accelerated NMS**
  - Uses `tf.image.nonMaxSuppressionAsync`
  - Runs on WebGL backend → avoids CPU bottlenecks & UI freezing

- **Letterboxing for Accuracy**
  - Preserves aspect ratio via padding
  - Prevents distortion and improves confidence scores

- **Universal Backend Fallback**
  - Tries **WebGL (F16 pipelines)** → falls back to **WASM / CPU**
  - Prevents crashes on unsupported devices

- **Strict Noise Filtering**
  - Confidence > 50%
  - IOU < 35%
  - Eliminates overlapping boxes & ghost detections

- **Client-Side AI**
  - No backend/server
  - Full privacy with real-time performance

---

# 📉 Current Shortcomings & Technical Analysis

### 1. Classification Fluctuations (Flickering)
- **Problem:** Objects (e.g., watches) may be misclassified  
- **Reason:** COCO dataset lacks certain classes → predicts closest match  

---

### 2. Accuracy vs Speed Trade-off (Mobile vs Desktop)
- **Problem:** Small/distant objects missed on mobile  
- **Reason:**
  - Mobile uses `320×320` input for 20–30 FPS
  - Reduced spatial detail vs `640×640` desktop mode

---

### 3. Hardware Dependency & Thermal Throttling
- **Problem:** FPS varies; mobile heating during long usage  
- **Reason:**
  - Heavy reliance on GPU via WebGL
  - Higher overhead vs native (TFLite / NPU)
  - Uses async throttling to reduce GC spikes

---

# 🚀 How to Run Locally

1. git clone [Object-Detection Repository](https://github.com/syedimad-ds/Object-Detection.git)
2. cd Object-Detection
3. Open using a local server (VS Code Live Server recommended)
4. Allow camera permissions
5. Start real-time object detection


# 🛣️ Future Roadmap

- Custom model training (e.g., watches, electronics)  
- WebGPU backend integration for faster inference  
- Temporal smoothing for stable predictions  

---

# 📊 Project Highlights (For Recruiters)

- ⚡ **Real-time Edge AI in Browser** — Zero backend dependency  
- 🔒 **Privacy-first architecture** — No data leaves user device  
- 🧠 **YOLOv8 + TensorFlow.js deployment**  

### 🎯 Advanced Inference Pipeline
- GPU-bound NMS  
- Letterboxing  
- Tensor slicing  

### 📱 Adaptive Hardware Scaling
- Dynamic memory & resolution based on device capability  

---

# 💡 Demonstrates Strong Understanding Of

- Computer Vision & Edge AI Deployment  
- Memory Management & Performance Optimization  
- Web-based AI Systems (WebGL / WASM)  
- Hyperparameter Tuning (IOU / Confidence Thresholds)  

---

# 👨‍💻 Author

**Syed Imad Muzaffar**  
🎓 3rd Year B.E. Student — AI & Data Science  
