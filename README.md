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
├── script.js                   # Core AI Engine (Inference, NMS, Drawing)
│
└── yolov8n_web_model/          # Exported TensorFlow.js Model Folder
    ├── model.json              # Model architecture and graph topology
    ├── metadata.yaml           # Class names and configuration
    ├── group1-shard1of2.bin
    ├── group1-shard1of6.bin
    ├── group1-shard2of2.bin
    ├── group1-shard2of6.bin
    ├── group1-shard3of6.bin
    ├── group1-shard4of6.bin
    ├── group1-shard5of6.bin
    └── group1-shard6of6.bin
```
---

## 📄 File Descriptions

| File / Folder            | Description |
|--------------------------|------------|
| `index.html`             | Handles webcam feed and UI layout |
| `style.css`              | Cyberpunk-themed styling |
| `script.js`              | Handles preprocessing, inference, and rendering |
| `yolov8n_web_model/`     | Contains model weights and config files |

---

## ✨ Key Features

- **Multi-Object Detection** — Detects multiple objects in real time  
- **NMS Integration** — Removes overlapping bounding boxes  
- **Dynamic Color Labels** — Unique neon colors per class  
- **Client-Side AI** — No server, ensuring full privacy  
- **Real-Time Performance** — Optimized TensorFlow.js inference  

---

## 📉 Current Shortcomings & Technical Analysis

### 1. Classification Fluctuations (Flickering)
- **Problem:** Objects like watches may be misclassified  
- **Reason:** COCO dataset lacks certain classes → model predicts closest match  

### 2. Accuracy vs Speed Trade-off
- **Problem:** Small objects may be missed  
- **Reason:** Frame resizing to **640×640** reduces fine spatial details  

### 3. Hardware Dependency
- **Problem:** FPS varies across devices  
- **Reason:** Relies on WebGL; performance depends on GPU availability  

---

## 🚀 How to Run Locally

```text
1.git clone https://github.com/syedimad-ds/Object-Detection.git
2.cd Object-Detection
3.Open using a local server (VS Code Live Server recommended)
4.Allow camera permissions
5.Start real-time object detection
```
---

## 🛣️ Future Roadmap

- Custom model training (e.g., watches, electronics)  
- WebGPU backend integration for faster inference  
- Temporal smoothing for stable predictions  

---

## 📊 Project Highlights (For Recruiters)

- ⚡ **Real-time Edge AI in Browser** — Zero backend dependency  
- 🔒 **Privacy-first architecture** — No data leaves user device  
- 🧠 **YOLOv8 + TensorFlow.js deployment**  
- 🎯 **Optimized inference pipeline (NMS + preprocessing)**  

### 💡 Demonstrates strong understanding of:
- Computer Vision  
- Deep Learning Deployment  
- Web-based AI systems  
- Performance optimization  


## 👨‍💻 Author

**Syed Imad Muzaffar**  
🎓 3rd Year B.E. Student — AI & Data Science



