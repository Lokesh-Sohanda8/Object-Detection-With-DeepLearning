# 📊 Comparative Analysis of Video Object Detection Models

## 🧠 Project Overview

This project is a **practical, experiment-driven comparison of modern object detection architectures on real-world video data**.
Instead of relying on benchmark datasets alone, the project evaluates how different models behave when applied to an actual video stream containing:

* Dynamic traffic scenes
* Small and distant objects
* Occlusions and background clutter
* Indoor and outdoor environments

The core objective is to **understand the real trade-offs between speed, accuracy, confidence stability, and deployment feasibility** when choosing an object detection model for production use.

---

## 🎯 Main Objective

To **identify the most suitable object detection model depending on real-world constraints**, such as:

* Live video processing vs offline analytics
* Accuracy vs inference speed
* GPU vs CPU / edge deployment
* False positives vs missed detections

Rather than declaring a single “best” model, the project aims to **explain *why* different models behave differently** and **where each one fits best**.

---

## 🗂️ Repository Structure & File Insights

### 📘 `yolo_model_training_new.ipynb`

This notebook focuses on **YOLO-based object detection**, particularly **YOLOv11x**.

**What happens inside:**

* Loads a pretrained YOLOv11x model
* Runs inference on a full video file (1,512 frames)
* Measures:

  * Total processing time
  * Real-world FPS (including video I/O)
  * Total object detections
  * Confidence distribution
* Generates an annotated output video

**Key takeaway:**
YOLOv11x delivers **usable real-time performance** and is ideal for **live monitoring**, but it tends to:

* Produce lower average confidence
* Be more sensitive (sometimes over-detecting objects)

---

### 📘 `hf_detr_training_model.ipynb`

This notebook evaluates **Facebook’s DETR (ResNet-50)** using Hugging Face Transformers.

**What happens inside:**

* Loads DETR using an encoder–decoder transformer architecture
* Processes the same video frame-by-frame
* Extracts:

  * Detection counts per class
  * Average confidence
  * Temporal stability of bounding boxes

**Key takeaway:**
DETR is **slow but extremely precise**.
It:

* Detects significantly more small and distant objects
* Produces very stable bounding boxes (minimal flicker)
* Shows strong global context understanding (e.g., motorcycles vs cars)

This makes it ideal for **offline analytics, audits, and forensic analysis**.

---

### 📘 `final_results_output.ipynb`

This notebook acts as the **central analysis and comparison hub**.

**What it contains:**

* Aggregated results from YOLO and DETR runs
* Side-by-side quantitative comparison:

  * FPS
  * Total detections
  * Average confidence
  * Processing time
* Class-wise breakdown (cars, people, motorcycles, cell phones, backpacks, etc.)
* Final conclusions and recommendations

**This is where raw model output turns into insight.**

---

### 📄 `📊 Object Detection Model Comparison Report.pdf`

This is the **final consolidated project report**.

**Includes:**

* Executive summary
* Model architecture comparison
* Real-world performance metrics
* Class sensitivity analysis
* Visual and numerical evidence
* Clear deployment recommendations

This document explains *why* YOLO is fast, *why* DETR is precise, and *why MobileNet SSD still matters*.

---

### 📊 `Object_Detection_Speed_Precision_Tradeoff.pptx`

A **presentation-ready summary** of the project.

**Useful for:**

* College reviews
* Project demos
* Interviews
* Technical presentations

It visually explains the **speed vs precision trade-off** across:

* MobileNet SSD
* YOLO
* DETR

---

## 🎥 Output Video Analysis

All models were evaluated on the **same 1,512-frame video**.

### 🔴 YOLOv11x

* ~11 FPS real-world speed
* Faster processing (2.7× faster than DETR)
* More detections with lower confidence
* Occasional false positives (e.g., backpacks)

### 🔵 Facebook DETR

* ~4 FPS
* Very high average confidence (~90%)
* Detects ~28% more objects
* Excellent stability and small-object recognition

**Critical insight:**
The speed drop from raw GPU inference to real-world FPS highlights how **video I/O and preprocessing matter as much as the model itself**.

---

## 🧩 How You Can Use This Project

You can adapt this repository for:

### ✅ Learning & Research

* Understand real-world model behavior beyond benchmarks
* Study CNN vs Transformer detection architectures

### ✅ Production Decision-Making

* Choose the right model based on your constraints
* Design hybrid pipelines (YOLO for live + DETR for offline)

### ✅ Portfolio & Interviews

* Demonstrates system-level thinking
* Shows performance analysis, not just model training

### ✅ Custom Use

* Replace the input video with your own footage
* Tune confidence thresholds
* Add new models for comparison

---

## 🧠 Final Recommendation

There is **no single “best” model**.

* **Need speed?** → YOLO
* **Need precision?** → DETR
* **Need edge deployment?** → MobileNet SSD
* **Need a robust system?** → Combine them

**Smart systems don’t choose one — they orchestrate many.**

---

## 🤝 Contributing & Contact

If you want to improve, extend, or collaborate on this project, feel free to reach out:

* 📧 **Email:** [lokeshsohanda27@gmail.com](mailto:lokeshsohanda27@gmail.com)
* 💼 **LinkedIn:** [https://www.linkedin.com/in/lokesh-sohanda-data-enthusiast/](https://www.linkedin.com/in/lokesh-sohanda-data-enthusiast/)
* 🧑‍💻 **GitHub:** [https://github.com/Lokesh-Sohanda8](https://github.com/Lokesh-Sohanda8)
* 📸 **Instagram:** [https://www.instagram.com/think.with.tech/](https://www.instagram.com/think.with.tech/)

---

⭐ If this project helped you understand object detection better, consider starring the repo and sharing it with others in the AI community.

---
