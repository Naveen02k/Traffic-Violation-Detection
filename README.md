# Traffic-Violation-Detection
Mini Project

A computer vision-based system to automatically detect traffic violations from video using **YOLOv8, OpenCV, and Python**.

---

## 📌 Overview

This project detects and tracks vehicles in a video and identifies traffic violations such as:

* 🚫 Signal Jumping
* 🔄 Wrong Way Driving
* 🪖 Helmet Violation (basic detection)

It also generates:

* 📸 Cropped violation images
* 🎥 Evidence video clips
* 📊 CSV report with details
* 📈 Dashboard analytics (via Streamlit UI)

---

## 🧠 Technologies Used

* **Python**
* **OpenCV** – Video processing
* **YOLOv8 (Ultralytics)** – Object detection
* **NumPy** – Numerical operations
* **Streamlit** – UI interface
* **FFmpeg** – Video compatibility
* **Pandas** – Data analysis

---

## ⚙️ Features

* Real-time vehicle detection and tracking
* ROI-based traffic signal detection (Red/Yellow/Green)
* Smart violation detection logic
* Duplicate violation filtering
* Cropped vehicle images (no irrelevant parts)
* Evidence clips with moving bounding box
* Risk score calculation
* CSV report generation
* Interactive UI with analytics dashboard

---

## 🖥️ System Workflow

1. Upload video
2. Extract frames using OpenCV
3. Detect vehicles using YOLOv8
4. Track vehicles using centroid tracking
5. Detect traffic signal using ROI (HSV color detection)
6. Apply violation detection logic
7. Save outputs (images, clips, CSV)
8. Display results in UI dashboard

---

## 📂 Project Structure

```
Traffic_Violation_System/
│
├── main.py                 # Core detection system
├── app.py                  # Streamlit UI
├── yolov8n.pt              # YOLO model weights
├── output/                 # Output folder
│   ├── signal_jump/
│   ├── wrong_way/
│   ├── helmet_violation/
│   ├── evidence_clips/
│   └── violations.csv
│
└── uploaded_videos/
```

---

## 🚀 How to Run

### 1️⃣ Install dependencies

```bash
pip install opencv-python ultralytics numpy pandas streamlit
```

---

### 2️⃣ Run UI

```bash
streamlit run app.py
```

---

### 3️⃣ Upload video and run detection

* Select traffic direction
* Set confidence
* Adjust signal ROI (optional)
* Click **Run Detection**

---

## 📊 Dashboard Features

* Violation count graph
* Risk score analysis
* Vehicle type distribution
* Frame-wise violation trends

---

## ⚠️ Limitations

* Helmet detection is rule-based (not deep learning)
* Requires proper camera angle
* Number plate recognition (ANPR) not implemented
* Signal ROI must be correctly selected

---

## 🔮 Future Enhancements

* 🚘 Automatic Number Plate Recognition (ANPR)
* 🧠 Deep learning-based helmet detection
* 📡 Real-time CCTV integration
* 💳 Auto fine generation system

---

## 🎯 Conclusion

This project demonstrates how **computer vision and deep learning** can be applied to automate traffic monitoring and violation detection, making roads safer and reducing manual effort.
