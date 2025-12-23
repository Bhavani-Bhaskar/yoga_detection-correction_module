# Yoga Detection and Correction Module

An AI-powered web application designed to provide real-time biomechanical feedback for yoga practitioners. Using **MediaPipe Pose** and **OpenCV**, the system analyzes body alignment and joint angles to ensure users perform poses safely and effectively.

## 🌟 Key Features

* **Real-Time Pose Analysis**: Captures webcam feed to track 33 body landmarks using MediaPipe.
* **Angle Validation**: Calculates joint angles for arms, legs, and torso based on biomechanical coordinates.
* **Dynamic Feedback**: Compares user posture against reference medians and Interquartile Ranges (IQR) to identify specific misalignments.
* **Automated Timer**: Features a countdown timer that activates only when the user achieves and maintains the correct posture.
* **Pose Encyclopedia**: Displays pose names, benefits, and step-by-step instructions fetched from local JSON datasets.

## 🛠️ Tech Stack

* **Backend**: Python, Flask
* **Computer Vision**: OpenCV, MediaPipe
* **Data Handling**: Pandas, NumPy
* **Concurrency**: Threading for simultaneous camera processing and web serving

## 📐 Angle Monitoring Logic

The module tracks the following specific joint configurations:
* **Arms**: `left-hand`, `right-hand`, `left-arm-body`, `right-arm-body`
* **Legs**: `left-leg`, `right-leg`, `left-leg-body`, `right-leg-body`

A pose is considered "correct" when the calculated angle $\theta$ satisfies:
$$\text{Median} - \text{IQR} \leq \theta \leq \text{Median} + \text{IQR}$$

## 🚀 Installation & Setup

### Prerequisites
* Docker installed on your system.
* A connected webcam.

### Build the Image
Navigate to the project root and run:
```bash
docker build -t yogaimg .
```

### Run the Container
```bash
docker run -it --rm \
  -p 5000:5000 \
  --device=/dev/video0:/dev/video0 \
  --privileged \
  yogaimg
```
