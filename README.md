# Vision-Based ADAS Pothole Detection & Mapping

This project explores a vision-based Advanced Driver Assistance System (ADAS) pipeline for pothole detection and sparse environment mapping using a monocular camera.

A custom-trained YOLOv8 model is used for real-time pothole detection, while camera motion is estimated using optical flow to visualize a vehicle or robot trajectory. The project also investigates sparse mapping using ORB-SLAM3. Click the image below, to watch the demo on YouTube
[![ADAS Pothole Detection Demo](https://img.youtube.com/vi/3sI9vEBju2g/0.jpg)](https://www.youtube.com/watch?v=3sI9vEBju2g)

---

## Project Motivation

The goal of this project is to gain hands-on experience with key ADAS concepts, including:
- Object detection and tracking
- Vision-based perception
- Sparse mapping and SLAM
- Sensor-based environment understanding

This project is intended as a learning and portfolio project rather than a production-ready system.

---

## Features Implemented

- Custom-trained YOLOv8 model for pothole detection  
- Real-time pothole detection using OpenCV and a webcam  
- Camera trajectory estimation using optical flow 
- Path visualization with pothole events highlighted  
- ADAS-relevant object detection (e.g., potholes, vehicles, traffic lights)  

---

## Technologies Used

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- Matplotlib
- ORB-SLAM3 (C++, Linux)
- Webcam (Monocular camera)

---

## Attempted / Experimental Work

- Installed and ran ORB-SLAM3 on Linux using a monocular webcam
- Generated sparse 3D map points from ORB-SLAM3
- Exported sparse map data and visualized it using a Python script on Windows
- Investigated depth estimation using the MiDaS depth model

While ORB-SLAM3 successfully runs and produces sparse maps, further calibration and sensor synchronization are required to improve mapping accuracy.

---

## Current Status

- ✔ Pothole detection working reliably
- ✔ Camera trajectory visualization implemented
- ⚠ Sparse mapping accuracy is limited
- ⚠ Depth-based mapping improvements in progress

---

## Future Improvements

- Improve scale estimation and drift correction
- Fuse depth estimation with trajectory mapping
- Integrate LiDAR or stereo vision 
- Associate pothole detections with world coordinates
---




