# 🚗 Advanced Driver Assistance System (ADAS) Simulation

A comprehensive autonomous vehicle simulation system featuring real-time path planning, object detection, and lane tracking capabilities. This project implements cutting-edge computer vision and AI techniques to create a robust driver assistance system.

## 📚 Table of Contents
- [Overview](#overview)
- [🗂️ Project Structure](#project-structure)
- [⚙️ Setup](#setup)
- [🚀 Usage](#usage)
- [🧠 System Architecture](#system-architecture)
- [⚙️ Configuration](#configuration)
- [🛠️ Technologies Used](#technologies-used)
- [🔍 Features](#features)

## Overview

This ADAS simulation system combines multiple AI and computer vision components to create a comprehensive autonomous driving assistant. Key features include:
- 🎯 **Real-time Object Detection** using YOLOv8
- 🛣️ **Lane Detection** with UltrafastLaneDetectorV2
- 🗺️ **Path Planning** using optimized A* algorithm
- 👁️ **Object Tracking** with BYTE Tracker
- 📏 **Distance Measurement** using single camera
- 📊 **Rich Visualization** with real-time statistics

## 🗂️ Project Structure

```plaintext
autopilot/
├── App/
│   └── control_panel.py      # Main application control and visualization
├── Object/
│   ├── ObjectDetector/       # YOLO-based object detection
│   ├── LaneDetector/         # Lane detection implementation
│   ├── PathPlanning/         # A* pathfinding algorithm
│   └── ObjectTracker/        # BYTE tracking system
└── config/                   # Configuration files
```

## ⚙️ Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Sufficient RAM for video processing

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd autopilot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model weights:**
   - YOLOv8 weights
   - UltrafastLaneDetectorV2 weights

## 🚀 Usage

To run the ADAS simulation:

```bash
python main.py
```

Controls:
- Press 'Q' to quit simulation
- Real-time statistics displayed in console

## 🧠 System Architecture

### Object Detection
- Uses YOLOv8 detector (supports v5, v8, v10)
- Real-time detection of vehicles, pedestrians, and obstacles
- Configurable through object_config dictionary

### Lane Detection
- UltrafastLaneDetectorV2 implementation
- Bird's eye view transformation
- Real-time lane curvature and vehicle offset calculation

### Path Planning
- Optimized A* algorithm implementation
- Real-time obstacle avoidance
- Efficient grid-based path calculation using NumPy

### Object Tracking
- BYTE Tracker integration
- Maintains object persistence across frames
- Efficient tracking algorithm optimized for vehicles

## ⚙️ Configuration

The system uses configuration dictionaries for flexible setup:

```python
object_config = {
    "model_type": YOLOv8,
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4
}

line_config = {
    "model_type": UFLDV2,
    "perspective_transform": True
}
```

## 🛠️ Technologies Used

- **OpenCV**: Image processing and visualization
- **NumPy**: Efficient array operations
- **PyTorch**: Deep learning models
- **Rich**: Console output and logging
- **PyCUDA**: GPU acceleration
- **Custom modules**: Detection and tracking

## 🔍 Features

### Real-time Processing
- Optimized frame processing
- Pre-allocated memory buffers
- Batch processing capabilities
- Efficient video codec usage

### Visualization
- Rich console output
- Real-time FPS monitoring
- System status table
- Bird's eye view transformation
- Path planning visualization
- Object tracking display

### Performance Optimizations
- NumPy-based calculations
- Efficient memory management
- GPU acceleration where possible
- Optimized grid operations

## 🔄 Future Improvements

- Enhanced path planning algorithms
- More sophisticated collision prediction
- Advanced machine learning model integration
- Improved real-time performance
- Additional sensor integration capabilities

## 🔒 Security Considerations

- Input validation for video sources
- Error handling for model loading
- Safe configuration management
- Resource cleanup procedures

---

For more information or contributions, please contact the project maintainers.
