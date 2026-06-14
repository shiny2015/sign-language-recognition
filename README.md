# Real-Time Sign Language Recognition

A complete real-time Sign Language Recognition system using MediaPipe hand
landmarks and a RandomForestClassifier, trainable either from a custom
webcam-collected dataset or from a Kaggle ASL image dataset.

## 1. Project Overview

This project detects hand landmarks from a webcam (or images) using
MediaPipe Hands, builds a landmark-based dataset, trains a machine learning
model (RandomForestClassifier), and performs real-time gesture prediction
with confidence scores and FPS display.

You can build the dataset in two ways:
- **Custom webcam collection** (`data_collection.py`)
- **Kaggle ASL image dataset conversion** (`convert_kaggle_to_landmarks.py`)

**Default gesture classes:** A, B, C, Hello, Thanks
(more classes can be added without modifying any code — see Section 6)

### Project Structure
