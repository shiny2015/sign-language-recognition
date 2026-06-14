# Real-Time Sign Language Recognition

A complete real-time Sign Language Recognition system using MediaPipe hand
landmarks and a custom dataset, with a RandomForestClassifier for gesture
classification.

## 1. Project Overview

This project captures hand landmarks from a webcam using MediaPipe Hands,
saves them into a custom CSV dataset, trains a machine learning model
(RandomForestClassifier) on that dataset, and then performs real-time
gesture prediction with confidence scores and FPS display.

**Default gesture classes:** A, B, C, Hello, Thanks
(more classes can be added without modifying any code — see Section 6)

### Project Structure