"""
utils.py
Reusable utility functions for the Sign Language Recognition project.
"""

import os
import csv
import numpy as np
import mediapipe as mp


def extract_landmarks(results):
    """
    Extract 21 hand landmarks (x, y, z) = 63 values total.
    Returns a list of 63 floats, or None if no hand is detected.
    """
    if not results.multi_hand_landmarks:
        return None

    # Use the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]

    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    if len(landmarks) != 63:
        return None

    return landmarks


def normalize_landmarks(landmarks):
    """
    Normalize landmark coordinates relative to the wrist (landmark 0)
    and scale by the max distance from the wrist, so the model becomes
    robust to hand position and size changes in the frame.

    Input: list of 63 floats (x1,y1,z1,...,x21,y21,z21)
    Output: list of 63 normalized floats
    """
    landmarks = np.array(landmarks, dtype=np.float32).reshape(21, 3)

    # Translate so wrist (landmark 0) is the origin
    wrist = landmarks[0].copy()
    translated = landmarks - wrist

    # Scale by the maximum Euclidean distance from the wrist
    max_dist = np.max(np.linalg.norm(translated, axis=1))
    if max_dist == 0:
        max_dist = 1e-6  # avoid division by zero

    normalized = translated / max_dist

    return normalized.flatten().tolist()


def draw_hand_landmarks(frame, results):
    """
    Draw hand landmarks and connections on the given frame.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

    return frame


def create_csv_if_not_exists(path):
    """
    Create the CSV dataset file with a header row if it does not exist.
    Header: x1,y1,z1,x2,y2,z2,...,x21,y21,z21,label
    """
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        header = []
        for i in range(1, 22):  # 21 landmarks
            header.extend([f"x{i}", f"y{i}", f"z{i}"])
        header.append("label")

        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)