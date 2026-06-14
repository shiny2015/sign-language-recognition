"""
predict.py
Real-time sign language gesture prediction using webcam, MediaPipe hand landmarks,
and a trained RandomForestClassifier model.

Press 'Q' to exit.
"""

import cv2
import pickle
import time
import os
import numpy as np
import mediapipe as mp

from utils import extract_landmarks, normalize_landmarks, draw_hand_landmarks

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MODEL_PATH = os.path.join("models", "sign_model.pkl")
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.0  # Set higher (e.g., 0.6) to filter low-confidence predictions

# ---------------------------------------------------------
# LOAD MODEL AND LABEL ENCODER
# ---------------------------------------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(
        "Trained model or label encoder not found. "
        "Run train_model.py first to generate 'models/sign_model.pkl' "
        "and 'models/label_encoder.pkl'."
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

print("Model and label encoder loaded successfully.")
print(f"Available classes: {list(label_encoder.classes_)}")

# ---------------------------------------------------------
# SETUP MEDIAPIPE
# ---------------------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

prev_time = 0

print("\nStarting real-time prediction. Press 'Q' to quit.\n")

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
while True:
    success, frame = cap.read()
    if not success:
        print("ERROR: Failed to read frame from webcam.")
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    # Draw landmarks on the frame
    frame = draw_hand_landmarks(frame, results)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Extract and predict only if a hand is detected
    landmarks = extract_landmarks(results)

    if landmarks is not None:
        normalized = normalize_landmarks(landmarks)
        input_data = np.array(normalized).reshape(1, -1)

        # Predict probabilities for confidence score
        probabilities = model.predict_proba(input_data)[0]
        predicted_index = np.argmax(probabilities)
        confidence = probabilities[predicted_index]

        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        if confidence >= CONFIDENCE_THRESHOLD:
            display_text = f"Sign: {predicted_label}"
            confidence_text = f"Confidence: {confidence * 100:.2f}%"
        else:
            display_text = "Sign: Uncertain"
            confidence_text = f"Confidence: {confidence * 100:.2f}%"

        cv2.putText(frame, display_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, confidence_text, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Display FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Real-Time Sign Language Prediction", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

# ---------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Prediction stopped.")