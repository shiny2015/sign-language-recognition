"""
data_collection.py
Collects custom hand-gesture landmark data via webcam and saves it to a CSV file.

Controls:
- Press 'S' to save the current frame's landmarks under the current label.
- Press 'N' to enter a new label without restarting the script.
- Press 'Q' to quit.
"""

import cv2
import csv
import time
import mediapipe as mp

from utils import (
    extract_landmarks,
    normalize_landmarks,
    draw_hand_landmarks,
    create_csv_if_not_exists,
)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
CSV_PATH = "data/gestures.csv"
CAMERA_INDEX = 0

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
create_csv_if_not_exists(CSV_PATH)

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

# ---------------------------------------------------------
# ASK FOR INITIAL LABEL
# ---------------------------------------------------------
current_label = input("Enter the gesture label to collect (e.g., A, B, C, Hello, Thanks): ").strip()
samples_collected = 0

print("\nControls:")
print("  S - Save current landmarks under current label")
print("  N - Change/enter a new label")
print("  Q - Quit\n")

prev_time = 0

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
while True:
    success, frame = cap.read()
    if not success:
        print("ERROR: Failed to read frame from webcam.")
        break

    frame = cv2.flip(frame, 1)  # Mirror image for natural interaction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    frame = draw_hand_landmarks(frame, results)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Overlay information on the frame
    cv2.putText(frame, f"Label: {current_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Samples Collected: {samples_collected}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, "Press S=Save, N=New Label, Q=Quit", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Data Collection - Sign Language", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') or key == ord('S'):
        landmarks = extract_landmarks(results)
        if landmarks is not None:
            normalized = normalize_landmarks(landmarks)
            row = normalized + [current_label]

            with open(CSV_PATH, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            samples_collected += 1
            print(f"Saved sample #{samples_collected} for label '{current_label}'")
        else:
            print("No hand detected. Sample not saved.")

    elif key == ord('n') or key == ord('N'):
        cv2.destroyAllWindows()
        current_label = input("Enter new gesture label: ").strip()
        samples_collected = 0
        print(f"Now collecting samples for label: '{current_label}'")

    elif key == ord('q') or key == ord('Q'):
        break

# ---------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Data collection finished. Dataset saved to:", CSV_PATH)