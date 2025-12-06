import cv2
import mediapipe as mp
import numpy as np
import os

GESTURES = ["hello","namaskar","love_you","home","hi","promise","help","name"]
DATA_DIR = "data"  # folder to save npy files

os.makedirs(DATA_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(results):
    """
    Return a 126-dim vector:
    [21*3 for LEFT hand] + [21*3 for RIGHT hand]
    If a hand is missing, its part is filled with zeros.
    """
    if not results.multi_hand_landmarks:
        return None

    # Prepare containers for left/right
    left_hand = None
    right_hand = None

    # Map landmarks to left/right using handedness info
    for hand_label, hand_landmarks in zip(results.multi_handedness,
                                          results.multi_hand_landmarks):
        label = hand_label.classification[0].label  # 'Left' or 'Right'
        if label == 'Left':
            left_hand = hand_landmarks
        elif label == 'Right':
            right_hand = hand_landmarks

    data = []

    # Helper to add 21*3 coords (or zeros if None)
    def add_hand_landmarks(hand):
        if hand is not None:
            for lm in hand.landmark:
                data.extend([lm.x, lm.y, lm.z])
        else:
            # 21 landmarks × 3 coords = 63 zeros
            data.extend([0.0] * 63)

    # Always: [LEFT then RIGHT] for consistency
    add_hand_landmarks(left_hand)
    add_hand_landmarks(right_hand)

    return np.array(data)  # shape (126,)


cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=2,             # 👈 allow 2 hands
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:

    print("Press keys 0-7 to save a sample for the corresponding gesture:")
    for i, g in enumerate(GESTURES):
        print(f"{i}: {g}")

    sample_count = {g: 0 for g in GESTURES}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        cv2.putText(
            image,
            "Press 0-7 to save, q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Show how many samples per class (small overlay)
        y0 = 60
        for i, g in enumerate(GESTURES):
            text = f"{i}:{g}={sample_count[g]}"
            cv2.putText(image, text, (10, y0 + 25 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Collect Data", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # If key between 0-7 pressed, save sample
        if ord('0') <= key <= ord('7'):
            label_index = key - ord('0')
            if label_index < len(GESTURES):
                landmarks = extract_landmarks(results)
                if landmarks is not None:
                    gesture_name = GESTURES[label_index]
                    filename = os.path.join(
                        DATA_DIR,
                        f"{gesture_name}_{sample_count[gesture_name]}.npy"
                    )
                    np.save(filename, landmarks)
                    sample_count[gesture_name] += 1
                    print(f"Saved {filename}")
                else:
                    print("No hands detected, sample not saved.")

cap.release()
cv2.destroyAllWindows()
