import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from tensorflow.keras.models import load_model

MODEL_PATH = "sign_model.h5"
LABELS_PATH = "label_classes.npy"

# Load trained model and label names
model = load_model(MODEL_PATH)
label_classes = np.load(LABELS_PATH)
label_classes = [str(l) for l in label_classes]   # ensure plain strings

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ====== CONFIG ======
SMOOTHING_WINDOW = 7      # number of frames to average predictions over
SHOW_TOP_K = 3            # show top-K classes on side panel
CAM_INDEX = 0             # usually 0; change if needed
# =====================

def extract_landmarks(results):
    """
    Return a 126-dim vector:
    [21*3 for LEFT hand] + [21*3 for RIGHT hand]
    If a hand is missing, its part is zeros.
    """
    if not results.multi_hand_landmarks:
        return None

    left_hand = None
    right_hand = None

    # Map hands using handedness info
    for hand_label, hand_landmarks in zip(results.multi_handedness,
                                          results.multi_hand_landmarks):
        label = hand_label.classification[0].label  # 'Left' or 'Right'
        if label == 'Left':
            left_hand = hand_landmarks
        elif label == 'Right':
            right_hand = hand_landmarks

    data = []

    def add_hand(hand):
        if hand is not None:
            for lm in hand.landmark:
                data.extend([lm.x, lm.y, lm.z])
        else:
            data.extend([0.0] * 63)  # 21*3 zeros

    # Always LEFT, then RIGHT
    add_hand(left_hand)
    add_hand(right_hand)

    return np.array(data)  # shape (126,)


def draw_main_label(image, text):
    """Big banner at bottom with current predicted label."""
    h, w, _ = image.shape
    banner_height = 70

    # semi-transparent rectangle
    overlay = image.copy()
    cv2.rectangle(overlay, (0, h - banner_height), (w, h), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # centered text
    font_scale = 1.2
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = (w - tw) // 2
    y = h - (banner_height - th) // 2 - 10
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 255, 0), thickness, cv2.LINE_AA)


def draw_topk_probs(image, probs, labels, k=3):
    """Draw a side panel with top-k gesture probabilities."""
    if probs is None:
        return image

    h, w, _ = image.shape
    panel_width = 260
    x0 = w - panel_width
    y0 = 0

    # panel background
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # Title
    cv2.putText(image, "Top predictions", (x0 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # sort by prob desc
    probs = np.array(probs)
    indices = np.argsort(probs)[::-1]
    indices = indices[:min(k, len(indices))]

    bar_x_start = x0 + 10
    bar_x_end = w - 20
    max_bar_width = bar_x_end - bar_x_start
    y = 60
    step = 40

    for idx in indices:
        label = labels[idx]
        p = float(probs[idx])
        bar_width = int(max_bar_width * p)

        # bar background
        cv2.rectangle(image, (bar_x_start, y - 15),
                      (bar_x_end, y + 10), (50, 50, 50), -1)
        # bar fill
        cv2.rectangle(image, (bar_x_start, y - 15),
                      (bar_x_start + bar_width, y + 10), (0, 180, 255), -1)

        text = f"{label}: {p*100:.1f}%"
        cv2.putText(image, text, (bar_x_start, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += step

    return image


def draw_header(image, fps):
    """Draw small header with FPS and instructions."""
    txt = f"FPS: {fps:.1f}  |  Press 'Q' to quit"
    cv2.putText(image, txt, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)


# ======== MAIN LOOP =========
cap = cv2.VideoCapture(CAM_INDEX)

# prediction smoothing
probs_history = deque(maxlen=SMOOTHING_WINDOW)

prev_time = time.time()

with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # compute FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-8)
        prev_time = curr_time

        image = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # default text if nothing detected
        main_text = "No hand detected"
        smoothed_probs = None

        if results.multi_hand_landmarks:
            # draw landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

            landmarks = extract_landmarks(results)
            if landmarks is not None:
                input_data = landmarks.reshape(1, -1)  # (1, 126)
                probs = model.predict(input_data, verbose=0)[0]  # (num_classes,)
                probs_history.append(probs)

                # smoothing
                smoothed_probs = np.mean(probs_history, axis=0)

                # final prediction from smoothed probs
                idx = int(np.argmax(smoothed_probs))
                conf = float(smoothed_probs[idx])
                main_text = f"{label_classes[idx]}  ({conf*100:.1f}%)"

        # draw HUD elements
        draw_header(image, fps)
        draw_main_label(image, main_text)
        if smoothed_probs is not None:
            draw_topk_probs(image, smoothed_probs, label_classes, k=SHOW_TOP_K)

        cv2.imshow("Sign Language Recognition", image)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

cap.release()
cv2.destroyAllWindows()
