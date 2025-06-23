# Virtual Mouse using Hand Gestures
# Built with OpenCV, Mediapipe, and PyAutoGUI

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
from collections import deque
import os
import ctypes
import pyttsx3

# Create screenshot directory if it doesn't exist
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# Screenshot log file
log_file_path = os.path.join("screenshots", "screenshot_log.txt")

# Initialize camera
start = time.time()
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
print("Cam initialized in:", round(time.time() - start, 2), "seconds")

# Mediapipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Screen dimensions
screen_w, screen_h = pyautogui.size()

# Cursor movement smoothing
smoothening = 5
plocX, plocY = 0, 0

# Gesture state tracking
paused = False
dragging = False
logs = deque(maxlen=6)
show_help = True
last_screenshot_time = 0

# Voice engine for feedback
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def fingers_up(lmList):
    fingers = []
    fingers.append(1 if lmList[4][1] < lmList[3][1] else 0)
    for id in [8, 12, 16, 20]:
        fingers.append(1 if lmList[id][2] < lmList[id - 2][2] else 0)
    return fingers

def draw_help_overlay(img):
    cv2.rectangle(img, (5, 5), (470, 230), (50, 50, 50), -1)
    help_text = [
        "✋ All Fingers Up       = Pause",
        "✊ Fist (All Down)      = Resume",
        "☝️ Only Index Up        = Move Cursor",
        "👉 Index + Thumb Touch  = Left Click",
        "✌️ Middle + Ring Up     = Right Click",
        "☝️✌️🤘 Index+Middle+Ring  = Scroll Up",
        "☝️✌️🤙 Index+Middle+Pinky = Scroll Down",
        "🤙 Pinky + Thumb        = Screenshot",
        "🤏 Pinch + Move         = Drag & Drop",
        "H key                   = Toggle Help",
        "ESC                     = Exit"
    ]
    for i, txt in enumerate(help_text):
        cv2.putText(img, txt, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

# Screenshot utility
def take_screenshot():
    global last_screenshot_time
    current_time = time.time()
    if current_time - last_screenshot_time > 1:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        path = os.path.join("screenshots", filename)
        pyautogui.screenshot(path)
        with open(log_file_path, "a") as f:
            f.write(f"{timestamp}: {filename}\n")
        ctypes.windll.user32.MessageBoxW(0, "Screenshot Taken!", "Virtual Mouse", 1)
        engine.say("Screenshot taken")
        engine.runAndWait()
        logs.appendleft("Screenshot 📸")
        last_screenshot_time = current_time

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if show_help:
        draw_help_overlay(img)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if lmList:
                fingers = fingers_up(lmList)
                x1, y1 = lmList[8][1:]  # Index
                x2, y2 = lmList[4][1:]  # Thumb

                if fingers == [1, 1, 1, 1, 1]:
                    paused = True
                    logs.appendleft("Paused ✋")
                    cv2.putText(img, "Paused", (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    continue
                elif fingers == [0, 0, 0, 0, 0]:
                    paused = False
                    logs.appendleft("Resumed ✊")
                    cv2.putText(img, "Resumed", (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    continue

                if not paused:
                    x3 = np.interp(x1, (100, 540), (0, screen_w))
                    y3 = np.interp(y1, (100, 380), (0, screen_h))
                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening

                    cv2.circle(img, (x1, y1), 10, (255, 255, 0), cv2.FILLED)

                    if fingers[1] == 1 and fingers[2:] == [0, 0, 0]:
                        pyautogui.moveTo(clocX, clocY)
                        plocX, plocY = clocX, clocY

                    length = np.hypot(x2 - x1, y2 - y1)
                    if fingers[0] == 1 and fingers[1] == 1 and length < 40:
                        pyautogui.click()
                        logs.appendleft("Left Click 👈")

                    if fingers == [0, 0, 1, 1, 0]:
                        pyautogui.rightClick()
                        logs.appendleft("Right Click ✌️")

                    if fingers == [0, 1, 1, 1, 0]:
                        pyautogui.scroll(20)
                        logs.appendleft("Scroll Up ⬆️")

                    if fingers == [0, 1, 1, 0, 1]:
                        pyautogui.scroll(-20)
                        logs.appendleft("Scroll Down ⬇️")

                    if fingers == [1, 0, 0, 0, 1]:
                        take_screenshot()

                    if fingers[0] == 1 and fingers[1] == 1 and length < 40:
                        if not dragging:
                            dragging = True
                            pyautogui.mouseDown()
                            logs.appendleft("Dragging 🖱️")
                        pyautogui.moveTo(clocX, clocY)
                        plocX, plocY = clocX, clocY
                    else:
                        if dragging:
                            pyautogui.mouseUp()
                            logs.appendleft("Dropped 📤")
                            dragging = False

    for i, log in enumerate(logs):
        cv2.putText(img, f"{log}", (950, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 255, 255), 2)

    cv2.imshow("Virtual Mouse", img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('h'):
        show_help = not show_help

cap.release()
cv2.destroyAllWindows()
