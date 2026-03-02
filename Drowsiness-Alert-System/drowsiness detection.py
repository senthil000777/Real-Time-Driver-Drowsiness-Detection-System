import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from keras.models import load_model
from pygame import mixer
import sys
import time

# -------------------------------------------------
# Base Directory
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def check_file(path, name):
    if not os.path.exists(path):
        print(f"❌ Missing {name}: {path}")
        sys.exit()

# -------------------------------------------------
# Initialize Alarm (Strong & Stable)
# -------------------------------------------------
mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

alarm_path = os.path.join(BASE_DIR, 'alarm.wav')
check_file(alarm_path, "alarm.wav")

sound = mixer.Sound(alarm_path)
sound.set_volume(1.0)   # 🔥 Maximum volume

# -------------------------------------------------
# Load Haar Cascades
# -------------------------------------------------
leye = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'haar cascade files', 'haarcascade_lefteye_2splits.xml'))
reye = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'haar cascade files', 'haarcascade_righteye_2splits.xml'))

# -------------------------------------------------
# Load CNN Model
# -------------------------------------------------
model_path = os.path.join(BASE_DIR, 'models', 'cnncat2.h5')
check_file(model_path, "CNN Model")
model = load_model(model_path)

# -------------------------------------------------
# Auto Camera Detection
# -------------------------------------------------
cap = None
for i in range(5):
    temp = cv2.VideoCapture(i)
    if temp.isOpened():
        cap = temp
        print(f"✅ Camera found at index {i}")
        break
    temp.release()

if cap is None:
    print("❌ No camera detected")
    sys.exit()

# -------------------------------------------------
# Variables
# -------------------------------------------------
font = cv2.FONT_HERSHEY_SIMPLEX
closed_start_time = None
alert_seconds = 5
alarm_on = False

# -------------------------------------------------
# Main Loop
# -------------------------------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    rpred = [1]
    lpred = [1]

    # -------- Right Eye --------
    for (x,y,w,h) in right_eye:
        eye = gray[y:y+h, x:x+w]
        eye = cv2.resize(eye, (24,24))
        eye = eye / 255.0
        eye = eye.reshape(1,24,24,1)
        rpred = np.argmax(model.predict(eye, verbose=0), axis=1)
        break

    # -------- Left Eye --------
    for (x,y,w,h) in left_eye:
        eye = gray[y:y+h, x:x+w]
        eye = cv2.resize(eye, (24,24))
        eye = eye / 255.0
        eye = eye.reshape(1,24,24,1)
        lpred = np.argmax(model.predict(eye, verbose=0), axis=1)
        break

    # -------------------------------------------------
    # Drowsiness Logic (5 Seconds Continuous)
    # -------------------------------------------------
    if rpred[0] == 0 and lpred[0] == 0:

        status = "CLOSED"
        frame_color = (0, 0, 255)

        if closed_start_time is None:
            closed_start_time = time.time()

        elapsed = time.time() - closed_start_time

        # Show timer on screen
        cv2.putText(frame, f"Closed Time: {int(elapsed)} sec",
                    (10, 40), font, 0.8,
                    (0,0,255), 2)

        if elapsed >= alert_seconds:

            cv2.putText(frame, "DROWSINESS ALERT!",
                        (50,100), font, 1,
                        (0,0,255), 3)

            if not alarm_on:
                sound.play(-1)   # 🔥 Loop alarm continuously
                alarm_on = True

    else:
        status = "OPEN"
        frame_color = (0, 255, 0)
        closed_start_time = None

        if alarm_on:
            sound.stop()
            alarm_on = False

    # -------------------------------------------------
    # Draw UI
    # -------------------------------------------------
    cv2.rectangle(frame, (0,0), (width,height), frame_color, 5)

    cv2.putText(frame, f"Status: {status}",
                (10,height-20), font,
                0.8, (255,255,255), 2)

    cv2.imshow("Drowsiness Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()