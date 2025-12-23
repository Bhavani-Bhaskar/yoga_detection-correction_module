import base64
import io
import os
import random
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from flask import (
    Flask,
    Response,
    json,
    render_template,
    request,
    send_from_directory,
)

from utils import calculate_angle

app = Flask(__name__)

# -------------------- Load Pose References --------------------

medianreference_path = os.path.join(os.path.dirname(__file__), 'descriptions/medianreference.json')
with open(medianreference_path) as f:
    POSE_REFERENCE = json.load(f)

POSE_NAMES = list(POSE_REFERENCE.keys())

POSE_IMAGE_DIR = "display_images"
os.makedirs(POSE_IMAGE_DIR, exist_ok=True)

POSE_DESCRIPTIONS = {}
pose_description_path = os.path.join(os.path.dirname(__file__), 'descriptions/pose_description.json')
try:
    with open(pose_description_path) as f:
        POSE_DESCRIPTIONS = json.load(f)
except FileNotFoundError:
    POSE_DESCRIPTIONS = {name: {"description": "", "instructions": ""} for name in POSE_NAMES}

# -------------------- MediaPipe Setup --------------------

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------- Globals --------------------

latest_frame = None
latest_landmarks = None
current_angles = None
lock = threading.Lock()

current_pose = None
timer_active = False
remaining_time = 0
pose_history = []

modal_visible = True
all_angles_correct = False
angle_statuses = {}

# -------------------- Angle Definitions --------------------

YOGA_ANGLES = {
    "right-hand": [16, 14, 12],
    "left-hand": [15, 13, 11],
    "right-arm-body": [14, 12, 24],
    "left-arm-body": [13, 11, 23],
    "right-leg-body": [12, 24, 26],
    "left-leg-body": [11, 23, 25],
    "right-leg": [24, 26, 28],
    "left-leg": [23, 25, 27],
}

# -------------------- Angle Calculations --------------------

def calc_yoga_angles(landmarks):
    ang = {}
    for name, (p1, p2, p3) in YOGA_ANGLES.items():
        vis = [landmarks[i]["visibility"] for i in (p1, p2, p3)]
        if min(vis) < 0.5:
            ang[name] = None
        else:
            a = (landmarks[p1]['x'], landmarks[p1]['y'])
            b = (landmarks[p2]['x'], landmarks[p2]['y'])
            c = (landmarks[p3]['x'], landmarks[p3]['y'])
            ang[name] = calculate_angle(a, b, c)
    return ang


def check_individual_angles_and_feedback(current_angles, current_pose):
    if current_pose not in POSE_REFERENCE or not current_angles:
        return ({name: False for name in YOGA_ANGLES}, "")

    pose_ref = POSE_REFERENCE[current_pose]
    statuses = {}
    feedback = []

    for angle_name, value in current_angles.items():
        statuses[angle_name] = False
        if value is None:
            continue

        ref = pose_ref.get(angle_name.replace("-", "_"))
        if not ref:
            continue

        if ref["median"] - ref["iqr"] <= value <= ref["median"] + ref["iqr"]:
            statuses[angle_name] = True
        else:
            feedback.append(angle_name)

    return statuses, " ".join(feedback)


def check_angles_within_threshold(current_angles, current_pose):
    statuses, _ = check_individual_angles_and_feedback(current_angles, current_pose)
    return all(statuses.values())


def get_angle_feedback(current_angles, current_pose):
    _, feedback = check_individual_angles_and_feedback(current_angles, current_pose)
    return feedback

# -------------------- Pose Logic --------------------

def get_next_pose():
    global pose_history
    if len(pose_history) == len(POSE_NAMES):
        pose_history = []
    pose = random.choice([p for p in POSE_NAMES if p not in pose_history])
    pose_history.append(pose)
    return pose


def set_current_pose(pose_name):
    global current_pose, timer_active, modal_visible, all_angles_correct, angle_statuses
    timer_active = False
    current_pose = pose_name
    modal_visible = True
    all_angles_correct = False
    angle_statuses = {k: False for k in YOGA_ANGLES}


def start_timer(duration=30):
    global timer_active, remaining_time, modal_visible
    if not check_angles_within_threshold(current_angles, current_pose):
        return
    modal_visible = False
    timer_active = True
    remaining_time = duration

    def countdown():
        global remaining_time, timer_active
        while remaining_time > 0 and timer_active:
            time.sleep(1)
            remaining_time -= 1

    threading.Thread(target=countdown, daemon=True).start()

# -------------------- Webcam Thread --------------------

def capture_loop():
    global latest_frame, current_angles, all_angles_correct, angle_statuses
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        with lock:
            latest_frame = frame.copy()
            if results.pose_landmarks:
                landmarks = [
                    {"x": lm.x, "y": lm.y, "visibility": lm.visibility}
                    for lm in results.pose_landmarks.landmark
                ]
                current_angles = calc_yoga_angles(landmarks)
                if current_pose:
                    angle_statuses, _ = check_individual_angles_and_feedback(
                        current_angles, current_pose
                    )
                    all_angles_correct = all(angle_statuses.values())

                mp_drawing.draw_landmarks(
                    latest_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

threading.Thread(target=capture_loop, daemon=True).start()

# -------------------- ROUTES (ALL RESTORED) --------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/app')
def yoga_app():
    pose = get_next_pose()
    set_current_pose(pose)
    return render_template('app.html')


@app.route('/get_angles')
def get_angles():
    return current_angles or {}


@app.route('/get_angle_statuses')
def get_angle_statuses():
    return angle_statuses


@app.route('/get_angle_status')
def get_angle_status():
    return {
        "all_correct": all_angles_correct,
        "feedback": "" if all_angles_correct else get_angle_feedback(current_angles, current_pose)
    }


@app.route('/get_thresholds')
def get_thresholds():
    if not current_pose:
        return {}
    pose_ref = POSE_REFERENCE[current_pose]
    return {
        k: {
            "lower": pose_ref[k.replace("-", "_")]["median"] - pose_ref[k.replace("-", "_")]["iqr"],
            "upper": pose_ref[k.replace("-", "_")]["median"] + pose_ref[k.replace("-", "_")]["iqr"]
        }
        for k in YOGA_ANGLES if k.replace("-", "_") in pose_ref
    }


@app.route('/get_modal_state')
def get_modal_state():
    return {"visible": modal_visible}


@app.route('/get_timer')
def get_timer():
    return {"remaining_time": remaining_time, "timer_active": timer_active}


@app.route('/close_modal', methods=['POST'])
def close_modal():
    global modal_visible
    modal_visible = False   # 🔴 THIS WAS MISSING
    start_timer()
    return ('', 204)



def gen_frames():
    while True:
        if latest_frame is None:
            continue
        _, buf = cv2.imencode('.jpg', latest_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.05)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_current_pose')
def get_current_pose():
    global current_pose

    if not current_pose:
        current_pose = get_next_pose()

    pose_data = POSE_DESCRIPTIONS.get(current_pose, {})

    image_filename_jpg = f"{current_pose}.jpg"
    image_filename_png = f"{current_pose}.png"

    image_path_jpg = os.path.join(POSE_IMAGE_DIR, image_filename_jpg)
    image_path_png = os.path.join(POSE_IMAGE_DIR, image_filename_png)

    if os.path.exists(image_path_jpg):
        image_url = f"/idealposes/{image_filename_jpg}"
    elif os.path.exists(image_path_png):
        image_url = f"/idealposes/{image_filename_png}"
    else:
        image_url = f"https://via.placeholder.com/300x200.png?text={current_pose.replace(' ', '+')}"

    return {
        "name": current_pose,
        "image_url": image_url,
        "benefits": pose_data.get("description", ""),
        "instructions": pose_data.get("instructions", "")
    }

from urllib.parse import unquote

@app.route('/idealposes/<path:filename>')
def serve_pose_image(filename):
    # Decode %20, %28, %29 etc.
    filename = unquote(filename)

    file_path = os.path.join(POSE_IMAGE_DIR, filename)

    if not os.path.exists(file_path):
        # Fail gracefully instead of breaking UI
        return ('', 204)

    return send_from_directory(POSE_IMAGE_DIR, filename)

if __name__ == '__main__':
    app.run(host='localhost', port=5000, threaded=True)
