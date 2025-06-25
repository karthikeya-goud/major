from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from threading import Thread, Lock
from ultralytics import YOLO
import time
import joblib

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
# Load ML model for traffic classification
ad_selector = joblib.load('ad_selector_model.pkl')

# Resize size for frames and ads
billboard_size = (640, 480)

# Ad metadata with pricing
ad_data = [
    {"name": "bmw", "path": "static/ads/bmw.jpg", "level": "low", "bid": 0.5},
    {"name": "star", "path": "static/ads/star.jpg", "level": "medium", "bid": 1.2},
    {"name": "diary", "path": "static/ads/diary.jpg", "level": "high", "bid": 1.8},
]

# Organize ads by traffic level
ads_by_level = {"low": [], "medium": [], "high": []}
for ad in ad_data:
    img = cv2.imread(ad["path"])
    if img is None:
        print(f"Warning: Could not read {ad['path']}")
        continue
    image = cv2.resize(img, billboard_size)
    ads_by_level[ad["level"]].append({
        "name": ad["name"],
        "image": image,
        "bid": ad["bid"]
    })

def select_ad_by_bid(traffic_level):
    candidates = ads_by_level.get(traffic_level, [])
    if not candidates:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    return max(candidates, key=lambda ad: ad["bid"])["image"]

# Shared variables
frame_lock = Lock()
latest_frame = None
output_frame = np.zeros((480, 640, 3), dtype=np.uint8)
current_ad_frame = np.zeros((480, 640, 3), dtype=np.uint8)
running = False

# Thread for grabbing frames
class FrameGrabber(Thread):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.daemon = True

    def run(self):
        global latest_frame
        while running and self.cap.isOpened():
            self.cap.grab()
            ret, frame = self.cap.retrieve()
            if not ret:
                time.sleep(0.05)
                continue
            with frame_lock:
                latest_frame = frame
        self.cap.release()

# Detection + ad selection loop
def detect_and_stream():
    global latest_frame, output_frame, current_ad_frame
    last_detection_time = 0
    detection_interval = 0.3  # seconds (about 3 FPS)

    while running:
        if time.time() - last_detection_time < detection_interval:
            time.sleep(0.01)
            continue
        last_detection_time = time.time()

        with frame_lock:
            if latest_frame is None:
                continue
            frame = cv2.resize(latest_frame, (640, 480))

        results = model(frame, verbose=False)
        boxes = results[0].boxes

        vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        vehicle_count = 0

        if boxes and boxes.cls is not None:
            classes = boxes.cls.cpu().numpy().astype(int)
            vehicle_count = sum(cls in vehicle_classes for cls in classes)

        ad_index = ad_selector.predict([[vehicle_count]])[0]
        traffic_levels = ["low", "medium", "high"]
        traffic_level = traffic_levels[ad_index]

        ad = select_ad_by_bid(traffic_level)

        with frame_lock:
            output_frame = frame
            current_ad_frame = ad

def generate_video():
    while True:
        with frame_lock:
            frame = output_frame
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_ad():
    while True:
        with frame_lock:
            ad = current_ad_frame
        _, buffer = cv2.imencode('.jpg', ad)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ad_feed')
def ad_feed():
    return Response(generate_ad(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control():
    global running
    action = request.form.get('action')
    if action == 'start' and not running:
        running = True
        FrameGrabber().start()
        Thread(target=detect_and_stream, daemon=True).start()
    elif action == 'stop':
        running = False
    return ('', 204)

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
