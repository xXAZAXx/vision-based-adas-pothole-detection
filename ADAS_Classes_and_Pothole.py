import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ---------------------------------------------------------------
# Loading models
# ---------------------------------------------------------------
adas_model = YOLO("yolov8n.pt")
pothole_model = YOLO("runs/detect/train9/weights/best.pt") # Custom Trained Pothole model

# ---------------------------------------------------------------
# Video input
# ---------------------------------------------------------------
cap = cv2.VideoCapture(0)

# ---------------------------------------------------------------
# Dynamic object tracking memory
# ---------------------------------------------------------------
track_history = defaultdict(list)

# ---------------------------------------------------------------
# ADAS classes 
# ---------------------------------------------------------------
ADAS_CLASSES = {
    "person", "car", "bus", "truck",
    "motorcycle", "bicycle",
    "traffic light", "stop sign"
}

# ---------------------------------------------------------------
# Pothole parameters 
# ---------------------------------------------------------------
MIN_POTHOLE_AREA = 500
MAX_POTHOLE_AREA = 30000
ROAD_REGION_Y = 0.5

CONF_THRESHOLD = 0.35
PERSIST_FRAMES = 4
MATCH_DISTANCE = 80
EXPIRE_FRAMES = 20
MAX_POTHOLES = 10

pothole_candidates = []
confirmed_potholes = []
next_pothole_id = 1   # ID 

# ---------------------------------------------------------------
# Helper: suppress near confirmed potholes
# ---------------------------------------------------------------
def near_confirmed(cx, cy, confirmed, dist=80):
    for p in confirmed:
        if abs(cx - p["cx"]) < dist and abs(cy - p["cy"]) < dist:
            return True
    return False

# ---------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # ---------------------------------------------------------------============================
    # Dynamic objects (ADAS)
    # ---------------------------------------------------------------============================
    results = adas_model.track(
        frame,
        tracker="bytetrack.yaml",
        persist=True,
        conf=0.4,
        iou=0.5
    )

    for r in results:
        for box in r.boxes:
            if box.id is None:
                continue

            label = adas_model.names[int(box.cls)]
            if label not in ADAS_CLASSES:
                continue

            obj_id = int(box.id)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2)//2, (y1 + y2)//2

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} ID:{obj_id}",
                        (x1, y1-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            track_history[obj_id].append((cx, cy))
            if len(track_history[obj_id]) > 10:
                track_history[obj_id].pop(0)

    # ---------------------------------------------------------------============================
    # Pothole detection (ID)
    # ---------------------------------------------------------------============================
    pothole_results = pothole_model(frame, conf=CONF_THRESHOLD)

    for r in pothole_results:
        for box in r.boxes:
            label = pothole_model.names[int(box.cls)].lower()
            if label != "pothole":
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            area = (x2-x1)*(y2-y1)

            if area < MIN_POTHOLE_AREA or area > MAX_POTHOLE_AREA:
                continue

            if y2 < int(h * ROAD_REGION_Y):
                continue

            cx, cy = (x1+x2)//2, (y1+y2)//2

            # Ignore if near confirmed pothole
            if near_confirmed(cx, cy, confirmed_potholes):
                continue

            matched = False

            for p in pothole_candidates:
                if abs(cx - p["cx"]) < MATCH_DISTANCE and abs(cy - p["cy"]) < MATCH_DISTANCE:
                    old_area = (p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1])

                    # size consistency check
                    if abs(area - old_area) < 0.5 * old_area:
                        p["count"] += 1
                    else:
                        p["count"] = max(1, p["count"] - 1)

                    p["bbox"] = (x1,y1,x2,y2)
                    p["cx"], p["cy"] = cx, cy
                    p["missed"] = 0
                    matched = True

                    if p["count"] >= PERSIST_FRAMES:
                        if len(confirmed_potholes) < MAX_POTHOLES:
                            p["id"] = f"P{len(confirmed_potholes)+1}"
                            confirmed_potholes.append(p)
                        pothole_candidates.remove(p)
                    break

            if not matched:
                pothole_candidates.append({
                    "cx": cx,
                    "cy": cy,
                    "count": 1,
                    "bbox": (x1,y1,x2,y2),
                    "missed": 0
                })

    # ---------------------------------------------------------------============================
    # Draw confirmed potholes (ONE per pothole)
    # ---------------------------------------------------------------============================
    updated = []
    for p in confirmed_potholes:
        p["missed"] += 1
        if p["missed"] < EXPIRE_FRAMES:
            x1,y1,x2,y2 = p["bbox"]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(frame, f"POTHOLE {p['id']}",
                        (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            updated.append(p)

    confirmed_potholes = updated

    # ---------------------------------------------------------------============================
    # Display
    # ---------------------------------------------------------------============================
    cv2.imshow("ADAS Perception (Tracking + Potholes)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
