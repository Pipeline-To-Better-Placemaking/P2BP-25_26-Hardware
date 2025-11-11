import os
import cv2
import time
import json
import yaml
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
    _CUDA_VERSION = getattr(torch.version, 'cuda', None)
except Exception:
    torch = None
    _CUDA_AVAILABLE = False
    _CUDA_VERSION = None

def load_homography(path):
    """
    Load a 3x3 homography matrix from a YAML file.
    Returns None if path is None.
    """
    if path is None:
        return None
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    H = np.array(data.get('homography', None), dtype=float)
    if H.shape != (3, 3):
        raise ValueError("Homography must be 3x3")
    return H

STALE_FRAMES = 30  # frames

SOURCES = {
    # 'cam0': 0,  # 0 for local webcam
    # 'cam1': 'rtsp://user:pass@192.168.1.20:554/stream',
    'cam0': 'rtsp://127.0.0.1:8554/cam0',
    'cam1': 'rtsp://127.0.0.1:8554/cam1',
    # 'cam2': 'rtsp://127.0.0.1:8554/cam2',
    # 'cam3': 'rtsp://127.0.0.1:8554/cam3',
}

HOMOGRAPHIES = {
    # 'cam1': 'homography_cam1.yml',
    'cam0': load_homography("homographies/passageway1-c0-homography.yml"),
    'cam1': load_homography("homographies/passageway1-c1-homography.yml"),
    'cam2': load_homography("homographies/passageway1-c2-homography.yml"),
    'cam3': load_homography("homographies/passageway1-c3-homography.yml"),
}

MODEL_PATH = 'yolov10m.pt'
DEVICE = os.environ.get('YOLO_DEVICE', 'auto')
if DEVICE == 'auto':
    DEVICE = 'cuda' if _CUDA_AVAILABLE else 'cpu'

model = YOLO(MODEL_PATH)
# results = model("4p-c0.avi", show=True, device=DEVICE, classes=[0])

# cap = cv2.VideoCapture(SOURCES['cam0'])
# while True:
#     ok, frame = cap.read()
#     if not ok:
#         break



def homography_transform(H, point):
    x, y = point
    pixel_point = np.array([x, y, 1])

    transformed_point = H @ pixel_point
    transformed_point /= transformed_point[2]

    return transformed_point[0], transformed_point[1]

def color_based_on_id(tid):
    if tid == -1:
        return (0, 255, 0)
    np.random.seed(tid)
    color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
    return color

trails = defaultdict(lambda: deque(maxlen=15))
last_seen = {}
frame_idx = 0

results = model.track(
    source=SOURCES['cam0'], 
    tracker='bytetrack.yaml', 
    device=DEVICE, 
    classes=[0], 
    show=False,
    stream=True,
    verbose=False
)

for r in results:
    frame = r.orig_img.copy()
    frame_idx += 1
    boxes = r.boxes
    if not boxes:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == 27:
            break
        continue
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        tid = int(box.id.item()) if box.id is not None else -1
        feet_px = ((x1 + x2) / 2, y2)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # cv2.putText(frame, f'ID {tid}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        
        axis_x = max(4, int((x2-x1) * 0.6))
        axis_y = max(2, int((y2-y1) * 0.08))
        center = (int(feet_px[0]), int(feet_px[1] - axis_y//2))
        cv2.ellipse(
            frame,
            center,
            (axis_x, axis_y),
            0,
            0, 270,
            color_based_on_id(tid),
            2
        )

        cv2.putText(frame, f'{tid}', (center[0]+5, center[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_based_on_id(tid), 1)

        gx, gy = homography_transform(HOMOGRAPHIES['cam0'], feet_px)
        print(f"ID: {tid}, Feet Pixel: {feet_px}, gx:{gx:2f}, gy:{gy:2f}")

        if tid != -1:
            trails[tid].append(feet_px)
            last_seen[tid] = frame_idx

    # draw a trail behind each person
    for tid, pts in trails.items():
        for i in range(1, len(pts)):
            p1 = pts[i - 1]
            p2 = pts[i]
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color_based_on_id(tid), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == 27:
        break

    stale_tids = [tid for tid, seen_at in last_seen.items() if frame_idx - seen_at > STALE_FRAMES]

    for tid in stale_tids:
        trails.pop(tid, None)
        last_seen.pop(tid, None)

