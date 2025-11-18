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

# White frame specs
MAP_WIDTH, MAP_HEIGHT = 400, 400

MAP_BOX_TOP_LEFT = (50, 50)
MAP_BOX_BOTTOM_RIGHT = (350, 350) 

GRID_COLS = 4 
GRID_ROWS = 4

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

world_min_x = world_max_x = None
world_min_y = world_max_y = None

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

    topview = 255 * np.ones((MAP_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8)

    # draw ground plane
    cv2.rectangle(
        topview,
        MAP_BOX_TOP_LEFT,
        MAP_BOX_BOTTOM_RIGHT,
        (0, 0, 0),
        2
    )

    # draw grid
    x1_box, y1_box = MAP_BOX_TOP_LEFT
    x2_box, y2_box = MAP_BOX_BOTTOM_RIGHT

    cell_w = (x2_box - x1_box) / GRID_COLS
    cell_h = (y2_box - y1_box) / GRID_ROWS

    # vertical grid lines
    for c in range(1, GRID_COLS):
        x = int(x1_box + c * cell_w)
        cv2.line(topview, (x, y1_box), (x, y2_box), (200, 200, 200), 1)

    # horizontal grid lines
    for r_grid in range(1, GRID_ROWS):
        y = int(y1_box + r_grid * cell_h)
        cv2.line(topview, (x1_box, y), (x2_box, y), (200, 200, 200), 1)

    current_ground_points = []

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
        gx, gy = homography_transform(HOMOGRAPHIES['cam0'], feet_px)

        # store ground-plane point for this frame
        current_ground_points.append((tid, gx, gy))

        # update global world bounds
        if world_min_x is None:
            world_min_x = world_max_x = gx
            world_min_y = world_max_y = gy
        else:
            world_min_x = min(world_min_x, gx)
            world_max_x = max(world_max_x, gx)
            world_min_y = min(world_min_y, gy)
            world_max_y = max(world_max_y, gy)


        cv2.ellipse(
            frame,
            center,
            (axis_x, axis_y),
            0,
            20, 270,
            color_based_on_id(tid),
            2
        )

        text_lines = [
            f"{tid}",
            f"({int(gx)}, {int(gy)})"
        ]
        for i, line in enumerate(text_lines):
            cv2.putText(
                frame,
                line,
                (center[0]+5, center[1]-10 + i * 12),   # shift each line down by 12 pixels
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color_based_on_id(tid),
                1
            )
        #cv2.putText(frame, f'{tid} ({int(gx)}, {int(gy)})', (center[0]+5, center[1]-10),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_based_on_id(tid), 1)

        
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

    # draw current ground-plane points on Plane
    if world_min_x is not None and world_max_x is not None:
        x1_box, y1_box = MAP_BOX_TOP_LEFT
        x2_box, y2_box = MAP_BOX_BOTTOM_RIGHT
        box_w = x2_box - x1_box
        box_h = y2_box - y1_box

        world_w = max(world_max_x - world_min_x, 1e-6)
        world_h = max(world_max_y - world_min_y, 1e-6)

        for tid, gx, gy in current_ground_points:
            tx = (gx - world_min_x) / world_w
            ty = (gy - world_min_y) / world_h

            u = int(x1_box + tx * box_w)
            v = int(y2_box - ty * box_h)

            # draw dot for this ID
            cv2.circle(topview, (u, v), 4, color_based_on_id(tid), -1)

            # draw the ID next to the dot
            cv2.putText(
                topview,
                str(tid),
                (u + 6, v - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color_based_on_id(tid),
                1
            )

    cv2.imshow('Frame', frame)
    cv2.imshow('TopView', topview)
    if cv2.waitKey(1) == 27:
        break


    cv2.imshow('Frame', frame)
    cv2.imshow('TopView', topview)
    if cv2.waitKey(1) == 27:
        break

    stale_tids = [tid for tid, seen_at in last_seen.items() if frame_idx - seen_at > STALE_FRAMES]

    for tid in stale_tids:
        trails.pop(tid, None)
        last_seen.pop(tid, None)

