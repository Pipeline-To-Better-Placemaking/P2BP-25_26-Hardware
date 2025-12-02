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

# === World / top-view config for 6-people laboratory ===
TV_ORIGIN_X = 0.0
TV_ORIGIN_Y = 0.0
TV_WIDTH    = 358.0   # top-view width in EPFL units
TV_HEIGHT   = 360.0   # top-view height in EPFL units

# Fixed world box in top-view coordinates
WORLD_MIN_X = TV_ORIGIN_X
WORLD_MAX_X = TV_ORIGIN_X + TV_WIDTH     # 358
WORLD_MIN_Y = TV_ORIGIN_Y
WORLD_MAX_Y = TV_ORIGIN_Y + TV_HEIGHT    # 360

# White frame specs
MAP_WIDTH  = 400
MAP_HEIGHT = int(MAP_WIDTH * (TV_HEIGHT / TV_WIDTH))

MAP_BOX_TOP_LEFT = (50, 50)
MAP_BOX_BOTTOM_RIGHT = (MAP_WIDTH - 50, MAP_HEIGHT - 50)

GRID_COLS = 56
GRID_ROWS = 56

# Rotate and/or flip the top-view
TOPVIEW_ROTATION = 90
TOPVIEW_FLIP_X   = True
TOPVIEW_FLIP_Y   = False


current_cam = 'cam0'
SOURCES = {
    # 'cam0': 0,  # 0 for local webcam
    # 'cam1': 'rtsp://user:pass@192.168.1.20:554/stream',
    'cam0': 'rtsp://127.0.0.1:8554/cam0',
    'cam1': 'rtsp://127.0.0.1:8554/cam1',
    # 'cam2': 'rtsp://127.0.0.1:8554/cam2',
    # 'cam3': 'rtsp://127.0.0.1:8554/cam3',
}

source_names = ['4p', 'terrace1', 'campus4', 'passageway1']
source_name = source_names[0]
HOMOGRAPHIES = {
    # 'cam1': 'homography_cam1.yml',
    'cam0': load_homography(f"homographies/{source_name}-c0-homography.yml"),
    'cam1': load_homography(f"homographies/{source_name}-c1-homography.yml"),
    'cam2': load_homography(f"homographies/{source_name}-c2-homography.yml"),
    'cam3': load_homography(f"homographies/{source_name}-c3-homography.yml"),
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
TV_trails = defaultdict(lambda: deque(maxlen=15))
last_seen = {}
frame_idx = 0

results = model.track(
    source=SOURCES[current_cam], 
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
        gx, gy = homography_transform(HOMOGRAPHIES[current_cam], feet_px)

        # store ground-plane point for this frame
        current_ground_points.append((tid, gx, gy))

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
    world_w = max(WORLD_MAX_X - WORLD_MIN_X, 1e-6)
    world_h = max(WORLD_MAX_Y - WORLD_MIN_Y, 1e-6)

    x1_box, y1_box = MAP_BOX_TOP_LEFT
    x2_box, y2_box = MAP_BOX_BOTTOM_RIGHT
    box_w = x2_box - x1_box
    box_h = y2_box - y1_box

    for tid, gx, gy in current_ground_points:
        tx = (gx - WORLD_MIN_X) / world_w  # 0 = left, 1 = right
        ty = (gy - WORLD_MIN_Y) / world_h  # 0 = bottom, 1 = top

        # rotate the grid if needed
        if TOPVIEW_ROTATION == 0:
            tx_r, ty_r = tx, ty
        elif TOPVIEW_ROTATION == 90:
            tx_r, ty_r = ty, 1.0 - tx
        elif TOPVIEW_ROTATION == 180:
            tx_r, ty_r = 1.0 - tx, 1.0 - ty
        elif TOPVIEW_ROTATION == 270:
            tx_r, ty_r = 1.0 - ty, tx
        else:
            tx_r, ty_r = tx, ty

        if TOPVIEW_FLIP_X:
            tx_r = 1.0 - tx_r
        if TOPVIEW_FLIP_Y:
            ty_r = 1.0 - ty_r

        u = int(x1_box + tx_r * box_w)
        v = int(y2_box - ty_r * box_h)
        TV_trails[tid].append((tx_r, ty_r))

        cv2.circle(topview, (u, v), 4, color_based_on_id(tid), -1)
        cv2.putText(
            topview,
            str(tid),
            (u + 6, v - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color_based_on_id(tid),
            1
        )
        # draw trails on top-view
        pts = TV_trails[tid]
        for i in range(1, len(pts)):
            p1 = pts[i - 1]
            p2 = pts[i]
            u1 = int(x1_box + p1[0] * box_w)
            v1 = int(y2_box - p1[1] * box_h)
            u2 = int(x1_box + p2[0] * box_w)
            v2 = int(y2_box - p2[1] * box_h)
            cv2.line(topview, (u1, v1), (u2, v2), color_based_on_id(tid), 2)


    cv2.imshow('Frame', frame)
    cv2.imshow('TopView', topview)
    if cv2.waitKey(1) == 27:
        break

    stale_tids = [tid for tid, seen_at in last_seen.items() if frame_idx - seen_at > STALE_FRAMES]

    for tid in stale_tids:
        trails.pop(tid, None)
        last_seen.pop(tid, None)

