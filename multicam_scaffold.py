"""
multicam_scaffold.py

install in your terminal:

    pip install ultralytics opencv-python numpy scipy pyyaml
    
"""

# --------------------------- IMPORTS ---------------------------
import os
import cv2
import time
import json
import queue
import threading
import yaml
import numpy as np
from datetime import datetime
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
try:
    import torch  # for CUDA availability check
    _CUDA_AVAILABLE = torch.cuda.is_available()
    _CUDA_VERSION = getattr(torch.version, 'cuda', None)
except Exception:
    torch = None
    _CUDA_AVAILABLE = False
    _CUDA_VERSION = None

# --------------------------- CONFIG ---------------------------
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
    'cam0': "4p-c0-homography.yml",
    'cam1': "4p-c1-homography.yml",
    'cam2': "4p-c2-homography.yml",
    'cam3': "4p-c3-homography.yml",
}

MODEL_PATH = 'yolov10m.pt'
CONF_THRESH = 0.35
# Device selection: 'auto' (default), 'cuda', or 'cpu'. Env override: YOLO_DEVICE
DEVICE = os.environ.get('YOLO_DEVICE', 'auto')
if DEVICE == 'auto':
    DEVICE = 'cuda' if _CUDA_AVAILABLE else 'cpu'
MAX_MISSED = 12
MODEL_IMAGE_SIZE = 512   # was 640, hardware is tough enough as is
USE_HALF_PRECISION = DEVICE != 'cpu'

CAPTURE_BACKEND = cv2.CAP_FFMPEG if hasattr(cv2, 'CAP_FFMPEG') else None
CAPTURE_BUFFER = 2
FRAME_QUEUE_SIZE = 4
RECONNECT_DELAY = 2.0
MAX_RECONNECT_ATTEMPTS = 5
FRAME_SKIP = 2  # was 0 (then later +1) becomes process frames where idx % 3 == 0, so once every 3 frames rather than every frame

ENABLE_VISUALIZATION = True  # we can set this to False later to disable drawing/windows for better performance during the gallery

RTSP_CAPTURE_OPTIONS = "rtsp_transport;tcp|max_delay;0|fflags;nobuffer|flags;low_delay|timeout;5000000"
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', RTSP_CAPTURE_OPTIONS)

DETECTOR_LOCK = threading.Lock()

# Cross-camera association parameters
ASSOCIATION_TIME_WINDOW = 8.0
CROSS_APPEARANCE_WEIGHT = 0.6
CROSS_SPATIAL_WEIGHT = 0.4
CROSS_COST_THRESHOLD = 0.7

OUTPUT_CSV = 'positions.csv'
TRACK_DIR = 'tracks'
os.makedirs(TRACK_DIR, exist_ok=True)

# --------------------------- UTILITIES ---------------------------
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

def project_point(H, point):
    """
    Project an image point to ground-plane coordinates using homography.
    Returns (X, Y) or None if H is None or division by zero occurs.
    """
    if H is None:
        return None
    p = np.array([point[0], point[1], 1.0], dtype=float)
    g = H @ p
    if g[2] == 0:
        return None
    return float(g[0]/g[2]), float(g[1]/g[2])

def iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two boxes.
    Boxes are (x1,y1,x2,y2).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(1e-6, (boxA[2]-boxA[0])*(boxA[3]-boxA[1]))
    areaB = max(1e-6, (boxB[2]-boxB[0])*(boxB[3]-boxB[1]))
    union = areaA + areaB - interArea
    return 0.0 if union <= 0 else interArea / union

def centroid_from_box(box):
    """Compute centroid (cx, cy) from box coordinates."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def crop_safe(frame, box):
    """Safely crop a region from frame, ensuring coordinates are within bounds."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(max(0, box[0])), int(max(0, box[1])),
                      int(min(w-1, box[2])), int(min(h-1, box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

def color_hist_feature(crop, bins=16): #from 32 to 16 halves feature dim
    """Compute normalized grayscale histogram as simple appearance feature."""
    if crop is None:
        return np.zeros(bins, dtype=float)
    # shrink cropping
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).flatten()
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist

def appearance_distance(f1, f2):
    """Compute appearance distance between two features (1 - cosine similarity)."""
    if f1 is None or f2 is None:
        return 1.0
    f1 = f1 / (np.linalg.norm(f1) + 1e-9)
    f2 = f2 / (np.linalg.norm(f2) + 1e-9)
    dot = np.clip(np.dot(f1, f2), -1.0, 1.0)
    return 1.0 - dot

def valid_ground(pt):
        return (
            pt is not None
            and isinstance(pt, (list, tuple))
            and len(pt) == 2
            and all(v is not None for v in pt)
        )

# --------------------------- TRACK & TRACKER ---------------------------
class Track:
    """Represents a single tracked person in a single camera."""
    def __init__(self, local_id, bbox, feature, frame_idx, timestamp, cam_id):
        self.local_id = local_id
        self.cam_id = cam_id
        self.bboxes = [bbox]
        self.features = [feature]
        self.last_bbox = bbox
        self.first_seen = frame_idx
        self.last_seen = frame_idx
        self.first_time = timestamp
        self.last_time = timestamp
        self.positions = []
        cx, cy = centroid_from_box(bbox)
        self.positions.append((frame_idx, timestamp, cx, cy, None, None))
        self.missed = 0
        self.global_id = None

    def predict(self):
        """Return the last known bounding box as prediction."""
        return self.last_bbox

    def update(self, bbox, feature, frame_idx, timestamp, ground=None):
        """Update track with new detection."""
        self.bboxes.append(bbox)
        self.features.append(feature)
        self.last_bbox = bbox
        self.last_seen = frame_idx
        self.last_time = timestamp
        cx, cy = centroid_from_box(bbox)
        gx, gy = ground if ground is not None else (None, None)
        self.positions.append((frame_idx, timestamp, cx, cy, gx, gy))
        self.missed = 0

    def mark_missed(self):
        """Increment missed detection counter."""
        self.missed += 1

    def to_summary(self):
        """Return JSON-serializable summary of track."""
        return {
            'global_id': self.global_id,
            'local_cam': self.cam_id,
            'first_seen_frame': int(self.first_seen),
            'last_seen_frame': int(self.last_seen),
            'first_seen_time': float(self.first_time),
            'last_seen_time': float(self.last_time),
            'num_detections': len(self.bboxes),
            'bbox_history': [[float(x) for x in b] for b in self.bboxes],
            'positions': [
                [int(p[0]), float(p[1]), float(p[2]), float(p[3]),
                 None if p[4] is None else float(p[4]),
                 None if p[5] is None else float(p[5])]
                for p in self.positions
            ]
        }

class SimpleTracker:
    """Lightweight per-camera tracker using IoU + appearance + Hungarian assignment."""
    def __init__(self, cam_id, iou_thresh=0.3, appearance_weight=0.5, max_missed=MAX_MISSED):
        self.cam_id = cam_id
        self.iou_thresh = iou_thresh
        self.appearance_weight = appearance_weight
        self.max_missed = max_missed
        self.next_local_id = 0
        self.tracks = []

    def _create_track(self, bbox, feature, frame_idx, timestamp):
        """Create a new track."""
        t = Track(self.next_local_id, bbox, feature, frame_idx, timestamp, cam_id=self.cam_id)
        self.next_local_id += 1
        self.tracks.append(t)
        return t

    def update(self, detections, frame, frame_idx, timestamp, homography=None):
        """
        Update tracker with new detections.
        Returns terminated tracks exceeding missed threshold.
        """
        det_boxes = [d[:4] for d in detections]
        det_feats = []
        det_grounds = []

        for b in det_boxes:
            crop = crop_safe(frame, b)
            det_feats.append(color_hist_feature(crop))
            c = centroid_from_box(b)
            g = project_point(homography, c) if homography is not None else None
            det_grounds.append(g)

        N, M = len(self.tracks), len(det_boxes)

        # If no existing tracks, create tracks for all detections
        if N == 0:
            for i, b in enumerate(det_boxes):
                self._create_track(b, det_feats[i], frame_idx, timestamp)
        else:
            # Compute cost matrix between tracks and detections
            cost = np.zeros((N, M), dtype=float)
            for i, tr in enumerate(self.tracks):
                pred = tr.predict()
                for j, db in enumerate(det_boxes):
                    iou_cost = 1.0 - iou(pred, db)
                    app_cost = appearance_distance(tr.features[-1], det_feats[j])
                    cost[i, j] = (1.0 - self.appearance_weight) * np.clip(iou_cost, 0, 1) + \
                                 self.appearance_weight * np.clip(app_cost, 0, 1)

            # Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost)
            assigned_tracks, assigned_dets = set(), set()

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < 0.7:
                    self.tracks[r].update(det_boxes[c], det_feats[c], frame_idx, timestamp, det_grounds[c])
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

            # Create tracks for unassigned detections
            for j in range(M):
                if j not in assigned_dets:
                    self._create_track(det_boxes[j], det_feats[j], frame_idx, timestamp)

            # Mark unassigned tracks as missed
            for i in range(N):
                if i not in assigned_tracks:
                    self.tracks[i].mark_missed()

        # Cleanup terminated tracks
        alive, terminated = [], []
        for t in self.tracks:
            if t.missed > self.max_missed:
                terminated.append(t)
            else:
                alive.append(t)
        self.tracks = alive
        return terminated

# --------------------------- CROSS-CAMERA ASSOCIATION ---------------------------
class CrossCameraManager:
    """
    Handles cross-camera association by maintaining a buffer of recently-terminated tracks.
    Assigns global IDs across cameras based on appearance, spatial distance, and time.
    """
    def __init__(self, homographies):
        self.homographies = homographies
        self.paused_tracks = []  # list of paused tracks for cross-camera matching
        self.global_id_counter = 0
        self.global_map = {}  # global_id -> list of (cam, local_id, start_time)

    def add_terminated_local_tracks(self, tracks):
        """Add recently-terminated local tracks to paused buffer for potential cross-camera matching."""
        now = time.time()
        for t in tracks:
            last_pos = t.positions[-1]
            gx, gy = last_pos[4], last_pos[5]
            feat = t.features[-1]
            self.paused_tracks.append({
                'cam': t.cam_id,
                'last_time': float(t.last_time),
                'last_pos_ground': (gx, gy) if gx is not None else None,
                'feature': feat,
                'global_id': t.global_id
            })
        # Remove old paused tracks
        cutoff = now - max(60, ASSOCIATION_TIME_WINDOW * 3)
        self.paused_tracks = [p for p in self.paused_tracks if p['last_time'] >= cutoff]


    def try_associate(self, new_tracklets):
        """Attempt to associate new tracklets with paused tracks from other cameras."""
        n, m = len(self.paused_tracks), len(new_tracklets)
        if n == 0 or m == 0:
            return []

        cost = np.zeros((n, m), dtype=float)
        for i, p in enumerate(self.paused_tracks):
            for j, nv in enumerate(new_tracklets):
                dt = nv['start_time'] - p['last_time']
                if dt < 0 or dt > ASSOCIATION_TIME_WINDOW:
                    cost[i, j] = 1e6
                    continue
                app = appearance_distance(p['feature'], nv['feature'])
                if not valid_ground(p['last_pos_ground']) or not valid_ground(nv['start_pos_ground']):
                    spatial = 50.0
                else:
                    spatial = float(
                        np.linalg.norm(
                            np.array(p['last_pos_ground'], dtype=float)
                            - np.array(nv['start_pos_ground'], dtype=float)
                        )
                    )
                spatial_n = min(spatial / 10.0, 1.0)
                cost[i, j] = CROSS_APPEARANCE_WEIGHT * app + CROSS_SPATIAL_WEIGHT * spatial_n

        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < CROSS_COST_THRESHOLD:
                matches.append((self.paused_tracks[r], new_tracklets[c], float(cost[r, c])))

        # Assign global IDs
        for p, nv, _ in matches:
            if p['global_id'] is None:
                gid = self.global_id_counter
                self.global_id_counter += 1
                p['global_id'] = gid
            else:
                gid = p['global_id']
            nv['assign_global'](gid)
            self.global_map.setdefault(gid, []).append((nv['cam'], nv.get('local_id', None), nv['start_time']))
        return matches

    def force_assign_new(self, new_entry):
        """Force a new global ID assignment for unmatched tracklet."""
        gid = self.global_id_counter
        self.global_id_counter += 1
        new_entry['assign_global'](gid)
        self.global_map.setdefault(gid, []).append((new_entry['cam'], new_entry.get('local_id', None), new_entry['start_time']))
        return gid

# --------------------------- CAMERA WORKER ---------------------------
class CameraWorker(threading.Thread):
    """Thread that captures frames, detects persons, and tracks per camera."""
    def __init__(self, cam_id, source, homography, detector, detector_lock, out_queue, stop_event):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.source = source
        self.homography = homography
        self.detector = detector
        self.detector_lock = detector_lock
        self.out_q = out_queue
        self.stop_event = stop_event
        self.tracker = SimpleTracker(cam_id)
        self.frame_buffer = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self._frame_counter = 0

    def _open_capture(self):
        cap = None
        if CAPTURE_BACKEND is not None:
            cap = cv2.VideoCapture(self.source, CAPTURE_BACKEND)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(self.source)
        if CAPTURE_BUFFER is not None:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, CAPTURE_BUFFER)
            except cv2.error:
                pass
        return cap

    def _reader_loop(self):
        cap = None
        reconnect_attempts = 0
        while not self.stop_event.is_set():
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = self._open_capture()
                if not cap or not cap.isOpened():
                    reconnect_attempts += 1
                    if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                        print(f"[{self.cam_id}] Unable to open stream after {reconnect_attempts} attempts.")
                        reconnect_attempts = 0
                    time.sleep(RECONNECT_DELAY)
                    continue
                reconnect_attempts = 0

            ret, frame = cap.read()
            if not ret:
                reconnect_attempts += 1
                if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                    print(f"[{self.cam_id}] Read failed {reconnect_attempts} times, reopening stream.")
                    cap.release()
                    cap = None
                    reconnect_attempts = 0
                    time.sleep(RECONNECT_DELAY)
                continue

            reconnect_attempts = 0
            ts = time.time()
            idx = self._frame_counter
            self._frame_counter += 1

            if FRAME_SKIP and (idx % (FRAME_SKIP + 1) != 0):
                continue

            if self.frame_buffer.full():
                try:
                    self.frame_buffer.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.frame_buffer.put_nowait((idx, ts, frame))
            except queue.Full:
                pass

        if cap is not None:
            cap.release()

    def _run_detector(self, frame):
        try:
            with self.detector_lock:
                results = self.detector.predict(frame, imgsz=MODEL_IMAGE_SIZE, device=DEVICE,
                                                half=USE_HALF_PRECISION, verbose=False)
        except Exception as exc:
            print(f"[{self.cam_id}] Detector failure: {exc}")
            return None
        return results[0] if results else None

    def run(self):
        reader = threading.Thread(target=self._reader_loop, daemon=True)
        reader.start()

        while not self.stop_event.is_set():
            try:
                frame_idx, ts, frame = self.frame_buffer.get(timeout=1.0)
            except queue.Empty:
                continue

            results = self._run_detector(frame)
            if results is None:
                continue

            # Extract person detections above confidence threshold
            dets = []
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < CONF_THRESH or cls != 0:
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                dets.append((x1, y1, x2, y2, conf))

            terminated = self.tracker.update(dets, frame, frame_idx, ts, homography=self.homography)

            # Collect newly created tracklets for cross-camera association
            new_tracklets = []
            for t in self.tracker.tracks:
                if t.first_seen == frame_idx:
                    last_pos = t.positions[-1]
                    start_ground = (last_pos[4], last_pos[5])
                    def make_assign_fn(track_ref=t):
                        return lambda gid: setattr(track_ref, 'global_id', gid)
                    new_tracklets.append({
                        'cam': self.cam_id,
                        'start_time': float(t.first_time),
                        'start_pos_ground': start_ground,
                        'feature': t.features[-1],
                        'local_id': t.local_id,
                        'assign_global': make_assign_fn()
                    })

            # Send updates to main thread
            self.out_q.put({
                'cam': self.cam_id,
                'frame_idx': frame_idx,
                'timestamp': ts,
                'new_tracklets': new_tracklets,
                'terminated_tracks': terminated,
                'active_tracks': list(self.tracker.tracks),
            })

            # Visualization (same idea as before but with the option to disable)
            if ENABLE_VISUALIZATION:
                vis = frame.copy()
                for t in self.tracker.tracks:
                    x1, y1, x2, y2 = map(int, t.last_bbox)
                    gid = t.global_id if t.global_id is not None else -1
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(vis, f"L{t.local_id}/G{gid}", (x1, max(15, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
                cv2.imshow(f"{self.cam_id}", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    break
        reader.join(timeout=2.0)

# --------------------------- IO UTILITIES ---------------------------
def write_csv_header():
    with open(OUTPUT_CSV, 'w') as f:
        f.write('global_id,camera_id,frame_idx,timestamp_iso,x_ground,y_ground,x1,y1,x2,y2,confidence\n')

def append_positions_csv(global_id, cam_id, frame_idx, ts, gx, gy, bbox, conf=1.0):
    iso = datetime.utcfromtimestamp(ts).isoformat()
    x1, y1, x2, y2 = bbox
    with open(OUTPUT_CSV, 'a') as f:
        f.write(f"{global_id},{cam_id},{frame_idx},{iso},{'' if gx is None else gx},{'' if gy is None else gy},{int(x1)},{int(y1)},{int(x2)},{int(y2)},{conf}\n")

def save_track_json(track: Track):
    gid = track.global_id if track.global_id is not None else -1
    fname = os.path.join(TRACK_DIR, f"track_{gid}_cam_{track.cam_id}_local_{track.local_id}.json")
    with open(fname, 'w') as f:
        json.dump(track.to_summary(), f, indent=2)

# --------------------------- MAIN ---------------------------
def main():
    # Load YOLO detector
    detector = YOLO(MODEL_PATH)
    try:
        tv = torch.__version__ if torch is not None else 'n/a'
    except Exception:
        tv = 'n/a'
    print(f"[INFO] Inference device: {DEVICE} | torch {tv} | cuda_version={_CUDA_VERSION} | cuda_available={_CUDA_AVAILABLE}")

    # Load homographies per camera
    homographies = {}
    for cam, path in HOMOGRAPHIES.items():
        try:
            homographies[cam] = load_homography(path) if path else None
        except Exception as e:
            print(f"[WARN] Failed to load homography for {cam}: {e}")
            homographies[cam] = None

    # Check camera availability
    q = queue.Queue()
    stop_event = threading.Event()
    workers = {}

    print("[INFO] Checking camera sources...")
    valid_sources = {}
    for cam, src in SOURCES.items():
        test_cap = cv2.VideoCapture(src)
        if test_cap.isOpened():
            print(f"[OK] Camera {cam} detected.")
            valid_sources[cam] = src
        else:
            print(f"[ERROR] Could not access camera {cam}. Skipping.")
        test_cap.release()
    if not valid_sources:
        print("[FATAL] No working cameras found. Exiting.")
        return

    # Start camera worker threads
    for cam, src in valid_sources.items():
        H = homographies.get(cam, None)
        w = CameraWorker(cam, src, H, detector, DETECTOR_LOCK, q, stop_event)
        workers[cam] = w
        w.start()

    manager = CrossCameraManager(homographies)
    write_csv_header()

    try:
        while not stop_event.is_set():
            try:
                item = q.get(timeout=1.0)
            except queue.Empty:
                continue

            terminated = item['terminated_tracks']
            new_tracklets = item['new_tracklets']
            active_tracks = item['active_tracks']

            manager.add_terminated_local_tracks(terminated)
            manager.try_associate(new_tracklets)

            # Assign new global IDs to unmatched tracks
            for nt in new_tracklets:
                if nt['assign_global'] is not None:
                    manager.force_assign_new(nt)

            # Append active tracks to CSV
            for t in active_tracks:
                last = t.positions[-1]
                gx, gy = last[4], last[5]
                append_positions_csv(
                    t.global_id if t.global_id is not None else -1,
                    t.cam_id, t.last_seen, t.last_time, gx, gy, t.last_bbox
                )

            # Save terminated tracks to JSON
            for t in terminated:
                save_track_json(t)

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user. Stopping...")
        stop_event.set()

    # Wait for threads to finish
    for w in workers.values():
        w.join()
    cv2.destroyAllWindows()
    print("Done. Outputs:", OUTPUT_CSV, "and", TRACK_DIR)

if __name__ == "__main__":
    main()
