import os, cv2, time, torch, threading, json
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import yaml
from osnet_ibn1_lite.encoder import OsNetEncoder
from queue import Queue

# ---------------- FILES ----------------
CAMERA_CONFIG_FILE = "config/cameras_runtime.json"
MAIN_CONFIG_FILE = "config/config.json"

# ---------------- LOAD CONFIG ----------------
with open(MAIN_CONFIG_FILE, "r") as f:
    MAIN_CONFIG = json.load(f)

TRACKING_ENABLED = MAIN_CONFIG["Tracking"]["Enabled"]
CONF_THRESH = MAIN_CONFIG["Tracking"]["ConfidenceThreshold"]
MODEL_PATH = f"{MAIN_CONFIG['Tracking']['Model']}.pt"
MAX_FPS = MAIN_CONFIG["Tracking"]["MaxFps"]
FRAME_INTERVAL = 1.0 / MAX_FPS
IMG_SIZE = 640  # YOLO inference size
ENABLED_MACS = {mac for mac, v in MAIN_CONFIG["TrackingCameras"].items() if v}

# ---------------- CAMERA SOURCES ----------------
def load_camera_sources(cam_path):
    with open(cam_path, 'r') as f:
        cams = json.load(f)

    sources, cam_names, macs = [], [], []
    for name, cam in cams.items():
        if cam["mac"] in ENABLED_MACS:
            sources.append(cam["rtsp"])
            cam_names.append(name)
            macs.append(cam["mac"])
    return sources, cam_names, macs

SOURCES, CAMERA_NAMES, CAMERA_MACS = load_camera_sources(CAMERA_CONFIG_FILE)
print(f"[INFO] Tracking enabled cameras: {CAMERA_NAMES}")

# ---------------- HOMOGRAPHY ----------------
def load_homography(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return np.array(data.get('homography'))

HOMOGRAPHIES = [load_homography(f"homographies/{name}-homography.yml") for name in CAMERA_NAMES]

def homography_transform(H, point):
    if H is None:
        return point
    x, y = point
    p = np.array([x, y, 1.0])
    p = H @ p
    p /= p[2]
    return float(p[0]), float(p[1])

# ---------------- MODELS ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_MODEL = YOLO(MODEL_PATH)
YOLO_LOCK = threading.Lock()

osnet = OsNetEncoder(
    input_width=704, input_height=480,
    weight_filepath="osnet_ibn1_lite/model_weights.pth.tar-40",
    batch_size=32, num_classes=2022,
    patch_height=256, patch_width=128,
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    GPU=(DEVICE == "cuda")
)

# ---------------- PARAMETERS ----------------
MIN_BOX = 56
FEATURE_INTERVAL = 5
SIM_STRONG = 0.82
SIM_WEAK = 0.72
SCORE_MARGIN = 0.05
MIN_FEATURES = 5
CAM_REID_COOLDOWN = 2.0
MAX_IDLE_TIME = 30.0
MAX_SPEED = 7.0  # metric m/s if homography is metric

# ---------------- STORAGE ----------------
FEATURES = defaultdict(list)
TRACKS = defaultdict(lambda: {"track": [], "cameraIDs": set(), "gid": None})
GLOBAL_IDENTITIES = {}
NEXT_GID = 0
FEATURE_LOCK = threading.Lock()
GID_LOCK = threading.Lock()

# ---------------- UTILS ----------------
def l2_normalize(v):
    v = np.array(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-6)

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

def motion_check(last_pos, new_pos, dt):
    dist = np.linalg.norm(np.array(new_pos) - np.array(last_pos))
    speed = dist / (dt + 1e-6)
    return speed <= MAX_SPEED

# ---------------- GLOBAL ID ASSIGNMENT ----------------
def assign_global_id(cam_id, feat, pos, now, active_gids, feat_count):
    global NEXT_GID
    feat = l2_normalize(feat)
    candidates = []

    with GID_LOCK:
        for gid, data in GLOBAL_IDENTITIES.items():
            dt = now - data["last_seen"]
            if dt > MAX_IDLE_TIME:
                continue

            last_cam_time = data.get("last_seen_by_cam", {}).get(cam_id, -1e9)
            if now - last_cam_time < CAM_REID_COOLDOWN:
                continue

            if not motion_check(data.get("last_pos", pos), pos, dt):
                continue

            sim = cosine_sim(feat, data["mean_feat"])
            candidates.append((gid, sim))

        goto_new = True
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_gid, best_sim = candidates[0]
            second_sim = candidates[1][1] if len(candidates) > 1 else -1

            goto_new = (
                best_sim < SIM_WEAK or
                (best_sim < SIM_STRONG and feat_count < MIN_FEATURES) or
                (best_sim - second_sim) < SCORE_MARGIN or
                best_gid in active_gids
            )

        if not goto_new:
            g = GLOBAL_IDENTITIES[best_gid]
            g["mean_feat"] = l2_normalize(0.9 * g["mean_feat"] + 0.1 * feat)
            g["last_seen"] = now
            g["last_pos"] = pos
            g["cams"].add(cam_id)
            g.setdefault("last_seen_by_cam", {})[cam_id] = now
            g["count"] += 1
            return best_gid

        # Assign new GID
        gid = NEXT_GID
        NEXT_GID += 1
        GLOBAL_IDENTITIES[gid] = {
            "mean_feat": feat,
            "last_seen": now,
            "last_pos": pos,
            "cams": {cam_id},
            "last_seen_by_cam": {cam_id: now},
            "count": 1
        }
        return gid

# ---------------- CAMERA THREAD ----------------
class CameraThread(threading.Thread):
    def __init__(self, src, cam_id, stop_event):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(src)
        self.frame_queue = Queue(maxsize=5)
        self.frame_id = 0
        self.frame = None
        self.bt_to_sid = {}
        self.free_sids = deque()
        self.next_sid = 0
        self.tracks_local = {}
        self.active_gids = set()
        self.stop_event = stop_event
        threading.Thread(target=self.read_frames, daemon=True).start()

    def read_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01)

    def _get_sid(self, bt_id):
        if bt_id in self.bt_to_sid:
            return self.bt_to_sid[bt_id]
        sid = self.free_sids.popleft() if self.free_sids else self.next_sid
        if sid == self.next_sid:
            self.next_sid += 1
        self.bt_to_sid[bt_id] = sid
        return sid

    def run(self):
        global NEXT_GID
        while not self.stop_event.is_set() or not self.frame_queue.empty():
            if self.frame_queue.empty():
                time.sleep(0.001)
                continue

            frame = self.frame_queue.get()
            self.frame_id += 1
            now = time.time()
            self.active_gids.clear()

            with YOLO_LOCK:
                results = YOLO_MODEL.track(frame, conf=CONF_THRESH, imgsz=IMG_SIZE,
                                           classes=[0], persist=True,
                                           tracker="bytetrack.yaml", verbose=False)

            if not results or results[0].boxes is None or results[0].boxes.id is None:
                self.frame = frame
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, bt_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                if y2 - y1 < MIN_BOX:
                    continue

                sid = self._get_sid(bt_id)
                t = self.tracks_local.setdefault(bt_id, {"sid": sid, "last_feat": 0, "avg_feat": None, "feat_count": 0})

                feet = ((x1 + x2) / 2, y2)
                gx, gy = homography_transform(HOMOGRAPHIES[self.cam_id], feet)

                TRACKS[sid]["track"].append({"time": int(now * 1000), "x": gx, "y": gy})
                TRACKS[sid]["cameraIDs"].add(self.cam_id)

                # Feature extraction
                if self.frame_id - t["last_feat"] >= FEATURE_INTERVAL:
                    crop = frame[y1:y2, x1:x2]
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    feat = l2_normalize(osnet.get_features([crop])[0])

                    if t["avg_feat"] is None:
                        t["avg_feat"] = feat
                    else:
                        t["avg_feat"] = l2_normalize(0.9 * t["avg_feat"] + 0.1 * feat)

                    t["feat_count"] += 1

                    # Assign global ID only if enough features collected
                    if TRACKS[sid]["gid"] is None and t["feat_count"] >= MIN_FEATURES:
                        gid = assign_global_id(self.cam_id, t["avg_feat"], (gx, gy), now,
                                               self.active_gids, t["feat_count"])
                        TRACKS[sid]["gid"] = gid
                    else:
                        gid = TRACKS[sid]["gid"]

                    if gid is not None:
                        self.active_gids.add(gid)
                        with FEATURE_LOCK:
                            FEATURES[str(gid)].append(t["avg_feat"].tolist())

                    t["last_feat"] = self.frame_id

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if TRACKS[sid]["gid"] is not None:
                    cv2.putText(frame, f"GID {TRACKS[sid]['gid']}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            self.frame = frame

# ---------------- EXPORT ----------------
def export_results():
    print("[INFO] Exporting tracking results...")
    # Tracks by camera
    export = {}
    for cam_id, cam_name in enumerate(CAMERA_NAMES):
        export[cam_name] = {}
        for sid, data in TRACKS.items():
            if cam_id in data["cameraIDs"]:
                export[cam_name][str(sid)] = {
                    "track": data["track"],
                    "cameraIDs": list(data["cameraIDs"]),
                    "gid": data["gid"]
                }

    with open("tracks_by_camera.json", "w") as f:
        json.dump(export, f, indent=2)

    # Features
    with FEATURE_LOCK:
        safe_features = {k: v.copy() for k, v in FEATURES.items()}
    with open("features.json", "w") as f:
        json.dump(safe_features, f, indent=2)
    print("[INFO] Exported tracks_by_camera.json and features.json")

# ---------------- MAIN ----------------
def main():
    if not TRACKING_ENABLED:
        print("[INFO] Tracking disabled in config.json")
        return

    stop_event = threading.Event()
    cams = [CameraThread(src, cam_id, stop_event) for cam_id, src in enumerate(SOURCES)]
    for c in cams:
        c.start()

    try:
        while True:
            for c in cams:
                if c.frame is not None:
                    cv2.imshow(f"{CAMERA_NAMES[c.cam_id]}", cv2.resize(c.frame, (640, 360)))
            if cv2.waitKey(1) == 27:  # ESC to exit
                break
    except KeyboardInterrupt:
        pass

    # Signal threads to stop and wait
    stop_event.set()
    for c in cams:
        c.join()

    cv2.destroyAllWindows()
    export_results()

if __name__ == "__main__":
    main()
