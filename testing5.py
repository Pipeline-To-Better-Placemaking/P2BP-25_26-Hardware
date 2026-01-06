import os, cv2, time, torch, threading, json
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import yaml
from osnet_ibn1_lite.encoder import OsNetEncoder
from queue import Queue

# ---------------- CONFIG ----------------
MODEL_PATH = "yolov10n.pt"
CONF_THRESH = 0.35
IMG_SIZE = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_BOX = 56
FEATURE_INTERVAL = 5
TRACK_TTL = 2.0  # seconds for local track expiration

SIM_THRESH = 0.7
MAX_IDLE_TIME = 30.0  # seconds for global GID expiration
MAX_SPEED = 10.0  # pixels/ms allowed for motion matching

# ---------------- INPUT / CAMERAS ----------------
CAMERA_CONFIG_FILE = "config/cameras_runtime.json"

def load_camera_sources(config_path):
    with open(config_path, 'r') as f:
        cams = json.load(f)
    sources, cam_names = [], []
    for name, cam in cams.items():
        sources.append(cam["rtsp"])
        cam_names.append(name)
    return sources, cam_names

SOURCES, CAMERA_NAMES = load_camera_sources(CAMERA_CONFIG_FILE)
print(f"[INFO] Loaded cameras: {CAMERA_NAMES}")

# ---------------- HOMOGRAPHY ----------------
def load_homography(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return np.array(data.get('homography'))

HOMOGRAPHY_FILES = [f"homographies/{name}-homography.yml" for name in CAMERA_NAMES]
HOMOGRAPHIES = [load_homography(p) for p in HOMOGRAPHY_FILES]

def homography_transform(H, point):
    x, y = point
    p = np.array([x, y, 1.0])
    p = H @ p
    p /= p[2]
    return float(p[0]), float(p[1])

# ---------------- MODELS ----------------
YOLO_MODEL = YOLO(MODEL_PATH)
YOLO_LOCK = threading.Lock()

osnet = OsNetEncoder(
    input_width=704,
    input_height=480,
    weight_filepath="osnet_ibn1_lite/model_weights.pth.tar-40",
    batch_size=32,
    num_classes=2022,
    patch_height=256,
    patch_width=128,
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    GPU=(DEVICE=="cuda")
)

# ---------------- STORAGE ----------------
FEATURES = defaultdict(list)  # gid -> list of features
TRACKS = defaultdict(lambda: {"track": [], "cameraIDs": set(), "gid": None})

GLOBAL_IDENTITIES = {}  # gid -> {mean_feat, last_seen, cams, count}
NEXT_GID = 0

FEATURE_LOCK = threading.Lock()
GID_LOCK = threading.Lock()

# ---------------- UTILS ----------------
def l2_normalize(v):
    v = np.array(v)
    return v / (np.linalg.norm(v) + 1e-6)

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

def motion_check(last_pos, new_pos, dt):
    dx = new_pos[0] - last_pos[0]
    dy = new_pos[1] - last_pos[1]
    dist = np.sqrt(dx**2 + dy**2)
    speed = dist / (dt*1000 + 1e-6)  # pixels/ms
    return speed <= MAX_SPEED

# ---------------- GLOBAL ID ASSIGNMENT ----------------
def assign_global_id(cam_id, feat, pos, now):
    global NEXT_GID
    feat = l2_normalize(feat)
    best_gid, best_score = None, -1

    with GID_LOCK:
        for gid, data in GLOBAL_IDENTITIES.items():
            dt = now - data["last_seen"]
            if dt > MAX_IDLE_TIME:
                continue
            if not motion_check(data["last_pos"], pos, dt):
                continue
            score = cosine_sim(feat, data["mean_feat"])
            if score > best_score:
                best_score, best_gid = score, gid

        if best_score > SIM_THRESH:
            g = GLOBAL_IDENTITIES[best_gid]
            g["mean_feat"] = l2_normalize(0.9 * g["mean_feat"] + 0.1 * feat)
            g["last_seen"] = now
            g["last_pos"] = pos
            g["cams"].add(cam_id)
            g["count"] += 1
            return best_gid
        else:
            gid = NEXT_GID
            NEXT_GID += 1
            GLOBAL_IDENTITIES[gid] = {
                "mean_feat": feat,
                "last_seen": now,
                "last_pos": pos,
                "cams": {cam_id},
                "count": 1
            }
            return gid

# ---------------- CAMERA THREAD ----------------
class CameraThread(threading.Thread):
    def __init__(self, src, cam_id):
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
        threading.Thread(target=self.read_frames, daemon=True).start()

    def read_frames(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame)

    def _get_sid(self, bt_id):
        if bt_id in self.bt_to_sid:
            return self.bt_to_sid[bt_id]
        sid = self.free_sids.popleft() if self.free_sids else self.next_sid
        if sid == self.next_sid:
            self.next_sid += 1
        self.bt_to_sid[bt_id] = sid
        return sid

    def run(self):
        while True:
            if self.frame_queue.empty():
                time.sleep(0.001)
                continue

            frame = self.frame_queue.get()
            self.frame_id += 1
            now = time.time()

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
                t = self.tracks_local.get(bt_id)
                if t is None:
                    t = self.tracks_local[bt_id] = {"sid": sid, "last": now, "last_feat": 0, "feats": [], "last_pos": None}
                t["last"] = now

                feet = ((x1+x2)/2, y2)
                gx, gy = homography_transform(HOMOGRAPHIES[self.cam_id], feet)

                TRACKS[sid]["track"].append({"time": int(now*1000), "x": gx, "y": gy})
                TRACKS[sid]["cameraIDs"].add(self.cam_id)

                if self.frame_id - t["last_feat"] >= FEATURE_INTERVAL:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    feat = osnet.get_features([crop])[0].tolist()

                    t["feats"].append(feat)
                    if len(t["feats"]) > 20:
                        t["feats"].pop(0)

                    avg_feat = np.mean(t["feats"], axis=0)
                    gid = assign_global_id(self.cam_id, avg_feat, (gx, gy), now)
                    TRACKS[sid]["gid"] = gid

                    with FEATURE_LOCK:
                        FEATURES[str(gid)].append([float(x) for x in feat])  # convert to list

                    t["last_feat"] = self.frame_id
                    t["last_pos"] = (gx, gy)

                gid = TRACKS[sid]["gid"]
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"GID {gid}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            self.frame = frame

# ---------------- MAIN ----------------
def main():
    cams = [CameraThread(src, cam_id) for cam_id, src in enumerate(SOURCES)]
    for c in cams:
        c.start()

    while True:
        for c in cams:
            if c.frame is not None:
                cv2.imshow(f"{CAMERA_NAMES[c.cam_id]}", cv2.resize(c.frame, (640,360)))
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

    export = {}
    for cam_id, cam_name in enumerate(CAMERA_NAMES):
        export[cam_name] = {}
        for sid, data in TRACKS.items():
            if cam_id in data["cameraIDs"]:
                export[cam_name][str(sid)] = {
                    "track": [{"time": t["time"], "x": float(t["x"]), "y": float(t["y"])} for t in data["track"]],
                    "cameraIDs": list(data["cameraIDs"]),
                    "gid": data["gid"]
                }

    with open("tracks_by_camera.json", "w") as f:
        json.dump(export, f, indent=2)

    with open("features.json", "w") as f:
        json.dump(FEATURES, f, indent=2)

    print("[INFO] Saved tracks_by_camera.json and features.json")

if __name__ == "__main__":
    main()
