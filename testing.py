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
TRACK_TTL = 2.0
YOLO_INTERVAL = 1  # detect every frame

# ---------------- INPUT / CAMERAS ----------------
SOURCES = [
    "rtsp://127.0.0.1:8554/cam0",
    "rtsp://127.0.0.1:8554/cam1"
]

HOMOGRAPHY_FILES = [
    "homographies/4p-c0-homography.yml",
    "homographies/4p-c1-homography.yml"
]

# ---------------- HOMOGRAPHY LOADING ----------------
def load_homography(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    H = np.array(data.get('homography'))
    if H.shape != (3,3):
        raise ValueError(f"Homography {path} must be 3x3")
    return H

HOMOGRAPHIES = [load_homography(path) for path in HOMOGRAPHY_FILES]

def homography_transform(H, point):
    x, y = point
    p = np.array([x, y, 1.0])
    p_trans = H @ p
    p_trans /= p_trans[2]
    return float(p_trans[0]), float(p_trans[1])

# ---------------- MODELS ----------------
YOLO_MODEL = YOLO(MODEL_PATH)

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
print("[INFO] OSNet loaded")

# ---------------- STORAGE ----------------
FEATURES = defaultdict(lambda: defaultdict(list))  # cam_id -> sid -> features
TRACKS = defaultdict(lambda: {"track": [], "cameraIDs": set()})
FEATURE_LOCK = threading.Lock()

# ---------------- CAMERA THREAD ----------------
class CameraThread(threading.Thread):
    def __init__(self, src, cam_id):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(src)
        self.frame_queue = Queue(maxsize=5)  # buffer for lag reduction
        self.frame_id = 0
        self.frame = None

        self.bt_to_sid = {}
        self.free_sids = deque()
        self.next_sid = 0
        self.tracks = {}

        # Start RTSP reader thread
        self.reader_thread = threading.Thread(target=self.read_frames, daemon=True)
        self.reader_thread.start()

    def read_frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                continue
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
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

            # YOLO + ByteTrack detection
            try:
                # Remove stream=True to get a list, not a generator
                results = YOLO_MODEL.track(
                    frame,
                    conf=CONF_THRESH,
                    imgsz=IMG_SIZE,
                    classes=[0],
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False
                )
            except Exception as e:
                print(f"[WARN] YOLO track failed: {e}")
                self.frame = frame
                continue

            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                self.frame = frame
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            bt_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, bt_id in zip(boxes, bt_ids):
                x1, y1, x2, y2 = map(int, box)
                if y2 - y1 < MIN_BOX:
                    continue

                sid = self._get_sid(bt_id)
                t = self.tracks.get(bt_id)
                if t is None:
                    t = self.tracks[bt_id] = {"sid": sid, "last": now, "last_feat": 0}
                t["last"] = now

                feet_px = ((x1 + x2)/2, y2)
                gx, gy = homography_transform(HOMOGRAPHIES[self.cam_id], feet_px)

                TRACKS[sid]["track"].append({
                    "time": int(now*1000),
                    "x": gx,
                    "y": gy
                })
                TRACKS[sid]["cameraIDs"].add(self.cam_id)

                # Feature extraction
                if self.frame_id - t["last_feat"] >= FEATURE_INTERVAL:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue  # skip invalid crops
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    try:
                        feat = osnet.get_features([crop])[0]
                        if feat is not None:
                            with FEATURE_LOCK:
                                FEATURES[self.cam_id][sid].append(feat.tolist())
                    except Exception as e:
                        print(f"[WARN] Feature extraction failed: {e}")
                    t["last_feat"] = self.frame_id

                # Draw rectangle & SID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"Cam {self.cam_id} | SID {sid}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Cleanup lost tracks
            for bt_id in list(self.tracks.keys()):
                if now - self.tracks[bt_id]["last"] > TRACK_TTL:
                    sid = self.tracks[bt_id]["sid"]
                    del self.tracks[bt_id]
                    del self.bt_to_sid[bt_id]
                    self.free_sids.append(sid)

            self.frame = frame


# ---------------- MAIN ----------------
def main():
    cams = [CameraThread(src, cam_id) for cam_id, src in enumerate(SOURCES)]
    for c in cams: 
        c.start()

    try:
        while True:
            for c in cams:
                if c.frame is not None:
                    disp = cv2.resize(c.frame, (640, 360))
                    cv2.imshow(f"Camera {c.cam_id}", disp)
            if cv2.waitKey(1) == 27:
                break
            time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()

        # Export JSON by camera → ID
        export = {}
        for cam_id, cam in enumerate(cams):
            export[f"camera_{cam_id}"] = {}
            for sid in TRACKS:
                if cam_id in TRACKS[sid]["cameraIDs"]:
                    export[f"camera_{cam_id}"][str(sid)] = {
                        "track": [pt for pt in TRACKS[sid]["track"] if cam_id in TRACKS[sid]["cameraIDs"]],
                        "pathLength": len(TRACKS[sid]["track"]),
                        "timeSpent": int(TRACKS[sid]["track"][-1]["time"] - TRACKS[sid]["track"][0]["time"]) if TRACKS[sid]["track"] else 0,
                        "vectors": FEATURES[cam_id].get(sid, [])
                    }

        with open("tracks_by_camera.json", "w") as f:
            json.dump(export, f, indent=2)
        print("[INFO] Tracks saved to tracks_by_camera.json")

if __name__ == "__main__":
    main()
