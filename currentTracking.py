import os, cv2, time, torch, threading, json
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import yaml
from osnet_ibn1_lite.encoder import OsNetEncoder
from queue import Queue

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

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
IMG_SIZE = 640
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

# ---------------- STORAGE ----------------
TRACKS = defaultdict(lambda: {"track": [], "features": [], "cameraIDs": set()})
FEATURE_LOCK = threading.Lock()

# ---------------- CAMERA THREAD ----------------
class CameraThread(threading.Thread):
    def __init__(self, src, cam_id, stop_event):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame_queue = Queue(maxsize=5)
        self.frame = None
        self.frame_id = 0
        self.stop_event = stop_event

        self.local_id_map = {}
        self.next_local_id = 0

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

    def get_stable_id(self, bt_id):
        if bt_id not in self.local_id_map:
            self.local_id_map[bt_id] = self.next_local_id
            self.next_local_id += 1
        return self.local_id_map[bt_id]

    def run(self):
        while not self.stop_event.is_set() or not self.frame_queue.empty():
            if self.frame_queue.empty():
                time.sleep(0.001)
                continue

            frame = self.frame_queue.get()
            self.frame_id += 1
            now = int(time.time() * 1000)

            with YOLO_LOCK:
                results = YOLO_MODEL.track(frame, conf=CONF_THRESH, imgsz=IMG_SIZE,
                                           classes=[0], persist=True,
                                           tracker="bytetrack.yaml", verbose=False)

            if not results or results[0].boxes is None or results[0].boxes.id is None:
                self.frame = frame
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            # Only process every 5 frames to align track and features
            if self.frame_id % 10 == 0:
                for box, bt_id in zip(boxes, ids):
                    tid = self.get_stable_id(bt_id)

                    x1, y1, x2, y2 = map(int, box)
                    if y2 - y1 < 50:  # filter too small boxes
                        continue

                    feet = ((x1 + x2) / 2, y2)
                    gx, gy = homography_transform(HOMOGRAPHIES[self.cam_id], feet)

                    # Append track point
                    TRACKS[(self.cam_id, tid)]["track"].append({"time": now, "x": gx, "y": gy})
                    TRACKS[(self.cam_id, tid)]["cameraIDs"].add(self.cam_id)

                    # Extract feature
                    crop = frame[y1:y2, x1:x2]
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    feat = osnet.get_features([crop])[0].tolist()
                    with FEATURE_LOCK:
                        TRACKS[(self.cam_id, tid)]["features"].append({"time": now, "feature": feat})

            # Draw bounding boxes for visualization
            for box, bt_id in zip(boxes, ids):
                tid = self.get_stable_id(bt_id)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {tid}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            self.frame = frame

# ---------------- EXPORT ----------------

def export_results():
    print("[INFO] Exporting results...")
    export = {}

    # Sort by camera order first
    for cam_idx, cam_name in enumerate(CAMERA_NAMES):
        cam_tracks = {tid: data for (c_id, tid), data in TRACKS.items() if c_id == cam_idx}
        if not cam_tracks:
            continue

        # Sort by track ID
        sorted_tracks = dict(sorted(cam_tracks.items(), key=lambda x: int(x[0])))

        export[cam_name] = {}
        for tid, data in sorted_tracks.items():
            export[cam_name][str(tid)] = {
                "track": data["track"],
                "features": data["features"],
                "cameraIDs": list(data["cameraIDs"])
            }

    with open("tracks_simple.json", "w") as f:
        json.dump(export, f, indent=2)

    print("[INFO] Exported tracks_simple.json")


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
                    cv2.imshow(CAMERA_NAMES[c.cam_id], cv2.resize(c.frame, (640, 360)))
            if cv2.waitKey(1) == 27:
                break
    except KeyboardInterrupt:
        pass

    stop_event.set()
    for c in cams:
        c.join()

    cv2.destroyAllWindows()
    export_results()

if __name__ == "__main__":
    main()
