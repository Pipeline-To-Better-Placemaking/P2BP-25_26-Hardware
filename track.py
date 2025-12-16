# ============================================================
# Jetson Orin Nano – Multi-Camera Person Tracking + ReID
# YOLOv10 + OSNet + Local Tracking + Global ID Fusion
# ============================================================

import os, sys, cv2, time, torch, threading
import numpy as np
from ultralytics import YOLO

# ---------------- PATH FIX ----------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ---------------- CONFIG ----------------
MODEL_PATH = "yolov10n.pt"
CONF_THRESH = 0.35
IMG_SIZE = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REID_INTERVAL = 1.0
COS_THRESHOLD = 0.45
MIN_BOX = 48

# ---------------- INPUT ----------------
SOURCES = sys.argv[1:]
if not SOURCES:
    print("Usage: python track.py <cam0> <cam1> ...")
    sys.exit(1)

# ---------------- OSNET ----------------
from osnet_ibn1_lite.encoder import OsNetEncoder

osnet = OsNetEncoder(
    input_width=640,
    input_height=480,
    weight_filepath="osnet_ibn1_lite/model_weights.pth.tar-40",
    batch_size=1,
    num_classes=2022,
    patch_height=256,
    patch_width=128,
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    GPU=(DEVICE == "cuda")
)
print("[INFO] OSNet loaded")

# ---------------- UTILS ----------------
def cosine_dist(a, b):
    return 1.0 - float(np.dot(a, b))

# ---------------- GLOBAL ID MANAGER ----------------
class GlobalIDManager:
    def __init__(self):
        self.next_id = 0
        self.features = {}
        self.lock = threading.Lock()

    def assign(self, feature):
        with self.lock:
            best_d, best_id = 1e9, None
            for gid, f in self.features.items():
                d = cosine_dist(feature, f)
                if d < best_d:
                    best_d, best_id = d, gid

            if best_id is not None and best_d < COS_THRESHOLD:
                self.features[best_id] = (
                    self.features[best_id] * 0.7 + feature * 0.3
                )
                self.features[best_id] /= np.linalg.norm(self.features[best_id])
                return best_id

            gid = self.next_id
            self.features[gid] = feature
            self.next_id += 1
            return gid

# ---------------- CAMERA THREAD ----------------
class CameraThread(threading.Thread):
    def __init__(self, src, cam_id, gid_manager):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src)
        self.cam_id = cam_id
        self.model = YOLO(MODEL_PATH)
        self.gid_manager = gid_manager
        self.frame = None

        self.local_tracks = {}
        self.next_local_id = 0

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model.predict(
                frame, conf=CONF_THRESH, imgsz=IMG_SIZE, classes=[0]
            )

            new_tracks = {}
            for r in results:
                for box in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    crop = frame[y1:y2, x1:x2]
                    if crop.shape[0] < MIN_BOX:
                        continue

                    # new local track
                    lid = self.next_local_id
                    self.next_local_id += 1

                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    feat = osnet.get_features([rgb])[0]
                    feat = feat / (np.linalg.norm(feat) + 1e-12)

                    gid = self.gid_manager.assign(feat)
                    new_tracks[lid] = gid

                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(
                        frame, f"GID {gid}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2
                    )

            self.frame = frame

# ---------------- GRID DISPLAY ----------------
def show_grid(threads):
    while True:
        frames = [t.frame for t in threads if t.frame is not None]
        if not frames:
            time.sleep(0.01)
            continue

        h, w = frames[0].shape[:2]
        resized = [cv2.resize(f, (w//2, h//2)) for f in frames]

        while len(resized) % 2 != 0:
            resized.append(np.zeros_like(resized[0]))

        rows = []
        for i in range(0, len(resized), 2):
            rows.append(np.hstack(resized[i:i+2]))

        grid = np.vstack(rows)
        cv2.imshow("Multi-Camera View", grid)

        if cv2.waitKey(1) & 0xFF == 27:
            break

# ---------------- MAIN ----------------
def main():
    gid_manager = GlobalIDManager()
    threads = []

    for i, src in enumerate(SOURCES):
        t = CameraThread(src, i, gid_manager)
        t.start()
        threads.append(t)

    show_grid(threads)

if __name__ == "__main__":
    main()
