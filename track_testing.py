"""
track_testing.py - Multi-camera person tracking with persistent IDs (torchreid + YOLO)
Usage:
    python track_testing.py rtsp://127.0.0.1:8554/cam0 rtsp://127.0.0.1:8554/cam1 ...
Requirements:
    pip install ultralytics opencv-python numpy scipy torchvision torchreid
"""

import os, cv2, time, queue, threading, numpy as np, sys
from datetime import datetime, timezone
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import torch
import torch.nn.functional as F
from torchvision import models, transforms

# ---------------- CONFIG ----------------
MODEL_PATH = 'yolov10m.pt'
CONF_THRESH = 0.35
MODEL_IMAGE_SIZE = 640
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_MISSED = 10
CROSS_COST_THRESHOLD = 0.55
PAST_MEMORY = 120.0
OUTPUT_CSV = 'positions.csv'
TRACK_DIR = 'tracks'
os.makedirs(TRACK_DIR, exist_ok=True)
DETECTOR_LOCK = threading.Lock()

# Box and text color
BOX_COLOR = (0, 255, 0)  # green
TEXT_COLOR = (0, 255, 0)

# ---------------- CAMERA SOURCES ----------------
SOURCES = {f'cam{i}': src for i, src in enumerate(sys.argv[1:])}
if not SOURCES:
    print("[ERROR] No camera sources provided!")
    sys.exit(1)

# ---------------- APPEARANCE MODEL ----------------
print(f"[INFO] DEVICE = {DEVICE}")
if DEVICE == 'cuda':
    print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")

try:
    import torchreid
    print("[INFO] Using torchreid OSNet model for cross-camera ID consistency...")
    reid_model = torchreid.models.build_model('osnet_x0_25', num_classes=1, pretrained=True)
    reid_model.to(DEVICE).eval()
except Exception as e:
    print(f"[WARN] torchreid not available ({e}); falling back to ResNet18.")
    reid_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(DEVICE).eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def cnn_feature(crop):
    if crop is None: return np.zeros(512)
    with torch.no_grad():
        x = preprocess(crop).unsqueeze(0).to(DEVICE)
        feat = reid_model(x)
        if isinstance(feat, (tuple, list)): feat = feat[0]
        feat = F.normalize(feat, dim=1)
    return feat.cpu().numpy().flatten()

def crop_safe(frame, box):
    h,w = frame.shape[:2]
    x1,y1,x2,y2 = int(max(0,box[0])),int(max(0,box[1])),int(min(w-1,box[2])),int(min(h-1,box[3]))
    if x2<=x1 or y2<=y1: return None
    return frame[y1:y2,x1:x2]

def appearance_distance(f1,f2):
    if f1 is None or f2 is None: return 1.0
    f1=f1/np.linalg.norm(f1); f2=f2/np.linalg.norm(f2)
    return 1.0 - np.clip(np.dot(f1,f2), -1.0, 1.0)

def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interW, interH = max(0, xB-xA), max(0, yB-yA)
    interArea = interW*interH
    areaA = max(1e-6, (boxA[2]-boxA[0])*(boxA[3]-boxA[1]))
    areaB = max(1e-6, (boxB[2]-boxB[0])*(boxB[3]-boxB[1]))
    return 0.0 if (areaA + areaB - interArea)<=0 else interArea/(areaA + areaB - interArea)

# ---------------- TRACK CLASSES ----------------
class Track:
    def __init__(self, local_id, bbox, feature, frame_idx, ts, cam_id):
        self.local_id = local_id
        self.cam_id = cam_id
        self.bboxes = [bbox]
        self.features = [feature]
        self.last_bbox = bbox
        self.first_seen = frame_idx
        self.last_seen = frame_idx
        self.first_time = ts
        self.last_time = ts
        self.missed = 0
        self.global_id = None
    def update(self,bbox,feature,frame_idx,ts):
        self.bboxes.append(bbox)
        self.features.append(feature)
        self.last_bbox = bbox
        self.last_seen = frame_idx
        self.last_time = ts
        self.missed = 0
    def mark_missed(self): self.missed +=1

class SimpleTracker:
    def __init__(self, cam_id, appearance_weight=0.75):
        self.cam_id = cam_id
        self.appearance_weight = appearance_weight
        self.next_local_id = 0
        self.tracks = []
    def _create_track(self,bbox,feature,frame_idx,ts):
        t = Track(self.next_local_id,bbox,feature,frame_idx,ts,self.cam_id)
        self.next_local_id += 1
        self.tracks.append(t)
        return t
    def update(self,detections,frame,frame_idx,ts):
        det_boxes = [d[:4] for d in detections]
        det_feats = [cnn_feature(crop_safe(frame,b)) for b in det_boxes]
        N,M = len(self.tracks), len(det_boxes)
        if N==0:
            for i in range(M): self._create_track(det_boxes[i],det_feats[i],frame_idx,ts)
        else:
            cost = np.zeros((N,M))
            for i,tr in enumerate(self.tracks):
                for j,b in enumerate(det_boxes):
                    iou_cost = 1.0 - iou(tr.last_bbox,b)
                    app_cost = appearance_distance(tr.features[-1],det_feats[j])
                    cost[i,j] = 0.25*iou_cost + 0.75*app_cost
            row_ind,col_ind = linear_sum_assignment(cost)
            assigned_tracks,assigned_dets = set(),set()
            for r,c in zip(row_ind,col_ind):
                if cost[r,c]<0.7:
                    self.tracks[r].update(det_boxes[c],det_feats[c],frame_idx,ts)
                    assigned_tracks.add(r)
                    assigned_dets.add(c)
            for j in range(M):
                if j not in assigned_dets:
                    self._create_track(det_boxes[j],det_feats[j],frame_idx,ts)
            for i in range(N):
                if i not in assigned_tracks: self.tracks[i].mark_missed()
        self.tracks = [t for t in self.tracks if t.missed<=MAX_MISSED]
        return self.tracks

# ---------------- GLOBAL TRACKER ----------------
# ---------------- GLOBAL TRACKER ----------------
class GlobalTracker:
    def __init__(self, history_len=5):
        self.global_tracks = {}  # gid -> {'features':[...], 'last_seen':timestamp, 'cam_id': cam}
        self.next_global_id = 0
        self.recent_memory = PAST_MEMORY
        self.history_len = history_len

    def associate(self, active_tracks):
        now = time.time()
        for t in active_tracks:
            best_gid = None
            best_dist = 1.0

            # compare with all existing global tracks
            for gid, data in self.global_tracks.items():
                # skip tracks expired from memory
                if now - data['last_seen'] > self.recent_memory:
                    continue
                # skip tracks in the same camera with different local IDs
                if data['cam_id'] == t.cam_id and any(tr.local_id == t.local_id for tr in active_tracks):
                    continue
                # compare against all past features
                for f in data['features']:
                    dist = appearance_distance(t.features[-1], f)
                    if dist < best_dist:
                        best_dist = dist
                        best_gid = gid

            # assign global ID if good match found
            if best_gid is not None and best_dist < CROSS_COST_THRESHOLD:
                t.global_id = best_gid
                data = self.global_tracks[best_gid]
                data['last_seen'] = now
                data['cam_id'] = t.cam_id
                data['features'].append(t.features[-1])
                if len(data['features']) > self.history_len:
                    data['features'].pop(0)
            else:
                # create new global ID
                gid = self.next_global_id
                t.global_id = gid
                self.global_tracks[gid] = {
                    'features': [t.features[-1]],
                    'last_seen': now,
                    'cam_id': t.cam_id
                }
                self.next_global_id += 1


# ---------------- CAMERA WORKER ----------------
class CameraWorker(threading.Thread):
    def __init__(self,cam_id,src,detector,out_q,stop_event,scale=0.45):
        super().__init__(daemon=True)
        self.cam_id,self.src,self.detector,self.out_q,self.stop_event,self.scale = cam_id,src,detector,out_q,stop_event,scale
        self.tracker = SimpleTracker(cam_id)
        self.frame_buffer = queue.Queue(maxsize=2)
        self._frame_counter = 0
    def _reader_loop(self):
        cap = cv2.VideoCapture(self.src)
        while not self.stop_event.is_set():
            ret,frame = cap.read()
            if not ret: time.sleep(0.01); continue
            ts = time.time()
            idx = self._frame_counter
            self._frame_counter += 1
            if self.frame_buffer.full():
                try: self.frame_buffer.get_nowait()
                except: pass
            try: self.frame_buffer.put_nowait((idx,ts,frame))
            except: pass
        cap.release()
    def run(self):
        reader = threading.Thread(target=self._reader_loop,daemon=True)
        reader.start()
        while not self.stop_event.is_set():
            try: idx,ts,frame=self.frame_buffer.get(timeout=1)
            except queue.Empty: continue
            with DETECTOR_LOCK:
                results=self.detector.predict(frame,imgsz=MODEL_IMAGE_SIZE,device=DEVICE,half=DEVICE=='cuda',verbose=False)
            dets=[]
            for box in results[0].boxes:
                cls=int(box.cls[0]); conf=float(box.conf[0])
                if cls!=0 or conf<CONF_THRESH: continue
                dets.append((*(box.xyxy[0].tolist()),conf))
            tracks = self.tracker.update(dets,frame,idx,ts)
            vis = cv2.resize(frame,(0,0),fx=self.scale,fy=self.scale)
            for t in tracks:
                x1,y1,x2,y2 = map(int,t.last_bbox)
                x1=int(x1*self.scale); y1=int(y1*self.scale)
                x2=int(x2*self.scale); y2=int(y2*self.scale)
                gid = t.global_id if t.global_id is not None else -1
                cv2.rectangle(vis,(x1,y1),(x2,y2),BOX_COLOR,2)
                cv2.putText(vis,f"L{t.local_id}/G{gid}",(x1,max(15,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.5,TEXT_COLOR,2)
            self.out_q.put({'cam':self.cam_id,'frame_idx':idx,'timestamp':ts,'active_tracks':tracks,'vis_frame':vis})
        reader.join(timeout=2)

# ---------------- CSV ----------------
def write_csv_header():
    with open(OUTPUT_CSV,'w') as f:
        f.write('global_id,camera_id,frame_idx,timestamp_iso,x1,y1,x2,y2,confidence\n')

def append_positions_csv(global_id,cam_id,frame_idx,ts,bbox,conf=1.0):
    iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    x1,y1,x2,y2 = bbox
    with open(OUTPUT_CSV,'a') as f:
        f.write(f"{global_id},{cam_id},{frame_idx},{iso},{int(x1)},{int(y1)},{int(x2)},{int(y2)},{conf}\n")

# ---------------- GRID ----------------
def make_grid(frames, target_size=(640,360)):
    cams = list(frames.keys())
    n = len(cams)
    cols = min(2,n)
    rows = (n+cols-1)//cols
    resized = [cv2.resize(frames[cam], target_size) for cam in cams]
    while len(resized)<rows*cols:
        resized.append(np.zeros((target_size[1],target_size[0],3),dtype=np.uint8))
    grid_rows = [np.hstack(resized[i*cols:(i+1)*cols]) for i in range(rows)]
    return np.vstack(grid_rows)

# ---------------- MAIN ----------------
def main():
    detector = YOLO(MODEL_PATH)
    q=queue.Queue(); stop_event=threading.Event()
    workers={}
    for cam,src in SOURCES.items():
        cap=cv2.VideoCapture(src)
        if cap.isOpened():
            w = CameraWorker(cam,src,detector,q,stop_event,scale=0.45)
            workers[cam] = w; w.start()
            print(f"[INFO] Started worker for {cam}")
        cap.release()

    manager = GlobalTracker()
    write_csv_header()
    latest_frames={}

    try:
        while not stop_event.is_set():
            try: item = q.get(timeout=1)
            except queue.Empty: continue
            latest_frames[item['cam']] = item['vis_frame']
            active_tracks = item['active_tracks']
            manager.associate(active_tracks)
            for t in active_tracks:
                append_positions_csv(t.global_id,t.cam_id,t.last_seen,t.last_time,t.last_bbox)
            if latest_frames:
                grid = make_grid(latest_frames)
                if grid is not None:
                    cv2.imshow("Camera Grid", grid)
                    if cv2.waitKey(1)&0xFF==ord('q'):
                        stop_event.set()
                        break
    except KeyboardInterrupt: stop_event.set()
    for w in workers.values(): w.join()
    cv2.destroyAllWindows()
    print("Done:", OUTPUT_CSV)

if __name__=="__main__":
    main()
