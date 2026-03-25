import os
import json
import cv2
import numpy as np
import tracker
import yaml
from tracker import Undistorter
from collections import defaultdict
from scipy.optimize import linear_sum_assignment   # Hungarian

#########################
# CONFIG
#########################

INPUT_FILE = "tracks_events.jsonl"
OUTPUT_FILE = "fused_tracks.json"
INTRINSICS_FILE = "ANNKE_camera_intrinsics.yml"

SIM_THRESHOLD = 0.4
MAX_SPEED = 3000
MAX_GAP = 6000
MAX_DIST = 10000

#########################
# UTIL
#########################

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)


#########################
# 1. LOAD JSONL
#########################

def load_jsonl(path):
    from collections import defaultdict
    import json
    import numpy as np

    vectors = defaultdict(list)
    tracks = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            typ = obj.get("type")

            # ✅ ignore metadata like session_start
            if typ not in ("track", "vector"):
                continue

            mac = obj.get("mac")
            sid = obj.get("sid")

            if mac is None or sid is None:
                print(f"[WARN] malformed row {lineno}: {obj}")
                continue

            key = (mac.lower(), sid)

            if typ == "vector":
                vectors[key].append(np.asarray(obj["vector"], dtype=np.float32))

            else:  # track
                tracks[key].append({
                    "time": obj["time"],
                    "x": obj["x"],
                    "y": obj["y"],
                    "cam": mac.lower(),  
                    "sid": sid            
                })

    return vectors, tracks


#########################
# 2. INTRINSICS
#########################

def load_intrinsics(path):
    with open(path, "r") as f:
        d = yaml.safe_load(f)

    return Undistorter(
        K=np.array(d["camera_matrix"], float),
        dist=np.array(d["distortion_coefficients"], float),
        expected_size=(d["image_width"], d["image_height"]),
        intrinsics_path=os.path.abspath(path)
    )

def load_homographies(folder="."):
    """
    Loads OpenCV FileStorage homographies:
    <mac>_homography.yml
    """
    homos = {}

    for fname in os.listdir(folder):
        if not fname.lower().endswith("_homography.yml"):
            continue

        mac = fname.replace("_homography.yml", "").replace("_", ":").lower()
        path = os.path.join(folder, fname)

        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            continue

        H = fs.getNode("homography").mat()
        fs.release()

        if H is None or H.shape != (3, 3):
            continue

        homos[mac] = H.astype(np.float64)

    return homos

#########################
# 3. KALMAN FILTER
#########################

class Kalman2D:
    """
    Constant velocity Kalman filter
    state = [x, y, vx, vy]
    """

    def __init__(self):
        self.x = np.zeros((4, 1), dtype=float)
        self.P = np.eye(4) * 100.0

        self.F = np.array([
            [1,0,1,0],
            [0,1,0,1],
            [0,0,1,0],
            [0,0,0,1]
        ], dtype=float)

        self.H = np.array([
            [1,0,0,0],
            [0,1,0,0]
        ], dtype=float)

        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 5.0

    def update(self, z):
        z = np.asarray(z, dtype=float).reshape(2,1)

        # predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # update
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        # ✅ FIXED scalar extraction
        return float(self.x[0,0]), float(self.x[1,0])


#########################
# 4. BUILD TRACK OBJECTS
#########################

def build_track_objects(vectors, tracks):
    objs = []

    for key, vecs in vectors.items():
        evs = sorted(tracks[key], key=lambda e: e["time"])

        rep = normalize(np.mean(vecs, axis=0))

        objs.append({
            "cam": key[0],
            "sid": key[1],
            "rep": rep,
            "events": evs,
            "t_start": evs[0]["time"],
            "t_end": evs[-1]["time"],
        })

    objs.sort(key=lambda o: o["t_start"])
    return objs


#########################
# 5. HOMOGRAPHY + KALMAN
#########################

def apply_world_and_smooth(track_objs, homographies, undistorter):

    for track in track_objs:

        H = homographies.get(track["cam"])
        if H is None:
            continue

        kf = Kalman2D()

        for e in track["events"]:

            x, y = undistorter.undistort_point(
                e["x"], e["y"], undistorter.expected_size
            )

            p = H @ np.array([x,y,1.0])
            p /= p[2]

            # KALMAN SMOOTH
            xs, ys = kf.update((p[0], p[1]))

            e["x"], e["y"] = xs, ys

        last = track["events"][-1]
        track["x"], track["y"] = last["x"], last["y"]


#########################
# 6. PHYSICS
#########################

def teleport_metrics(gid, track):
    first = track["events"][0]

    dx = first["x"] - gid["x"]
    dy = first["y"] - gid["y"]

    dist = np.hypot(dx, dy)
    dt_ms = track["t_start"] - gid["t_end"]

    if dt_ms <= 0:
        return False, dist
    
    if dt_ms > MAX_GAP:
        return True, dist

    speed = dist / (dt_ms/1000)
    return speed > MAX_SPEED, dist


#########################
# 7. HUNGARIAN FUSION
#########################

def offline_fusion(track_objs):

    gids = []
    next_gid = 0

    for track in track_objs:

        if not gids:
            gids.append(new_gid(track, next_gid))
            next_gid += 1
            continue

        costs = []
        candidates = []

        for gid in gids:

            teleport, dist = teleport_metrics(gid, track)

            if teleport or dist > MAX_DIST:
                costs.append(1e6)
            else:
                sim = float(np.dot(track["rep"], gid["rep"]))
                costs.append(1 - sim)

            candidates.append(gid)

        cost_matrix = np.array(costs).reshape(1, -1)

        r, c = linear_sum_assignment(cost_matrix)

        best_cost = cost_matrix[r[0], c[0]]

        if best_cost > (1 - SIM_THRESHOLD):
            gids.append(new_gid(track, next_gid))
            next_gid += 1
        else:
            merge_gid(candidates[c[0]], track)

    return finalize(gids)


def new_gid(track, gid):
    return {
        "gid": gid,
        "rep": track["rep"].copy(),
        "count": 1,
        "t_end": track["t_end"],
        "x": track["x"],
        "y": track["y"],
        "tracks": [track["events"]],
    }


def merge_gid(gid, track):
    c = gid["count"]
    gid["rep"] = normalize((gid["rep"]*c + track["rep"]) / (c+1))
    gid["count"] += 1
    gid["t_end"] = track["t_end"]
    gid["x"], gid["y"] = track["x"], track["y"]
    gid["tracks"].append(track["events"])


def finalize(gids):
    for gid in gids:
        flat = [e for t in gid["tracks"] for e in t]
        gid["tracks"] = sorted(flat, key=lambda e: e["time"])

        seen = set()
        src = []
        for e in gid["tracks"]:
            k=(e["cam"],e["sid"])
            if k not in seen:
                seen.add(k)
                src.append({"cam":e["cam"],"sid":e["sid"]})
        gid["sources"]=src
    return gids


#########################
# 8. EXPORT
#########################

def export_gids(gids):
    out = {str(g["gid"]):{
        "sources":g["sources"],
        "tracks":g["tracks"]
    } for g in gids}

    with open(OUTPUT_FILE,"w") as f:
        json.dump(out,f,indent=2)


#########################
# MAIN
#########################

def main():
    vectors, tracks = load_jsonl(INPUT_FILE)
    track_objs = build_track_objects(vectors, tracks)

    undistorter = load_intrinsics(INTRINSICS_FILE)

    homographies = load_homographies(".")

    apply_world_and_smooth(track_objs, homographies, undistorter)

    gids = offline_fusion(track_objs)

    export_gids(gids)

    print("[DONE]")


if __name__ == "__main__":
    main()