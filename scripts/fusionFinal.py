
import os
import json
import cv2
import yaml
import numpy as np
from tracker import Undistorter
from collections import defaultdict
from scipy.spatial.distance import cosine, euclidean

# =========================
# CONFIG
# =========================

INPUT_FILE = "tracks_events.jsonl"
OUTPUT_FILE = "fused_tracks.json"
INTRINSICS_FILE = "ANNKE_camera_intrinsics.yml"

MAX_GAP_S = 5.0
MAX_POSITION_PX = 120
MIN_APPEAR_SIM = 0.60

DUPLICATE_SIM = 0.68
MAX_DUPLICATE_OVERLAP_S = 3.0
MAX_OVERLAP_S = 2.0

MAX_SPEED = 3000
MAX_DIST = 10000

MIN_TRACK_POINTS = 5
MIN_DURATION = 0.4
MAX_JUMP_CLEAN = 3000
DUP_EPS = 1e-3
SMOOTH_WIN = 2

# =========================
# UTIL
# =========================

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)

def cos_sim(a, b):
    return float(1 - cosine(a, b))

def bbox_center(b):
    return np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])

def euclid(a, b):
    return float(euclidean(a, b))

def _endpoint_sim(self, td1, td2):
        """
        Similarity between the tail of track1 and the head of track2.
        Averages the last 3 features of td1 vs the first 3 of td2 for
        robustness against single bad crops.
        """
        tail = td1['features'][-min(3, len(td1['features'])):]
        head = td2['features'][:min(3, len(td2['features']))]
        sims = [
            self._cosine_sim(
                np.array(a['feature_vector']),
                np.array(b['feature_vector'])
            )
            for a in tail for b in head
        ]
        return float(np.median(sims)) if sims else 0.0

# =========================
# LOAD JSONL
# =========================

def load_jsonl(path):
    vectors = defaultdict(list)
    tracks = defaultdict(list)

    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)

            if obj.get("type") not in ("track", "vector"):
                continue

            key = (obj["mac"].lower(), obj["sid"])

            if obj["type"] == "vector":
                vectors[key].append(np.asarray(obj["vector"], dtype=np.float32))
            else:
                obj["cam"] = obj["mac"].lower()
                tracks[key].append(obj)

    return vectors, tracks

# =========================
# INTRINSICS + HOMOGRAPHY
# =========================

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

# =========================
# UNDISTORTION (ONLY ADDITION)
# =========================

def undistort_point(undistorter, x, y):
    return undistorter.undistort_point(
        x, y, undistorter.expected_size
    )

# =========================
# KALMAN (UNCHANGED)
# =========================

class Kalman2D:
    def __init__(self):
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 100

        self.F = np.eye(4)
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], float)

        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 5.0

    def update(self, z, dt):
        z = np.asarray(z).reshape(2, 1)

        self.F = np.array([
            [1,0,dt,0],
            [0,1,0,dt],
            [0,0,1,0],
            [0,0,0,1]
        ], float)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return float(self.x[0,0]), float(self.x[1,0])

# =========================
# BUILD TRACK OBJECTS
# =========================

def build_tracks(vectors, tracks):
    objs = []

    for key, vecs in vectors.items():
        evs = sorted(tracks[key], key=lambda e: e["time"])
        rep = normalize(np.mean(vecs, axis=0))

        objs.append({
            "key": key,
            "rep": rep,
            "events": evs,
            "t_start": evs[0]["time"],
            "t_end": evs[-1]["time"],
            "x": evs[-1]["x"],
            "y": evs[-1]["y"]
        })

    return sorted(objs, key=lambda x: x["t_start"])

# =========================
# WORLD TRANSFORM (FIXED ONLY HERE)
# =========================

def apply_world(tracks, homographies, undistorter):
    for t in tracks:
        cam = t["key"][0]

        H = homographies.get(cam)
        

        if H is None:
            continue

        

        kf = Kalman2D()
        prev_time = None

        for e in t["events"]:

            # =========================
            #  UNDISTORT 
            # =========================
            ux, uy = undistort_point(undistorter, e["x"], e["y"])

            # =========================
            # HOMOGRAPHY (UNCHANGED)
            # =========================
            p = np.array([ux, uy, 1.0])
            w = H @ p
            w /= w[2]

            # =========================
            # KALMAN (UNCHANGED)
            # =========================
            if prev_time is None:
                dt = 0.0
            else:
                dt = max(1e-3, e["time"] - prev_time)

            x, y = kf.update((w[0], w[1]), dt)

            e["x"], e["y"] = x, y
            prev_time = e["time"]

        last = t["events"][-1]
        t["x"], t["y"] = last["x"], last["y"]

# =========================
# CORE FUNCTIONS (UNCHANGED)
# =========================

def spatial_gap(a, b):
    ax, ay = a["events"][-1]["x"], a["events"][-1]["y"]
    bx, by = b["events"][0]["x"], b["events"][0]["y"]
    return np.hypot(ax - bx, ay - by)

def time_range(t):
    ts = [e["time"] for e in t["events"]]
    return min(ts), max(ts)

def overlap(a, b):
    s1, e1 = time_range(a)
    s2, e2 = time_range(b)
    return max(0, min(e1, e2) - max(s1, s2))

def endpoint_sim(a, b):
    """
    Robust endpoint similarity:
    compare last 3 features of A with first 3 of B
    """
    tail = a.get("features", [])[-min(3, len(a.get("features", []))):]
    head = b.get("features", [])[:min(3, len(b.get("features", [])))]

    if not tail or not head:
        return cos_sim(a["rep"], b["rep"])  # fallback

    sims = []
    for ta in tail:
        for hb in head:
            if "feature_vector" in ta and "feature_vector" in hb:
                sims.append(cos_sim(
                    np.array(ta["feature_vector"]),
                    np.array(hb["feature_vector"])
                ))

    return float(np.median(sims)) if sims else 0.0

# =========================
# EXCLUSION SYSTEM (UNCHANGED)
# =========================

def build_exclusions(tracks):
    excluded = set()
    duplicates = set()

    keys = list(tracks.keys())

    for i, k1 in enumerate(keys):
        for k2 in keys[i+1:]:

            t1, t2 = tracks[k1], tracks[k2]
            ov = overlap(t1, t2)

            if ov <= MAX_OVERLAP_S:
                continue

            sim = full_track_sim(t1, t2)

            pair = (min(k1, k2), max(k1, k2))

            is_short_overlap = ov <= MAX_DUPLICATE_OVERLAP_S
            is_high_sim = sim >= DUPLICATE_SIM

            if is_short_overlap and is_high_sim:
                duplicates.add(pair)
            else:
                excluded.add(pair)

    return excluded, duplicates

# =========================
# MATCHING (UNCHANGED)
# =========================

def full_track_sim(a, b, n=6):
    def sample(feats, n):
        if len(feats) <= n:
            return feats
        idx = np.linspace(0, len(feats)-1, n, dtype=int)
        return [feats[i] for i in idx]

    f1 = sample(a.get("features", []), n)
    f2 = sample(b.get("features", []), n)

    sims = []
    for x in f1:
        for y in f2:
            if "feature_vector" in x and "feature_vector" in y:
                sims.append(cos_sim(
                    np.array(x["feature_vector"]),
                    np.array(y["feature_vector"])
                ))

    return float(np.median(sims)) if sims else 0.0

def find_candidates(tracks):
    excluded, duplicates = build_exclusions(tracks)

    keys = sorted(tracks.keys(), key=lambda k: time_range(tracks[k])[0])
    candidates = []
    seen_pairs = set()

    # =========================
    # 1. ADD DUPLICATE MERGES FIRST
    # =========================
    for (k1, k2) in duplicates:
        sim = full_track_sim(tracks[k1], tracks[k2])
        pair = (min(k1, k2), max(k1, k2))

        candidates.append({
            "k1": k1,
            "k2": k2,
            "sim": sim,
            "type": "duplicate"
        })
        seen_pairs.add(pair)

    # =========================
    # 2. BEST CONTINUATION MATCHING
    # =========================
    for i, k1 in enumerate(keys):
        t1 = tracks[k1]
        _, end1 = time_range(t1)

        continuation_pool = []

        for k2 in keys[i+1:]:
            t2 = tracks[k2]
            start2, _ = time_range(t2)

            gap = start2 - end1

            if gap < -0.5:
                continue
            if gap > MAX_GAP_S:
                break

            pair = (min(k1, k2), max(k1, k2))
            if pair in excluded:
                continue

            spatial = spatial_gap(t1, t2)

            # Speed constraint
            if gap > 0:
                speed = spatial / gap
                if speed > MAX_SPEED:
                    continue

            # Spatial constraint
            if spatial > MAX_POSITION_PX:
                continue

            sim = endpoint_sim(t1, t2)

            continuation_pool.append((k2, sim, gap, spatial))

        if not continuation_pool:
            continue

        # 🔥 PICK BEST MATCH ONLY
        best_k2, best_sim, gap, spatial = max(
            continuation_pool, key=lambda x: x[1]
        )

        pair = (min(k1, best_k2), max(k1, best_k2))

        if best_sim >= MIN_APPEAR_SIM and pair not in seen_pairs:
            candidates.append({
                "k1": k1,
                "k2": best_k2,
                "sim": best_sim,
                "type": "continuation"
            })
            seen_pairs.add(pair)

    return candidates, excluded, duplicates

def absorb_orphans(groups, tracks, excluded):
    """
    Absorb small leftover tracks into best matching groups.
    Works AFTER fusion.
    """

    # Build reverse map: track -> group
    track_to_group = {}
    for gid, keys in groups.items():
        for k in keys:
            track_to_group[k] = gid

    def group_tracks(gid):
        return groups[gid]

    def group_size(gid):
        return sum(len(tracks[k]["events"]) for k in groups[gid])

    def is_orphan(k):
        gid = track_to_group[k]
        return len(groups[gid]) == 1 and len(tracks[k]["events"]) < 50

    changed = True

    while changed:
        changed = False

        for k in list(track_to_group.keys()):
            if not is_orphan(k):
                continue

            best_gid = None
            best_sim = MIN_APPEAR_SIM

            for gid, members in groups.items():
                if k in members:
                    continue

                # Check exclusion against entire group
                conflict = any(
                    (min(k, m), max(k, m)) in excluded
                    for m in members
                )
                if conflict:
                    continue

                # Compute similarity vs group (use max over members)
                sims = [
                    full_track_sim(tracks[k], tracks[m])
                    for m in members
                ]

                if not sims:
                    continue

                sim = max(sims)

                if sim > best_sim:
                    best_sim = sim
                    best_gid = gid

            if best_gid is not None:
                old_gid = track_to_group[k]

                # Move track
                groups[best_gid].append(k)
                groups[old_gid].remove(k)

                if not groups[old_gid]:
                    del groups[old_gid]

                track_to_group[k] = best_gid
                changed = True

    return groups

# =========================
# FUSION (UNCHANGED)
# =========================

def fuse(tracks, candidates, excluded, duplicates):

    parent = {k: k for k in tracks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for c in sorted(candidates, key=lambda x: -x["sim"]):

        k1, k2 = c["k1"], c["k2"]

        r1, r2 = find(k1), find(k2)
        if r1 == r2:
            continue

        g1 = [k for k in tracks if find(k) == r1]
        g2 = [k for k in tracks if find(k) == r2]

        conflict = any(
            (min(a, b), max(a, b)) in excluded
            for a in g1 for b in g2
        )

        is_dup = (min(k1, k2), max(k1, k2)) in duplicates

        if conflict and not is_dup:
            continue

        union(k1, k2)

    groups = defaultdict(list)
    for k in tracks:
        groups[find(k)].append(k)

    return groups

# =========================
# CLEANING (UNCHANGED)
# =========================

def dist(a, b):
    return np.hypot(a["x"] - b["x"], a["y"] - b["y"])

def remove_duplicates(track):
    if not track:
        return track

    cleaned = [track[0]]
    for p in track[1:]:
        prev = cleaned[-1]
        if abs(p["x"] - prev["x"]) > DUP_EPS or abs(p["y"] - prev["y"]) > DUP_EPS:
            cleaned.append(p)
    return cleaned

def remove_large_jumps(track):
    if not track:
        return track

    cleaned = [track[0]]
    for p in track[1:]:
        if dist(cleaned[-1], p) < MAX_JUMP_CLEAN:
            cleaned.append(p)
    return cleaned

def smooth_track(track):
    smoothed = []
    n = len(track)

    for i in range(n):
        x_sum, y_sum, count = 0, 0, 0

        for j in range(i - SMOOTH_WIN, i + SMOOTH_WIN + 1):
            if 0 <= j < n:
                x_sum += track[j]["x"]
                y_sum += track[j]["y"]
                count += 1

        smoothed.append({
            "x": x_sum / count,
            "y": y_sum / count,
            "t": track[i]["t"],
            "cam": track[i]["cam"]
        })

    return smoothed

def clean_track(track):
    if not track:
        return None

    track = remove_duplicates(track)
    track = remove_large_jumps(track)
    track = smooth_track(track)

    duration = track[-1]["t"] - track[0]["t"]

    if len(track) < MIN_TRACK_POINTS and duration < MIN_DURATION:
        return None

    return track

# =========================
# EXPORT (UNCHANGED)
# =========================

def export(groups, tracks):
    out = {}
    idx = 0

    for _, keys in groups.items():

        merged = []
        for k in keys:
            merged.extend(tracks[k]["events"])

        merged = sorted(merged, key=lambda x: x["time"])

        trajectory = [{
            "x": e["x"],
            "y": e["y"],
            "t": e["time"],
            "cam": e["cam"]
        } for e in merged]

        trajectory = clean_track(trajectory)
        if trajectory is None:
            continue

        out[idx] = {
            "sources": keys,
            "num_events": len(trajectory),
            "tracks": trajectory
        }
        idx += 1

    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2)

# =========================
# MAIN
# =========================

def main():
    vectors, tracks = load_jsonl(INPUT_FILE)

    track_objs = build_tracks(vectors, tracks)

    undistorter = load_intrinsics(INTRINSICS_FILE)
  

    homographies = load_homographies(".")

    apply_world(track_objs, homographies, undistorter)

    track_dict = {str(i): t for i, t in enumerate(track_objs)}

    candidates, excluded, duplicates = find_candidates(track_dict)

    groups = fuse(track_dict, candidates, excluded, duplicates)

    groups = absorb_orphans(groups, track_dict, excluded)
    
    export(groups, track_dict)

    print("[DONE] Strict identity fusion complete (with undistortion)")

if __name__ == "__main__":
    main()