import os
import json
import cv2
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scripts import camera_handler
from scripts.tracker import Undistorter, _try_load_intrinsics_json

#########################
# CONFIGURABLE SETTINGS
#########################

INPUT_FILE = "tracks_events.jsonl"   # input JSONL from tracker
OUTPUT_FILE = "fused_tracks.json"    # output JSON with fused tracks

SIM_THRESHOLD = 0.75    # min cosine similarity
MAX_SPEED = 3000        # max allowed speed (pixels/sec)
MAX_GAP = 8000          # max allowed time gap (ms)

#########################
# 1. Load JSONL Tracks
#########################

def load_jsonl(path):
    vectors = defaultdict(list)
    tracks = defaultdict(list)
    
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            key = (obj.get("mac"), obj.get("sid"))
            if obj["type"] == "vector":
                vectors[key].append(np.array(obj["vector"], dtype=np.float32))
            elif obj["type"] == "track":
                tracks[key].append(obj)
    
    return vectors, tracks

#########################
# 2. Prepare track objects
#########################

def build_track_objects(vectors, tracks):
    track_objs = []
    for key in vectors:
        vecs = np.array(vectors[key])
        rep = vecs.mean(axis=0)
        evs = tracks.get(key, [])
        if not evs:
            continue
        t_start = min(e["time"] for e in evs)
        t_end = max(e["time"] for e in evs)
        x = evs[-1].get("x")
        y = evs[-1].get("y")
        cam = evs[-1].get("mac")
        track_objs.append({
            "key": key,
            "rep": rep,
            "t_start": t_start,
            "t_end": t_end,
            "x": x,
            "y": y,
            "cam": cam,
            "events": sorted(evs, key=lambda e: e["time"])
        })
    track_objs.sort(key=lambda t: t["t_start"])
    return track_objs

#########################
# 3. Load homographies & undistorters from camera_handler
#########################

def build_camera_matrices():
    homographies = {}
    undistorters = {}

    states = camera_handler.get_camera_states()
    for mac, cam_state in states.items():
        # --- Homography ---
        H_list = cam_state.get("Homography")
        if H_list:
            try:
                H = np.array(H_list, dtype=np.float64)
                if H.shape == (3, 3):
                    homographies[mac] = H
            except Exception:
                pass

        # --- Intrinsics / Undistorter ---
        K = cam_state.get("camera_matrix")
        dist = cam_state.get("distortion_coefficients")
        declared_size = cam_state.get("resolution")
        intr_path = cam_state.get("intrinsics_json")  # legacy per-camera JSON

        und = None
        intrinsics_source = "camera_handler"

        if K is not None and dist is not None:
            try:
                K_np = np.array(K, dtype=np.float64)
                dist_np = np.array(dist, dtype=np.float64).reshape(-1)
            except Exception:
                K_np, dist_np = None, None
        else:
            K_np, dist_np = None, None

        if K_np is None or dist_np is None:
            # fallback to legacy JSON
            intrinsics_source = "legacy"
            if intr_path and os.path.exists(intr_path):
                loaded = _try_load_intrinsics_json(intr_path)
                if loaded:
                    K_np, dist_np = loaded

        if K_np is not None and dist_np is not None:
            expected_size = declared_size if intrinsics_source == "legacy" else None
            und = Undistorter(K=K_np, dist=dist_np, expected_size=expected_size)
            if not und.ready():
                und = None

        if und is not None:
            undistorters[mac] = und

    return homographies, undistorters

def undistort_and_apply_homography(track_objs, homographies, undistorters):
    for track in track_objs:
        x, y = track["x"], track["y"]
        if x is None or y is None:
            continue

        und = undistorters.get(track["cam"])
        if und is not None:
            pt = np.array([[[x, y]]], dtype=np.float64)  # (1,1,2)
            pt_und = cv2.undistortPoints(pt, und.K, und.dist, P=und.K)
            x, y = float(pt_und[0,0,0]), float(pt_und[0,0,1])

        H = homographies.get(track["cam"])
        if H is not None:
            pt = np.array([x, y, 1.0], dtype=np.float64)
            pt = H @ pt
            if abs(pt[2]) > 1e-9:
                pt /= pt[2]
                x, y = float(pt[0]), float(pt[1])

        track["x"], track["y"] = x, y

#########################
# 4. Teleport & time checks
#########################

def is_teleporting(gid, track, max_speed=MAX_SPEED):
    if None in (gid["x"], gid["y"], track["x"], track["y"]):
        return False
    dx = track["x"] - gid["x"]
    dy = track["y"] - gid["y"]
    dist = np.sqrt(dx**2 + dy**2)
    dt = (track["t_start"] - gid["t_end"]) / 1000.0
    if dt <= 0:
        return False
    return dist / dt > max_speed

def time_compatible(gid, track, max_gap=MAX_GAP):
    return (track["t_start"] - gid["t_end"]) <= max_gap

#########################
# 5. Offline multi-camera fusion
#########################

def offline_fusion(track_objs, sim_threshold=SIM_THRESHOLD, max_speed=MAX_SPEED, max_gap=MAX_GAP):
    global_ids = []
    next_gid = 0

    for track in track_objs:
        best_gid = None
        best_sim = -1

        for gid in global_ids:
            sim = cosine_similarity(track["rep"].reshape(1, -1), gid["rep"].reshape(1, -1))[0,0]
            if sim < sim_threshold:
                continue
            if not time_compatible(gid, track, max_gap):
                continue
            if is_teleporting(gid, track, max_speed):
                continue
            if sim > best_sim:
                best_sim = sim
                best_gid = gid

        if best_gid is None:
            global_ids.append({
                "gid": next_gid,
                "rep": track["rep"],
                "t_end": track["t_end"],
                "x": track["x"],
                "y": track["y"],
                "cams": {track["cam"]},
                "tracks": [track["events"]]
            })
            next_gid += 1
        else:
            best_gid["rep"] = (best_gid["rep"] + track["rep"]) / 2
            best_gid["t_end"] = max(best_gid["t_end"], track["t_end"])
            best_gid["x"] = track["x"]
            best_gid["y"] = track["y"]
            best_gid["cams"].add(track["cam"])
            best_gid["tracks"].append(track["events"])

    for gid in global_ids:
        all_events = [e for sub in gid["tracks"] for e in sub]
        gid["tracks"] = sorted(all_events, key=lambda e: e["time"])

    return global_ids

#########################
# 6. Export GIDs -> tracks
#########################

def export_gids(global_ids, output_path):
    out = {str(gid["gid"]): gid["tracks"] for gid in global_ids}
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

#########################
# 7. Full offline run
#########################

def run_offline_fusion(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    vectors, tracks = load_jsonl(input_file)
    track_objs = build_track_objects(vectors, tracks)

    homographies, undistorters = build_camera_matrices()
    undistort_and_apply_homography(track_objs, homographies, undistorters)

    global_ids = offline_fusion(track_objs)
    export_gids(global_ids, output_file)

    print(f"Offline fusion complete! {len(global_ids)} unique GIDs created.")
    print(f"Output saved to: {output_file}")

#########################
# MAIN
#########################

if __name__ == "__main__":
    run_offline_fusion()
