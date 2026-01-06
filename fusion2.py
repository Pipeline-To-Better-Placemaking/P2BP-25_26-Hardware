import json
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine

# --- CONFIGURATION ---
SIM_THRESH = 0.88           # Cosine similarity threshold for embedding match
MIN_STRONG_MATCHES = 3      # Minimum strong embedding matches to fuse
MAX_SPEED = 3.0             # pixels per millisecond (adjust to your scene)
MAX_GAP = 2000              # max time gap in ms allowed between consecutive points
TOTAL_PEOPLE = 4            # total number of people expected in the video

TRACKS_FILE = "tracks_by_camera.json"
FEATURES_FILE = "features.json"
OUTPUT_FILE = "fused_identities.json"

# --- UTILITIES ---
def l2_normalize(v):
    v = np.array(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-6)

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

def mean_feature(feats):
    return l2_normalize(np.mean(feats, axis=0))

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# --- PHYSICAL CHECK ---
def is_physical(tracks_a, tracks_b):
    """
    Check if two sets of tracks could belong to the same person:
    - Consecutive points movement is possible
    - Time gaps are within MAX_GAP
    """
    combined = sorted(tracks_a + tracks_b, key=lambda p: p["time"])
    for i in range(1, len(combined)):
        dt = combined[i]["time"] - combined[i - 1]["time"]
        if dt <= 0 or dt > MAX_GAP:
            return False
        dx = combined[i]["x"] - combined[i - 1]["x"]
        dy = combined[i]["y"] - combined[i - 1]["y"]
        distance = np.sqrt(dx**2 + dy**2)
        speed = distance / dt
        if speed > MAX_SPEED:
            return False
    return True

# --- IDENTITY FUSION ---
def fuse_identities(features, tracks):
    fused = []
    gid_to_fused = {}

    # Precompute mean features for each track
    avg_feats = {gid: mean_feature([l2_normalize(f) for f in feats]) 
                 for gid, feats in features.items() if feats}

    for gid, feats in features.items():
        feats = [l2_normalize(f) for f in feats]
        if not feats or not tracks.get(gid):
            continue

        mf = avg_feats.get(gid)
        matched = False

        for idx, entry in enumerate(fused):
            # Check if physical movement is possible
            if not is_physical(tracks[gid], entry["tracks"]):
                continue

            # Count strong embedding matches
            strong = sum(1 for fa in feats for fb in entry["features"] if cosine_sim(fa, fb) > SIM_THRESH)
            if strong >= MIN_STRONG_MATCHES:
                entry["gids"].append(gid)
                entry["features"].extend(feats)
                entry["tracks"].extend(tracks[gid])
                entry["mean_feat"] = mean_feature(entry["features"])
                gid_to_fused[gid] = idx
                matched = True
                break

        if not matched:
            fused.append({
                "gids": [gid],
                "features": feats,
                "tracks": tracks[gid],
                "mean_feat": mf
            })
            gid_to_fused[gid] = len(fused) - 1

    # Merge to respect TOTAL_PEOPLE if necessary
    while len(fused) > TOTAL_PEOPLE:
        min_dist = float("inf")
        merge_pair = None
        for i in range(len(fused)):
            for j in range(i + 1, len(fused)):
                if is_physical(fused[i]["tracks"], fused[j]["tracks"]):
                    d = cosine(fused[i]["mean_feat"], fused[j]["mean_feat"])
                    if d < min_dist:
                        min_dist = d
                        merge_pair = (i, j)
        if merge_pair:
            i, j = merge_pair
            fused[i]["gids"].extend(fused[j]["gids"])
            fused[i]["features"].extend(fused[j]["features"])
            fused[i]["tracks"].extend(fused[j]["tracks"])
            fused[i]["mean_feat"] = mean_feature(fused[i]["features"])
            fused.pop(j)
        else:
            break  # no more physically plausible merges

    return fused, gid_to_fused

# --- TRACK FUSION ---
def fuse_tracks(tracks_by_camera, gid_map):
    fused_tracks = defaultdict(lambda: {"tracks": [], "source_gids": [], "cameras": set()})

    for cam, persons in tracks_by_camera.items():
        for sid, data in persons.items():
            gid = str(data["gid"])
            fused_id = gid_map.get(gid)
            if fused_id is None:
                continue
            fused_tracks[fused_id]["tracks"].extend(data["track"])
            fused_tracks[fused_id]["source_gids"].append(gid)
            fused_tracks[fused_id]["cameras"].update(data["cameraIDs"])

    for f in fused_tracks.values():
        f["cameras"] = list(f["cameras"])

    return fused_tracks

def flatten_tracks(tracks_by_camera):
    flat = defaultdict(list)
    for cam in tracks_by_camera.values():
        for sid, data in cam.items():
            flat[str(data["gid"])].extend(data["track"])
    return flat

# --- MAIN FUNCTION ---
def main():
    print("[INFO] Loading data...")
    tracks_by_camera = load_json(TRACKS_FILE)
    features = load_json(FEATURES_FILE)

    flat_tracks = flatten_tracks(tracks_by_camera)

    print("[INFO] Fusing identities (precision-first, physical constraints)...")
    fused, gid_map = fuse_identities(features, flat_tracks)
    print(f"[INFO] Reduced {len(features)} → {len(fused)} identities")

    fused_tracks = fuse_tracks(tracks_by_camera, gid_map)

    output = {
        "fused_identities": [
            {"fused_id": i, "source_gids": f["gids"]}
            for i, f in enumerate(fused)
        ],
        "fused_tracks": fused_tracks
    }

    save_json(output, OUTPUT_FILE)
    print(f"[INFO] Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
