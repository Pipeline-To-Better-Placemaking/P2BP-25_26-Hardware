import json
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine

# --- CONFIGURATION ---
SIM_THRESH = 0.88           # Cosine similarity threshold for embedding match
MIN_STRONG_MATCHES = 3      # Minimum strong embedding matches to fuse
MAX_SPEED = 3.0             # pixels per millisecond
MAX_GAP = 2000              # max time gap in ms between consecutive points
TOTAL_PEOPLE = 4            # expected number of people in the scene

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
def physical_check(tracks_a, tracks_b):
    """
    Checks if two track sets could belong to the same person:
    - Consecutive points movement is physically possible
    - Time gaps are within MAX_GAP
    """
    combined = sorted(tracks_a + tracks_b, key=lambda p: p["time"])
    for i in range(1, len(combined)):
        dt = combined[i]["time"] - combined[i-1]["time"]
        if dt <= 0 or dt > MAX_GAP:
            return False
        dx = combined[i]["x"] - combined[i-1]["x"]
        dy = combined[i]["y"] - combined[i-1]["y"]
        speed = np.sqrt(dx**2 + dy**2) / dt
        if speed > MAX_SPEED:
            return False
    return True

# --- FUSE IDENTITIES ---
def fuse_identities(features, tracks, total_people=TOTAL_PEOPLE):
    fused = []
    gid_map = {}

    # Precompute mean features
    avg_feats = {gid: mean_feature([l2_normalize(f) for f in feats])
                 for gid, feats in features.items() if feats}

    for gid, feats in features.items():
        feats_norm = [l2_normalize(f) for f in feats]
        if not feats_norm or not tracks.get(gid):
            continue
        mf = avg_feats.get(gid)
        matched = False

        for idx, entry in enumerate(fused):
            if not physical_check(entry["tracks"], tracks[gid]):
                continue
            sim = cosine_sim(mf, entry["mean_feat"])
            if sim > SIM_THRESH:
                entry["gids"].append(gid)
                entry["features"].extend(feats_norm)
                entry["tracks"].extend(tracks[gid])
                entry["mean_feat"] = mean_feature(entry["features"])
                gid_map[gid] = idx
                matched = True
                break

        if not matched:
            if len(fused) < total_people:
                fused.append({
                    "gids": [gid],
                    "features": feats_norm,
                    "tracks": tracks[gid],
                    "mean_feat": mf
                })
                gid_map[gid] = len(fused) - 1
            else:
                # Assign to closest fused identity if we reached TOTAL_PEOPLE
                sims = [cosine_sim(mf, e["mean_feat"]) for e in fused]
                best_idx = int(np.argmax(sims))
                fused[best_idx]["gids"].append(gid)
                fused[best_idx]["features"].extend(feats_norm)
                fused[best_idx]["tracks"].extend(tracks[gid])
                fused[best_idx]["mean_feat"] = mean_feature(fused[best_idx]["features"])
                gid_map[gid] = best_idx

    return fused, gid_map

# --- TRACK FUSION ---
def flatten_tracks(tracks_by_camera):
    flat = defaultdict(list)
    for cam in tracks_by_camera.values():
        for sid, data in cam.items():
            flat[str(data["gid"])].extend(data["track"])
    return flat

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

    # convert sets to lists
    for f in fused_tracks.values():
        f["cameras"] = list(f["cameras"])

    return fused_tracks

# --- MAIN ---
def main():
    print("[INFO] Loading tracks and features...")
    tracks_by_camera = load_json(TRACKS_FILE)
    features = load_json(FEATURES_FILE)

    flat_tracks = flatten_tracks(tracks_by_camera)

    print("[INFO] Fusing identities with physical & feature constraints...")
    fused, gid_map = fuse_identities(features, flat_tracks, TOTAL_PEOPLE)
    print(f"[INFO] Reduced {len(features)} → {len(fused)} fused identities")

    fused_tracks = fuse_tracks(tracks_by_camera, gid_map)

    output = {
        "fused_identities": [{"fused_id": i, "source_gids": f["gids"]} for i, f in enumerate(fused)],
        "fused_tracks": fused_tracks
    }

    save_json(output, OUTPUT_FILE)
    print(f"[INFO] Saved fused identities to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
