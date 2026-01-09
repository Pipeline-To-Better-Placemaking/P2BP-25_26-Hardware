import json
import numpy as np
from scipy.spatial.distance import cosine

# ---------------- SETTINGS ----------------
INPUT_FILE = "tracks_simple.json"
OUTPUT_FILE = "tracks_fused.json"
WINDOW_SIZE = 10          # last few features to compute rolling mean
MAX_DIST = 0.30         # cosine distance threshold for matching
MAX_TIME_DIFF = 16000     # max ms gap to consider temporal continuity
TEMPORAL_TOLERANCE = 8000  # allow small overshoot in time gaps

# ---------------- LOAD TRACKS ----------------
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

global_people = {}  # fused ID dictionary
next_global_id = 0

# ---------------- FUSION ----------------
for cam_name in sorted(data.keys()):  # process cameras in order
    cam_data = data[cam_name]
    for tid in sorted(cam_data.keys(), key=int):  # process IDs in order
        tdata = cam_data[tid]
        if len(tdata["features"]) == 0:
            continue

        local_feats = [np.array(f["feature"]) for f in tdata["features"]]
        local_times = [f["time"] for f in tdata["features"]]

        assigned_gid = None

        # Compare each local feature to all global tracks
        for i, feat in enumerate(local_feats):
            time = local_times[i]
            best_match = None
            best_dist = float("inf")

            for gid, gdata in global_people.items():
                if len(gdata["features"]) == 0:
                    continue

                # last feature time in global track
                last_time = gdata["features"][-1]["time"]
                if time - last_time > MAX_TIME_DIFF + TEMPORAL_TOLERANCE:
                    continue  # too far in time

                # rolling mean of last WINDOW_SIZE features
                last_feats = [np.array(f["feature"]) for f in gdata["features"][-WINDOW_SIZE:]]
                g_mean = np.mean(last_feats, axis=0)

                dist = cosine(feat, g_mean)
                if dist < best_dist:
                    best_dist = dist
                    best_match = gid

            # Decide assignment
            if best_match is not None and best_dist < MAX_DIST:
                assigned_gid = best_match
            else:
                # no match, create new global ID
                if assigned_gid is None:  # only create once per local track
                    assigned_gid = next_global_id
                    next_global_id += 1
                    global_people[assigned_gid] = {"track": [], "features": [], "cameraIDs": set()}

            # merge this single feature + track point into global
            global_people[assigned_gid]["track"].append(tdata["track"][i])
            global_people[assigned_gid]["features"].append(tdata["features"][i])
            global_people[assigned_gid]["cameraIDs"].update(tdata["cameraIDs"])


# ---------------- EXPORT ----------------
export = {}
for gid in sorted(global_people.keys()):
    gdata = global_people[gid]

    # sort track points and features by time
    sorted_track = sorted(gdata["track"], key=lambda x: x["time"])
    sorted_features = sorted(gdata["features"], key=lambda x: x["time"])

    export[str(gid)] = {
        "track": sorted_track,
        "features": sorted_features,
        "cameraIDs": list(gdata["cameraIDs"])
    }

with open(OUTPUT_FILE, "w") as f:
    json.dump(export, f, indent=2)

print(f"[INFO] Exported {len(global_people)} fused tracks to {OUTPUT_FILE}")

