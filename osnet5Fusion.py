import json
import numpy as np
from scipy.spatial.distance import cosine

INPUT_FILE = "tracks_simple.json"
OUTPUT_FILE = "tracks_fused.json"
FEAT_THRESH = 0.31 # appearance similarity threshold
DIST_THRESH = 200.0        # max allowed spatial distance for merging
TIME_TOLERANCE = 50        # max time gap tolerance

with open(INPUT_FILE) as f:
    data = json.load(f)

# Flatten all features into individual entries
all_features = []
for cam, cam_data in data.items():
    for tid, tdata in cam_data.items():
        for f, p in zip(tdata.get("features", []), tdata["track"]):
            all_features.append({
                "camera": cam,
                "track_id": tid,
                "time": p["time"],
                "x": p["x"],
                "y": p["y"],
                "feat": np.array(f["feature"]),
                "assigned": False
            })

fused_ids = []

def feature_distance(f1, f2):
    return cosine(f1["feat"], f2["feat"])

def spatial_distance(f1, f2):
    dx = f1["x"] - f2["x"]
    dy = f1["y"] - f2["y"]
    return np.sqrt(dx*dx + dy*dy)

for i, feat in enumerate(all_features):
    if feat["assigned"]:
        continue
    # Start a new fused ID
    new_id = [feat]
    feat["assigned"] = True

    changed = True
    while changed:
        changed = False
        for j, other in enumerate(all_features):
            if other["assigned"]:
                continue
            # Check appearance similarity with any feature in new_id
            for f in new_id:
                if feature_distance(f, other) < FEAT_THRESH:
                    # Check temporal and spatial consistency
                    if abs(f["time"] - other["time"]) <= TIME_TOLERANCE or spatial_distance(f, other) < DIST_THRESH:
                        new_id.append(other)
                        other["assigned"] = True
                        changed = True
                        break
    fused_ids.append(new_id)

# Convert fused features back to track format
export = {}
for gid, group in enumerate(fused_ids):
    merged_track = []
    cams = set()
    for f in group:
        merged_track.append({
            "time": f["time"],
            "x": f["x"],
            "y": f["y"]
        })
        cams.add(f["camera"])
    export[str(gid)] = {
        "track": sorted(merged_track, key=lambda x: x["time"]),
        "cameraIDs": list(cams)
    }

with open(OUTPUT_FILE, "w") as f:
    json.dump(export, f, indent=2)

print(f"[INFO] Exported {len(export)} fused identities.")
