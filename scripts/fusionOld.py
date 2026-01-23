import json
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

#########################
# CONFIGURABLE SETTINGS
#########################

INPUT_FILE = "tracks_events.jsonl"
OUTPUT_FILE = "fused_tracks.json"

SIM_THRESHOLD = 0.75
MAX_SPEED = 3000        # px/sec
MAX_GAP = 8000          # ms

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

        evs = sorted(evs, key=lambda e: e["time"])

        t_start = evs[0]["time"]
        t_end = evs[-1]["time"]
        x = evs[-1].get("x")
        y = evs[-1].get("y")
        cam = evs[-1].get("sid")

        track_objs.append({
            "key": key,
            "rep": rep,
            "t_start": t_start,
            "t_end": t_end,
            "x": x,
            "y": y,
            "cam": cam,
            "events": evs
        })

    track_objs.sort(key=lambda t: t["t_start"])
    return track_objs

#########################
# 3. Teleport & time checks
#########################

def is_teleporting(gid, track, max_speed=MAX_SPEED):
    if None in (gid["x"], gid["y"], track["x"], track["y"]):
        return False

    dx = track["x"] - gid["x"]
    dy = track["y"] - gid["y"]
    dist = np.sqrt(dx**2 + dy**2)

    dt = (track["t_start"] - gid["t_end"]) / 1000.0

    # Disallow overlapping or backward fusion
    if dt <= 0:
        return True

    return dist / dt > max_speed

def time_compatible(gid, track, max_gap=MAX_GAP):
    dt = track["t_start"] - gid["t_end"]

    # Disallow backward or overlapping fusion entirely
    if dt < 0:
        return False

    return dt <= max_gap

#########################
# 4. Offline multi-camera fusion
#########################

def offline_fusion(track_objs,
                   sim_threshold=SIM_THRESHOLD,
                   max_speed=MAX_SPEED,
                   max_gap=MAX_GAP):

    global_ids = []
    next_gid = 0

    for track in track_objs:
        best_gid = None
        best_sim = -1

        for gid in global_ids:
            sim = cosine_similarity(
                track["rep"].reshape(1, -1),
                gid["rep"].reshape(1, -1)
            )[0, 0]

            if sim < sim_threshold:
                continue

            if not time_compatible(gid, track, max_gap):
                continue

            # Hard motion gate
            if gid["x"] is not None and track["x"] is not None:
                dx = track["x"] - gid["x"]
                dy = track["y"] - gid["y"]
                dist = np.sqrt(dx**2 + dy**2)

                dt = (track["t_start"] - gid["t_end"]) / 1000.0
                if dt <= 0 or dist > max_speed * dt:
                    continue

            if is_teleporting(gid, track, max_speed):
                continue

            if sim > best_sim:
                best_sim = sim
                best_gid = gid

        if best_gid is None:
            # Create new GID
            events = sorted(track["events"], key=lambda e: e["time"])
            last_ev = events[-1]

            global_ids.append({
                "gid": next_gid,
                "rep": track["rep"],
                "t_end": last_ev["time"],
                "x": last_ev.get("x"),
                "y": last_ev.get("y"),
                "cams": {track["cam"]},
                "tracks": events[:]
            })
            next_gid += 1

        else:
            # Fuse into existing GID
            best_gid["rep"] = (best_gid["rep"] + track["rep"]) / 2
            best_gid["cams"].add(track["cam"])

            best_gid["tracks"].extend(track["events"])
            best_gid["tracks"].sort(key=lambda e: e["time"])

            last_ev = best_gid["tracks"][-1]
            best_gid["t_end"] = last_ev["time"]
            best_gid["x"] = last_ev.get("x")
            best_gid["y"] = last_ev.get("y")

    return global_ids

#########################
# 5. Export GIDs -> tracks
#########################

def export_gids(global_ids, output_path):
    out = {}

    for gid in global_ids:
        out[str(gid["gid"])] = gid["tracks"]

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

#########################
# 6. Full offline run
#########################

def run_offline_fusion(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    vectors, tracks = load_jsonl(input_file)
    track_objs = build_track_objects(vectors, tracks)

    global_ids = offline_fusion(track_objs)

    export_gids(global_ids, output_file)

    print(f"Offline fusion complete! {len(global_ids)} unique GIDs created.")
    print(f"Output saved to: {output_file}")

#########################
# MAIN
#########################

if __name__ == "__main__":
    run_offline_fusion()
