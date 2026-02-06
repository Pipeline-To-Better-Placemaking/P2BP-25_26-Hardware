import os
import json
import cv2
import numpy as np
import tracker
import yaml
from tracker import Undistorter
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity


#########################
# CONFIG
#########################

INPUT_FILE = "tracks_events.jsonl"
OUTPUT_FILE = "fused_tracks.json"
INTRINSICS_FILE = "ANNKE_camera_intrinsics.yml"
SIM_THRESHOLD = 0.75
MAX_SPEED = 4000       # world units per second
MAX_GAP = 12000         # ms

#########################
# 1. LOAD JSONL
#########################

def load_jsonl(path):
    """
    Load tracking vectors and track points.
    Assumes vectors belong to the same (mac, sid) track.
    Coordinates are RAW PIXELS.
    """
    vectors = defaultdict(list)
    tracks = defaultdict(list)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            mac = obj.get("mac")
            sid = obj.get("sid")
            t = obj.get("time")

            if mac is None or sid is None:
                continue

            key = (mac.lower(), sid)

            if obj["type"] == "vector":
                vectors[key].append({
                    "vector": np.array(obj["vector"], dtype=np.float32),
                    "time": t,
                })

            elif obj["type"] == "track":
                tracks[key].append({
                    "x": obj.get("x"),
                    "y": obj.get("y"),
                    "time": t,
                    "cam": mac.lower(),
                    "sid": sid,
                })

    return vectors, tracks

#########################
# 2. LOAD INTRINSICS
#########################


def load_intrinsics(path):
    abs_path = os.path.abspath(path)
    print(f"[INTRINSICS] Loading: {abs_path}")

    if not os.path.exists(abs_path):
        raise FileNotFoundError(abs_path)

    with open(abs_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    K = np.array(data["camera_matrix"], dtype=np.float64)
    D = np.array(data["distortion_coefficients"], dtype=np.float64).reshape(-1)

    w = int(data["image_width"])
    h = int(data["image_height"])

    print("[INTRINSICS] Loaded OK")
    print("K:\n", K)
    print("D:", D)
    print("size:", (w, h))

    # ✅ directly return the ready-to-use undistorter
    return Undistorter(
        K=K,
        dist=D,
        expected_size=(w, h),
        intrinsics_path=abs_path
    )



#########################
# 3. BUILD TRACK OBJECTS
#########################

def build_track_objects(vectors, tracks):
    """
    One object per (camera, sid).
    """
    objs = []

    for key, vecs in vectors.items():
        evs = tracks.get(key)
        if not evs:
            continue

        evs = sorted(evs, key=lambda e: e["time"])
        rep = np.mean([v["vector"] for v in vecs], axis=0)

        objs.append({
            "cam": key[0],
            "sid": key[1],
            "rep": rep,
            "t_start": evs[0]["time"],
            "t_end": evs[-1]["time"],
            "x": evs[-1]["x"],
            "y": evs[-1]["y"],
            "events": evs,
        })

    objs.sort(key=lambda o: o["t_start"])
    return objs

#########################
# 4. LOAD HOMOGRAPHIES
#########################

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
# 5. APPLY HOMOGRAPHY
#########################

#########################
# 5. APPLY HOMOGRAPHY
#########################

def apply_homographies(track_objs, homographies, undistorter, debug=True):
    """
    Convert pixel coords -> world coords.
    Adds verbose debugging for:
      - undistortion
      - homography
    """

    for track in track_objs:
        cam = track["cam"]
        H = homographies.get(cam)

        if H is None:
            if debug:
                print(f"[HOMO][SKIP] No homography for camera {cam}")
            continue

        if debug:
            print(f"\n[TRACK] cam={cam} sid={track['sid']} events={len(track['events'])}")

        for i, e in enumerate(track["events"]):
            x, y = e["x"], e["y"]
            if x is None or y is None:
                continue

            orig_x, orig_y = x, y

            # -------------------------
            # UNDISTORT
            # -------------------------
            x_u, y_u = undistorter.undistort_point(x, y, undistorter.expected_size)

            if debug:
                print(
                    f"  [UNDISTORT] #{i} "
                    f"({orig_x:.2f}, {orig_y:.2f}) -> "
                    f"({x_u:.2f}, {y_u:.2f})"
                )

            # -------------------------
            # HOMOGRAPHY
            # -------------------------
            p = np.array([x_u, y_u, 1.0], dtype=np.float64)
            p = H @ p

            if abs(p[2]) < 1e-9:
                if debug:
                    print("  [HOMO][WARN] invalid homogeneous scale (p[2]≈0)")
                continue

            p /= p[2]
            x_w, y_w = float(p[0]), float(p[1])

            if debug:
                print(
                    f"  [HOMO] world -> ({x_w:.2f}, {y_w:.2f})"
                )

            e["x"], e["y"] = x_w, y_w

        last = track["events"][-1]
        track["x"], track["y"] = last["x"], last["y"]

        if debug:
            print(f"[TRACK END] final world pos=({track['x']:.2f}, {track['y']:.2f})")


#########################
# 6. FUSION HELPERS
#########################

def time_compatible(gid, track):
    gap_ms = track["t_start"] - gid["t_end"]
    return gap_ms <= MAX_GAP, gap_ms


def teleport_metrics(gid, track):
    dx = track["x"] - gid["x"]
    dy = track["y"] - gid["y"]

    dist = float(np.hypot(dx, dy))
    dt_ms = track["t_start"] - gid["t_end"]

    if dt_ms <= 0:
        return False, dist, dt_ms, 0.0

    dt_s = dt_ms / 1000.0
    speed = dist / dt_s

    teleport = speed > MAX_SPEED
    return teleport, dist, dt_ms, speed


#########################
# 7. OFFLINE FUSION
#########################

def offline_fusion(track_objs, debug=False):
    gids = []
    next_gid = 0

    for track in track_objs:
        best = None
        best_sim = -1

        if debug:
            print(f"\n[FUSION] track cam={track['cam']} sid={track['sid']}")

        for gid in gids:
            # -------------------------
            # 1️⃣ PHYSICS
            # -------------------------
            teleport, dist, dt_ms, speed = teleport_metrics(gid, track)

            if teleport:
                if debug:
                    print(
                        f"  gid={gid['gid']} rejected (teleport) "
                        f"dist={dist:.2f}  dt={dt_ms}ms  speed={speed:.2f} > {MAX_SPEED}"
                    )
                continue


            # -------------------------
            # 2️⃣ TIME
            # -------------------------
            ok_time, gap_ms = time_compatible(gid, track)

            if not ok_time:
                if debug:
                    print(
                        f"  gid={gid['gid']} rejected (time gap) "
                        f"gap={gap_ms}ms > {MAX_GAP}"
                    )
                continue


            # -------------------------
            # 3️⃣ APPEARANCE CHECK (expensive → last)
            # -------------------------
            sim = cosine_similarity(
                track["rep"].reshape(1, -1),
                gid["rep"].reshape(1, -1),
            )[0, 0]

            if debug:
                print(f"  gid={gid['gid']} sim={sim:.3f}")

            if sim < SIM_THRESHOLD:
                if debug:
                    print("     rejected (low similarity)")
                continue

            # -------------------------
            # 4️⃣ choose best similarity
            # -------------------------
            if sim > best_sim:
                best_sim = sim
                best = gid

        # -------------------------
        # ASSIGN OR CREATE
        # -------------------------
        if best is None:
            if debug:
                print("  → NEW GID")

            gids.append({
                "gid": next_gid,
                "rep": track["rep"],
                "t_end": track["t_end"],
                "x": track["x"],
                "y": track["y"],
                "tracks": [track["events"]],
            })
            next_gid += 1

        else:
            if debug:
                print(f"  → MERGED into gid={best['gid']} (sim={best_sim:.3f})")

            best["rep"] = (best["rep"] + track["rep"]) / 2
            best["t_end"] = track["t_end"]
            best["x"], best["y"] = track["x"], track["y"]
            best["tracks"].append(track["events"])

    # -------------------------
    # Flatten tracks + collect sources
    # -------------------------
    for gid in gids:
        flat_tracks = [e for t in gid["tracks"] for e in t]
        gid["tracks"] = sorted(flat_tracks, key=lambda e: e["time"])

        sources = []
        seen = set()
        for e in gid["tracks"]:
            src = (e["cam"], e["sid"])
            if src not in seen:
                seen.add(src)
                sources.append({"cam": e["cam"], "sid": e["sid"]})
        gid["sources"] = sources

    return gids


#########################
# 8. EXPORT
#########################

def export_gids(gids):
    out = {}
    for g in gids:
        out[str(g["gid"])] = {
            "sources": g["sources"],  # cam + sid before fusion
            "tracks": g["tracks"],    # fused trajectory, each point with its cam + sid
        }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

#########################
# 9. MAIN
#########################

def main():
    vectors, tracks = load_jsonl(INPUT_FILE)
    track_objs = build_track_objects(vectors, tracks)

    homographies = load_homographies(".")
    undistorter= load_intrinsics(INTRINSICS_FILE)
    apply_homographies(track_objs, homographies, undistorter, False)

    gids = offline_fusion(track_objs, True)
    export_gids(gids)

    print(f"[DONE] {len(gids)} fused identities written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
