import argparse
import json
from pathlib import Path
import cv2
import numpy as np

# After crop_bev_images.py does its thing the homography matrix it was originally using is now wrong since it maps camera pixels to the old uncropped BEV.
# So this script should fix that by applying the crop translation and resize scaling to the homography matrix so everything lines up again.

# Input: *_homography.yml (opencv FileStorage yaml w/ homography: !!opencv-matrix), *_crop_meta.json
# Outputs: *_homography_cropped.yml

# The math :( (H_new = S * T * H_old), T subtracts the crop offset (x0, y0) where cropping shifts origin, S scales the cropped region back to the BEV original size (since crop_bev_images.py resizes it back)

# run (per cam):
#   python update_homography_yml_for_crop.py \
#       --in_yml d0_3b_f4_02_44_85_homography.yml \
#       --crop_meta bevs_cropped/d0_3b_f4_02_44_85_crop_meta.json \
#       --out_yml bevs_cropped/d0_3b_f4_02_44_85_homography_cropped.yml

def read_yml_all(path: Path): # read keys from opencv yaml
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    keys = [
        "homography", "camera_key", "rtsp", "config_resolution", "frame_size", "inliers", "corners_used",
        "rmse_board", "rmse_world", "ransac_thresh_px", "undistorted", "markers_detected", "charuco_detected",
        "used_undistorted_image", "board_dictionary", "board_squares", "board_square_length", "board_marker_length",
        "p1_corner_id", "p2_corner_id", "p1_world", "p2_world", "timestamp_unix"
    ]
    data = {}
    for k in keys:
        node = fs.getNode(k)
        if node.empty():
            continue
        if node.isString():
            data[k] = node.string()
        elif node.isInt():
            data[k] = int(node.real())
        elif node.isReal():
            data[k] = float(node.real())
        else:
            data[k] = node.mat()
    fs.release()
    return data


def write_yml(path: Path, data: dict):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if "homography" in data:
        fs.write("homography", data["homography"])
    for k, v in data.items():
        if k == "homography":
            continue
        if isinstance(v, str):
            fs.write(k, v)
        elif isinstance(v, (int, np.integer)):
            fs.write(k, int(v))
        elif isinstance(v, (float, np.floating)):
            fs.write(k, float(v))
        elif isinstance(v, np.ndarray):
            fs.write(k, v)
        else:
            fs.write(k, str(v))
    fs.release()


def adjust_homography(H_full: np.ndarray, meta: dict): # H' = S * T * H
    x0 = meta["crop"]["x0"]
    y0 = meta["crop"]["y0"]
    crop_w = meta["crop"]["w"]
    crop_h = meta["crop"]["h"]
    out_w = meta["resize"]["w"]
    out_h = meta["resize"]["h"]

    sx = out_w / crop_w
    sy = out_h / crop_h

    T = np.array([[1.0, 0.0, -float(x0)],
                  [0.0, 1.0, -float(y0)],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    S = np.array([[float(sx), 0.0, 0.0],
                  [0.0, float(sy), 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    H_new = S @ T @ H_full.astype(np.float64)
    if abs(H_new[2, 2]) > 1e-12:
        H_new = H_new / H_new[2, 2]
    return H_new, sx, sy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_yml", required=True, help="Original *_homography.yml")
    ap.add_argument("--crop_meta", required=True, help="*_crop_meta.json from crop_bev_images.py")
    ap.add_argument("--out_yml", required=True, help="Output *_homography_cropped.yml")
    args = ap.parse_args()

    in_yml = Path(args.in_yml)
    meta = json.loads(Path(args.crop_meta).read_text())

    y = read_yml_all(in_yml)
    if "homography" not in y:
        raise RuntimeError("No 'homography' field found in YAML.")

    H_new, sx, sy = adjust_homography(y["homography"], meta)

    y_out = dict(y)
    y_out["homography"] = H_new
    # store crop info inside YAML for traceability
    y_out["crop_x0"] = int(meta["crop"]["x0"])
    y_out["crop_y0"] = int(meta["crop"]["y0"])
    y_out["crop_x1"] = int(meta["crop"]["x1"])
    y_out["crop_y1"] = int(meta["crop"]["y1"])
    y_out["crop_w"] = int(meta["crop"]["w"])
    y_out["crop_h"] = int(meta["crop"]["h"])
    y_out["crop_scale_x"] = float(sx)
    y_out["crop_scale_y"] = float(sy)
    y_out["crop_method"] = str(meta.get("chosen", "unknown"))

    out_yml = Path(args.out_yml)
    out_yml.parent.mkdir(parents=True, exist_ok=True)
    write_yml(out_yml, y_out)
    print(f"[OK] wrote {out_yml}")


if __name__ == "__main__":
    main()