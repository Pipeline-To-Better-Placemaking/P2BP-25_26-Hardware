# With the advent of rely on local homographies, the top-down imgs we're relying on for the puzzle piece system are currently coming out with
# frankly absurd tails that harm the final resolution of the homographies and that of the image as well, which will have further negative effects
# when trying to attach together the entire field into an effective homography map. To address this, this script will attempt to crop the top-down
# img then work in tandem with homography_recalculation.py in order to crop and clean up the preview image for the puzzle piece system and apply 
# the appropriate changes to the homography output to correctly match that newly adjusted topdown image

# This script will take a saved uncropped BEV png (warped ofc), attempt to automatically find the ground, then resize it to center on that ground and crop
# out the excess tails and black space. --black_thresh can be changed to adjust what is considered background excess space besides tails
# Output: *_BEV_cropfill.png (cropped img), *_crop_meta.json (factors for homography recalculation), *_BEV_crop_preview.png (debug, original img and crop)

# This script has a default crop attempt then a backup in case of failure.

# First attempt turns the img grayscale to identify black space, computes dt = distanceTransform(mask) which finds how far each pixel is from the 
# nearest background pixel (floor should have large dt vs tails having small), then creating seeds for each high dt and selecting the best one using 
# (gradient magnitude) * (dt), followed by ground core (r = r_frac * dt(seed)) where dt >= r, lastly define a bounding box around that core

# The second attempt exists in the event of one of the cameras being a pain and deciding to give us a tail that pushes the actual ground all the way to oblivion.
# First is counts mask pixels per row and finds min/max of rows with significant content. Take bottom 35% (adjustable) of the area to try to factor out tail ends.
# Itll compute intensity percentiles here and keep pixels that are darker than the threshold (walls).

# run: python crop_bev_images.py --in_dir ./bevs --out_dir ./bevs_cropped

import argparse
import json
import os
from pathlib import Path
import cv2
import numpy as np

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def dt_core_bbox(img_bgr, black_thresh=12, dt_percentile=70, seed_topk=5000, r_frac=0.5, pad=20, expand_mult=4.0): #quick dist/transform crop attempt
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mask = (gray > black_thresh).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)

    vals = dt[mask > 0]
    dt_thr = float(np.percentile(vals, dt_percentile)) if vals.size else 0.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)

    cand = (dt >= dt_thr) & (mask > 0)
    ys, xs = np.where(cand)
    if xs.size == 0:
        sy, sx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        sy, sx = int(sy), int(sx)
    else:
        idx = np.argsort(dt[ys, xs])[::-1]
        idx = idx[:min(seed_topk, idx.size)]
        ys2, xs2 = ys[idx], xs[idx]
        score = gmag[ys2, xs2] * dt[ys2, xs2]
        j = int(np.argmax(score))
        sy, sx = int(ys2[j]), int(xs2[j])

    dt_seed = float(dt[sy, sx])
    r = max(2, int(r_frac * dt_seed))

    core = (dt >= r).astype(np.uint8) * 255
    n, labels = cv2.connectedComponents(core)
    lab = int(labels[sy, sx])
    if lab == 0 and n > 1:
        areas = [(l, int((labels == l).sum())) for l in range(1, n)]
        lab = max(areas, key=lambda x: x[1])[0] if areas else 0

    comp = (labels == lab) if lab != 0 else (mask > 0)
    ys3, xs3 = np.where(comp)

    x0, x1 = int(xs3.min()), int(xs3.max())
    y0, y1 = int(ys3.min()), int(ys3.max())

    expand = int(pad + expand_mult * r)
    x0 = clamp(x0 - expand, 0, W - 1)
    y0 = clamp(y0 - expand, 0, H - 1)
    x1 = clamp(x1 + expand, 0, W - 1)
    y1 = clamp(y1 + expand, 0, H - 1)

    info = {
        "method": "dt_core",
        "dt_percentile": dt_percentile,
        "dt_thr": dt_thr,
        "seed": [sx, sy],
        "dt_seed": dt_seed,
        "r": r,
    }
    return (x0, y0, x1, y1), info


def band_dark_bbox(img_bgr, black_thresh=12, row_thresh=30, band_frac=0.35, dark_p=60, pad=20): #if first doesnt work, look for densest vertical band and focus on darker parts (temp gallery backup since walls are brighter than the ground)
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > black_thresh)

    row_counts = mask.sum(axis=1)
    rows = np.where(row_counts > row_thresh)[0]
    if rows.size == 0:
        ys, xs = np.where(mask)
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), {"method": "band_dark", "fallback": "mask_bbox"}

    ymin, ymax = int(rows.min()), int(rows.max())
    h = ymax - ymin + 1
    band_start = int(ymax - band_frac * h)

    ys = np.arange(H)
    band_rows = (ys >= band_start) & (ys <= ymax)
    band = mask & band_rows[:, None]

    vals = gray[band]
    if vals.size < 2000:
        band_start = int(ymax - 0.6 * h)
        band_rows = (ys >= band_start) & (ys <= ymax)
        band = mask & band_rows[:, None]
        vals = gray[band]

    if vals.size == 0:
        ys2, xs2 = np.where(mask)
        return (int(xs2.min()), int(ys2.min()), int(xs2.max()), int(ys2.max())), {"method": "band_dark", "fallback": "mask_bbox2"}

    thr = int(np.percentile(vals, dark_p))
    cand = band & (gray <= thr)

    cand_u8 = (cand.astype(np.uint8) * 255)
    cand_u8 = cv2.morphologyEx(cand_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if cand_u8.sum() < 5000:
        cand_u8 = (band.astype(np.uint8) * 255)

    ys3, xs3 = np.where(cand_u8 > 0)
    x0, x1 = int(xs3.min()), int(xs3.max())
    y0, y1 = int(ys3.min()), int(ys3.max())

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)

    info = {
        "method": "band_dark",
        "ymin": ymin,
        "ymax": ymax,
        "band_start": band_start,
        "dark_thr": thr,
        "dark_p": dark_p,
        "band_frac": band_frac,
    }
    return (x0, y0, x1, y1), info


def score_bbox(img_bgr, bbox, black_thresh=12): #pick better crop/bbox
    x0, y0, x1, y1 = bbox
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    crop_gray = gray[y0:y1 + 1, x0:x1 + 1]
    mask = (crop_gray > black_thresh)
    if mask.sum() == 0:
        return -1.0, {}

    gx = cv2.Sobel(crop_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop_gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)

    content_ratio = float(mask.mean())
    mean_gray = float(crop_gray[mask].mean())
    mean_grad = float(gmag[mask].mean())

    #higher gradient, calm light, not empty
    score = float(mean_grad / (mean_gray + 1.0) * (content_ratio ** 0.5))
    return score, {"content_ratio": content_ratio, "mean_gray": mean_gray, "mean_grad": mean_grad}


def process_one(img_path: Path, out_dir: Path, black_thresh: int):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read: {img_path}")

    H, W = img.shape[:2]
    base = img_path.stem.replace("_BEV", "")

    bbox_dt, info_dt = dt_core_bbox(img, black_thresh=black_thresh)
    bbox_bd, info_bd = band_dark_bbox(img, black_thresh=black_thresh)

    s_dt, m_dt = score_bbox(img, bbox_dt, black_thresh=black_thresh)
    s_bd, m_bd = score_bbox(img, bbox_bd, black_thresh=black_thresh)

    if s_bd > s_dt:
        bbox, chosen = bbox_bd, "band_dark"
    else:
        bbox, chosen = bbox_dt, "dt_core"

    x0, y0, x1, y1 = bbox
    crop = img[y0:y1 + 1, x0:x1 + 1].copy()
    crop_h, crop_w = crop.shape[:2]

    #keep original size for downstream consistency
    crop_fill = cv2.resize(crop, (W, H), interpolation=cv2.INTER_CUBIC)

    out_img = out_dir / f"{base}_BEV_cropfill.png"
    out_prev = out_dir / f"{base}_BEV_crop_preview.png"
    out_meta = out_dir / f"{base}_crop_meta.json"

    cv2.imwrite(str(out_img), crop_fill)

    prev = img.copy()
    cv2.rectangle(prev, (x0, y0), (x1, y1), (0, 255, 0), 3)
    cv2.putText(prev, f"{chosen} score={max(s_dt,s_bd):.3f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_prev), prev)

    meta = {
        "original_size": {"w": W, "h": H},
        "crop": {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "w": crop_w, "h": crop_h},
        "resize": {"enabled": True, "w": W, "h": H},
        "scale_after_crop": {"sx": W / crop_w, "sy": H / crop_h},
        "chosen": chosen,
        "candidate_scores": {"dt_core": float(s_dt), "band_dark": float(s_bd)},
        "candidate_metrics": {"dt_core": m_dt, "band_dark": m_bd},
        "candidate_info": {"dt_core": info_dt, "band_dark": info_bd},
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    return out_img, out_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder containing *_BEV.png images")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--pattern", default="*_BEV.png")
    ap.add_argument("--black_thresh", type=int, default=12)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(in_dir.glob(args.pattern))
    if not imgs:
        raise RuntimeError(f"No images found with pattern {args.pattern} in {in_dir}")

    for p in imgs:
        out_img, out_meta = process_one(p, out_dir, args.black_thresh)
        print(f"[OK] {p.name} -> {out_img.name} + {out_meta.name}")


if __name__ == "__main__":
    main()