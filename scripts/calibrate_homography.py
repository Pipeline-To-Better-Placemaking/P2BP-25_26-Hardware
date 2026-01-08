import cv2
import yaml
import numpy as np
import argparse

# this should popup the first frame of the chosen camera and tell you which corner to pick (based on heading)
# USE CORRECT ORDER DESCRIBED
# the corner you picked will be represented by a dot with text next to it with its supposed heading
# r to reset, q to quit

# after clicking all corner headings, you'll get a warped top/down view of the ground, this view must be a square and NOT a trapezoid
# ensure square is not mirrored, rotated, stretched, or go off screen
# confirm this view is as desired then press space to continue, final output should be a valid yaml file

# repeat this process a total of 4 times, once for each camera, below are all the commands for ease of use:
# python calibrate_homography.py --source rtsp://127.0.0.1:8554/cam0 --out homographies/garage-c0-homography.yml
# python calibrate_homography.py --source rtsp://127.0.0.1:8554/cam1 --out homographies/garage-c1-homography.yml
# python calibrate_homography.py --source rtsp://127.0.0.1:8554/cam2 --out homographies/garage-c2-homography.yml
# python calibrate_homography.py --source rtsp://127.0.0.1:8554/cam3 --out homographies/garage-c3-homography.yml

IN_TO_MM = 25.4

SW_IN = (96.0, 103.125)
SE_IN = (108.0, 103.125)
NW_IN = (96.0, 115.125)
NE_IN = (108.0, 115.125)

# origin = SW tape corner
def world_points_mm_local():
    sw = np.array(SW_IN) * IN_TO_MM
    se = np.array(SE_IN) * IN_TO_MM
    nw = np.array(NW_IN) * IN_TO_MM
    ne = np.array(NE_IN) * IN_TO_MM
    # subtract SW so SW = (0,0)
    se -= sw
    nw -= sw
    ne -= sw
    sw = np.array([0.0, 0.0])
    return np.array([sw, se, ne, nw], dtype=np.float32)

LABELS = ["SW", "SE", "NE", "NW"]  # click order

def make_warp_matrix_from_Hworld(H_world, world_pts_mm, mm2px=2.0, pad=40):
    min_xy = world_pts_mm.min(axis=0)
    max_xy = world_pts_mm.max(axis=0)
    w_mm = max_xy[0] - min_xy[0]
    h_mm = max_xy[1] - min_xy[1]

    W = int(w_mm * mm2px + 2 * pad)
    H = int(h_mm * mm2px + 2 * pad)

    # world(mm) -> topdown(px)
    # x_px = pad + (x - min_x)*mm2px
    # y_px = pad + (max_y - y)*mm2px  # flip so North is up?
    S = np.array([
        [mm2px,   0.0, pad - min_xy[0]*mm2px],
        [0.0,   -mm2px, pad + max_xy[1]*mm2px],
        [0.0,    0.0,  1.0]
    ], dtype=np.float64)

    H_warp = S @ H_world
    return H_warp, (W, H)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="RTSP/Video path (e.g. rtsp://127.0.0.1:8554/cam0)")
    ap.add_argument("--out", required=True, help="Output YAML path (e.g. homographies/garage-c0-homography.yml)")
    ap.add_argument("--mm2px", type=float, default=2.0, help="Top-down preview scale")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read from source.")

    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicks) < 4:
                clicks.append((x, y))

    win = "Click tape corners: SW, SE, NE, NW  (r=reset, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = frame.copy()

        # draw already clicked points
        for i, (cx, cy) in enumerate(clicks):
            cv2.circle(vis, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(vis, LABELS[i], (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # instruction for next click
        if len(clicks) < 4:
            cv2.putText(vis, f"Next: {LABELS[len(clicks)]}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

        cv2.imshow(win, vis)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == 27:
            cv2.destroyAllWindows()
            return
        if key == ord('r'):
            clicks.clear()

        if len(clicks) == 4:
            src = np.array(clicks, dtype=np.float32) # image px
            dst_world = world_points_mm_local() # world mm (local)
            H_world, _ = cv2.findHomography(src, dst_world, 0) # exact 4-pt
            if H_world is None:
                print("Homography failed. Press r and try again.")
                clicks.clear()
                continue

            # preview warp
            H_warp, (W, H) = make_warp_matrix_from_Hworld(H_world, dst_world, mm2px=args.mm2px, pad=40)
            warped = cv2.warpPerspective(frame, H_warp, (W, H))

            # draw the square outline in preview
            square = np.array([
                [40, H - 40],          # SW
                [W - 40, H - 40],      # SE
                [W - 40, 40],          # NE
                [40, 40],              # NW
            ], dtype=np.int32)
            cv2.polylines(warped, [square], True, (0, 0, 255), 2)

            cv2.imshow("Top-down preview (SPACE=accept, r=redo)", warped)
            while True:
                k2 = cv2.waitKey(0) & 0xFF
                if k2 == ord('r'):
                    cv2.destroyWindow("Top-down preview (SPACE=accept, r=redo)")
                    clicks.clear()
                    break
                if k2 == 32:  # SPACE
                    out = {"homography": H_world.tolist()}
                    with open(args.out, "w") as f:
                        yaml.safe_dump(out, f)
                    print(f"Saved: {args.out}")
                    cv2.destroyAllWindows()
                    return

if __name__ == "__main__":
    main()