import cv2
import yaml
import numpy as np
import argparse
import math

"""BOARD CONFIGURATION - Edit per printed board"""

SQUARES_X = 8  # Columns
SQUARES_Y = 8  # Rows

# Size of board in millimeters
SQUARE_SIZE_MM = 50.0   # Size of each chessboard square
MARKER_SIZE_MM = 40.0   # Size of ArUco markers

# ArUco dictionary 
ARUCO_DICT = cv2.aruco.DICT_4X4_50

"""REFERENCE POINT CONFIGURATION"""

# P1 and P2 are two inner corners used to establish board position and rotation
# P1 = top-right inner corner of the ChArUco grid
# P2 = bottom-right inner corner of the ChArUco grid

# Compute corner IDs for P1 and P2 based on board dimensions
CORNERS_X = SQUARES_X - 1  # Inner corners per row
CORNERS_Y = SQUARES_Y - 1  # Inner corner rows

# P2 is bottom-right inner corner: last column of first row (row 0)
P2_CORNER_ID = CORNERS_X - 1

# P1 is top-right inner corner: last column of last row
P1_CORNER_ID = (CORNERS_Y - 1) * CORNERS_X + (CORNERS_X - 1)


### Visualization parameters ###
MM_TO_PX = 2.0 
PREVIEW_PAD = 50

"""Parse command-line arguments."""
def parse_arguments():
    ap = argparse.ArgumentParser(
        description="Calculate camera homography using a ChArUco board"
    )
    ap.add_argument(
        "--source", 
        required=True, 
        help="RTSP URL or video file path (e.g., rtsp://127.0.0.1:8554/cam0)"
    )
    ap.add_argument(
        "--out", 
        required=True, 
        help="Output YAML file path (e.g., homographies/cam0-homography.yml)"
    )
    ap.add_argument(
        "--p1-world", 
        required=True, 
        help="P1 world coordinates in mm as 'X,Y' (e.g., 2500,3200)"
    )
    ap.add_argument(
        "--p2-world", 
        required=True, 
        help="P2 world coordinates in mm as 'X,Y' (e.g., 2500,2900)"
    )
    return ap.parse_args()


def parse_point(point_str):
    try:
        x, y = point_str.split(',')
        return np.array([float(x), float(y)], dtype=np.float64)
    except ValueError:
        raise ValueError(f"Invalid point format: '{point_str}'. Expected 'X,Y' (e.g., 2500,3200)")


"""Compute world coordinates for all Charuco inner corners."""
def compute_corner_world_positions(p1_world, p2_world):
    # P1 and P2 positions in board-local coordinates (mm)
    # Board origin (0,0) is at bottom-left inner corner
    p1_board = np.array([
        (CORNERS_X - 1) * SQUARE_SIZE_MM,
        (CORNERS_Y - 1) * SQUARE_SIZE_MM
    ], dtype=np.float64)
    
    p2_board = np.array([
        (CORNERS_X - 1) * SQUARE_SIZE_MM,
        0.0
    ], dtype=np.float64)
    
    # Calculate rotation angle
    board_vec = p1_board - p2_board
    world_vec = p1_world - p2_world
    
    # Angle of each vector relative to positive X-axis
    board_angle = math.atan2(board_vec[1], board_vec[0])
    world_angle = math.atan2(world_vec[1], world_vec[0])
    
    # Rotation needed to align board with world
    theta = world_angle - board_angle
    
    # Rotation matrix
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    # Calculate translation: rotate p2_board, then find offset to p2_world
    p2_board_rotated = np.array([
        p2_board[0] * cos_t - p2_board[1] * sin_t,
        p2_board[0] * sin_t + p2_board[1] * cos_t
    ])
    translation = p2_world - p2_board_rotated
    
    # Compute world position for every inner corner
    corner_world_positions = {}
    
    for corner_id in range(CORNERS_X * CORNERS_Y):
        # Corner position in board-local coordinates
        col = corner_id % CORNERS_X
        row = corner_id // CORNERS_X
        board_x = col * SQUARE_SIZE_MM
        board_y = row * SQUARE_SIZE_MM
        
        # Rotate and translate to world coordinates
        world_x = board_x * cos_t - board_y * sin_t + translation[0]
        world_y = board_x * sin_t + board_y * cos_t + translation[1]
        
        corner_world_positions[corner_id] = (world_x, world_y)
    
    return corner_world_positions

def create_charuco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    charuco_board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_SIZE_MM,
        MARKER_SIZE_MM,
        aruco_dict
    )
    charuco_detector = cv2.aruco.CharucoDetector(charuco_board)
    return charuco_detector

def capture_frame(source):
    """Open camera and capture a frame when user presses SPACE."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")
    
    print("Press SPACE to capture frame, Q to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame, retrying...")
            continue
        
        cv2.imshow("Press SPACE to capture, Q to quit", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            captured_frame = frame.copy()
            cap.release()
            cv2.destroyAllWindows()
            return captured_frame
        elif key == ord('q') or key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return None

def main():
    # PARSE ARGUMENTS AND SETUP
    args = parse_arguments()
    p1_world = parse_point(args.p1_world)
    p2_world = parse_point(args.p2_world)
    
    corner_world_positions = compute_corner_world_positions(p1_world, p2_world)
    charuco_detector = create_charuco_detector()
    
    # CAPTURE FRAME
    captured_frame = capture_frame(args.source)
    if captured_frame is None:
        return
    
    """ DETECTION AND HOMOGRAPHY LOOP """
    while True:
        # Detect Charuco corners
        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            charuco_detector.detectBoard(captured_frame)
        
        # Validate detection
        if charuco_ids is None or len(charuco_ids) < 4:
            num_found = 0 if charuco_ids is None else len(charuco_ids)
            print(f"Insufficient corners detected: {num_found} (need at least 4)")
            print("Press R to recapture, Q to quit")
            
            cv2.imshow("Detection failed - R to recapture, Q to quit", captured_frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('r'):
                cv2.destroyAllWindows()
                captured_frame = capture_frame(args.source)
                if captured_frame is None:
                    return
                continue
            else:
                cv2.destroyAllWindows()
                return
        
        num_detected = len(charuco_ids)
        print(f"Detected {num_detected} corners")
        
        # Build correspondence arrays
        src_points = []
        dst_points = []
        
        for i in range(len(charuco_ids)):
            corner_id = charuco_ids[i][0]
            pixel_coords = charuco_corners[i][0]
            
            if corner_id in corner_world_positions:
                world_coords = corner_world_positions[corner_id]
                src_points.append(pixel_coords)
                dst_points.append(world_coords)
        
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        # Compute homography
        H_world, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        
        if H_world is None:
            print("Homography computation failed. Press R to recapture, Q to quit")
            cv2.imshow("Homography failed", captured_frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('r'):
                cv2.destroyAllWindows()
                captured_frame = capture_frame(args.source)
                if captured_frame is None:
                    return
                continue
            else:
                cv2.destroyAllWindows()
                return
        
        num_inliers = int(np.sum(mask))
        inlier_ratio = num_inliers / len(mask)
        print(f"Homography computed: {num_inliers}/{len(mask)} inliers ({inlier_ratio:.1%})")
        
        if inlier_ratio < 0.8:
            print("Warning: Low inlier ratio - calibration may be inaccurate")
        
        # Create visualization preview
        all_world_pts = np.array(list(corner_world_positions.values()))
        min_xy = all_world_pts.min(axis=0)
        max_xy = all_world_pts.max(axis=0)
        
        W = int((max_xy[0] - min_xy[0]) * MM_TO_PX + 2 * PREVIEW_PAD)
        H = int((max_xy[1] - min_xy[1]) * MM_TO_PX + 2 * PREVIEW_PAD)
        
        S = np.array([
            [MM_TO_PX, 0, PREVIEW_PAD - min_xy[0] * MM_TO_PX],
            [0, -MM_TO_PX, PREVIEW_PAD + max_xy[1] * MM_TO_PX],
            [0, 0, 1]
        ], dtype=np.float64)
        
        H_warp = S @ H_world
        warped = cv2.warpPerspective(captured_frame, H_warp, (W, H))
        
        # Draw expected board outline on preview
        board_corner_ids = [
            0,
            CORNERS_X - 1,
            P1_CORNER_ID,
            (CORNERS_Y - 1) * CORNERS_X
        ]
        
        preview_corners = []
        for cid in board_corner_ids:
            wx, wy = corner_world_positions[cid]
            px = int(MM_TO_PX * (wx - min_xy[0]) + PREVIEW_PAD)
            py = int(PREVIEW_PAD + (max_xy[1] - wy) * MM_TO_PX)
            preview_corners.append([px, py])
        
        cv2.polylines(warped, [np.array(preview_corners)], True, (0, 255, 0), 2)
        
        # Draw corner labels
        labels = ["BL", "BR(P2)", "TR(P1)", "TL"]
        for i, cid in enumerate(board_corner_ids):
            wx, wy = corner_world_positions[cid]
            px = int(MM_TO_PX * (wx - min_xy[0]) + PREVIEW_PAD)
            py = int(PREVIEW_PAD + (max_xy[1] - wy) * MM_TO_PX)
            cv2.putText(warped, labels[i], (px + 5, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # User confirmation
        cv2.imshow("Top-down preview (SPACE=accept, R=retry, Q=quit)", warped)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('r'):
            cv2.destroyAllWindows()
            captured_frame = capture_frame(args.source)
            if captured_frame is None:
                return
            continue
        
        elif key == ord(' '):
            # Save homography
            output_data = {
                "homography": H_world.tolist(),
                "calibration_info": {
                    "method": "charuco",
                    "board_squares": [SQUARES_X, SQUARES_Y],
                    "square_size_mm": SQUARE_SIZE_MM,
                    "marker_size_mm": MARKER_SIZE_MM,
                    "p1_world_mm": p1_world.tolist(),
                    "p2_world_mm": p2_world.tolist(),
                    "corners_detected": num_detected,
                    "inliers": num_inliers,
                    "inlier_ratio": float(inlier_ratio),
                }
            }
            
            with open(args.out, 'w') as f:
                yaml.safe_dump(output_data, f)
            
            print(f"Saved: {args.out}")
            cv2.destroyAllWindows()
            return
        
        else:  # Q or ESC
            cv2.destroyAllWindows()
            return


if __name__ == "__main__":
    main()