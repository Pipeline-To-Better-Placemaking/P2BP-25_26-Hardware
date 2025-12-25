import cv2
import numpy as np
import yaml
import sys

"""
Camera Calibration with ChArUco
Usage:
  1. python camera_calibration_minimal.py create
  2. Print charuco_board.png, tape to cardboard
  3. python camera_calibration_minimal.py rtsp://127.0.0.1:8554/cam0 calibration/cam0.yml
  
Controls: SPACE=capture, c=calibrate, q=quit
"""

# ChArUco config
SQUARES_X, SQUARES_Y = 5, 7
SQUARE_LENGTH, MARKER_LENGTH = 0.04, 0.02
ARUCO_DICT = cv2.aruco.DICT_6X6_250

def create_board():
    """Generate ChArUco board PNG."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    try:
        board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
        img = board.generateImage((2480, 3508), marginSize=100, borderBits=1)
    except:
        board = cv2.aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, dictionary)
        img = board.draw((2480, 3508), marginSize=100, borderBits=1)
    cv2.imwrite("charuco_board.png", img)
    print("✓ Saved charuco_board.png - Print on A4 paper")

def calibrate(source, output):
    """Calibrate camera from video source."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    try:
        board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
    except:
        board = cv2.aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, dictionary)
        params = cv2.aruco.DetectorParameters_create()
        detector = None
    
    all_corners, all_ids = [], []
    cap = cv2.VideoCapture(source)
    img_size = None
    
    print(f"\nCalibrating {source}")
    print("SPACE=capture frame, c=calibrate (need 15+), q=quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if img_size is None:
            img_size = (frame.shape[1], frame.shape[0])
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()
        
        # Detect markers
        if detector:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            try:
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            except:
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)[:3]
            
            if charuco_corners is not None and len(charuco_corners) > 3:
                cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
                cv2.putText(display, f"{len(charuco_corners)} corners - Press SPACE", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(display, f"Captured: {len(all_corners)}/15+", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Calibration", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32 and ids is not None and charuco_corners is not None and len(charuco_corners) > 3:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            print(f"✓ Captured {len(all_corners)}")
        elif key == ord('c') and len(all_corners) >= 10:
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Calibrate
    print(f"\nCalibrating with {len(all_corners)} images...")
    ret, mtx, dist, _, _ = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, board, img_size, None, None)
    
    if ret:
        data = {
            'camera_matrix': mtx.tolist(),
            'distortion_coefficients': dist.tolist(),
            'image_width': img_size[0],
            'image_height': img_size[1],
            'reprojection_error': float(ret),
            'num_images': len(all_corners)
        }
        with open(output, 'w') as f:
            yaml.safe_dump(data, f)
        print(f"✓ Saved to {output}")
        print(f"  Reprojection error: {ret:.4f} pixels")
    else:
        print("❌ Calibration failed")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Create board:  python camera_calibration_minimal.py create")
        print("  Calibrate cam: python camera_calibration_minimal.py rtsp://... output.yml")
    elif sys.argv[1] == "create":
        create_board()
    elif len(sys.argv) == 3:
        calibrate(sys.argv[1], sys.argv[2])
    else:
        print("Error: Invalid arguments")