import cv2
import numpy as np
import yaml
import sys
import argparse
import time

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


def _get_cuda_device_count() -> int:
    try:
        if not hasattr(cv2, "cuda"):
            return 0
        return int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        return 0


def _enable_opencl() -> bool:
    try:
        if not hasattr(cv2, "ocl"):
            return False
        cv2.ocl.setUseOpenCL(True)
        return bool(cv2.ocl.useOpenCL())
    except Exception:
        return False


class _FramePreprocessor:
    def __init__(self, accel: str):
        cv2.setUseOptimized(True)

        cuda_count = _get_cuda_device_count()
        opencl_ok = _enable_opencl()

        if accel == "auto":
            if cuda_count > 0:
                accel = "cuda"
            elif opencl_ok:
                accel = "opencl"
            else:
                accel = "cpu"

        self.accel = accel
        self.cuda_enabled = accel == "cuda" and cuda_count > 0
        self.opencl_enabled = accel == "opencl" and opencl_ok

        self._gpu_frame = None
        if self.cuda_enabled:
            try:
                self._gpu_frame = cv2.cuda_GpuMat()
            except Exception:
                self.cuda_enabled = False
                self.accel = "cpu"

    def describe(self) -> str:
        if self.cuda_enabled:
            return "CUDA (preprocess only; ArUco/ChArUco detection still CPU)"
        if self.opencl_enabled:
            return "OpenCL via UMat (preprocess only; ArUco/ChArUco detection still CPU)"
        return "CPU"

    def to_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.cuda_enabled and self._gpu_frame is not None:
            self._gpu_frame.upload(frame_bgr)
            gpu_gray = cv2.cuda.cvtColor(self._gpu_frame, cv2.COLOR_BGR2GRAY)
            return gpu_gray.download()

        if self.opencl_enabled:
            umat = cv2.UMat(frame_bgr)
            gray_umat = cv2.cvtColor(umat, cv2.COLOR_BGR2GRAY)
            return gray_umat.get()

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

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

def calibrate(
    source,
    output,
    accel: str = "auto",
    target_fps: float | None = 10.0,
    drop_old_frames: bool = True,
    max_grabs: int = 8,
):
    """Calibrate camera from video source."""
    pre = _FramePreprocessor(accel)

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
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if target_fps is not None:
        try:
            cap.set(cv2.CAP_PROP_FPS, float(target_fps))
        except Exception:
            pass
    img_size = None
    
    print(f"\nCalibrating {source}")
    print(f"Acceleration: {pre.describe()}")
    if target_fps is None:
        print("Frame rate: as fast as possible")
    else:
        print(f"Frame rate: target {target_fps:g} FPS (drops old frames: {drop_old_frames})")
    print("SPACE=capture frame, c=calibrate (need 15+), q=quit\n")

    next_tick = time.perf_counter()
    
    while True:
        if target_fps is not None and target_fps > 0:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(max(0.0, next_tick - now))
            next_tick = max(next_tick, time.perf_counter()) + (1.0 / float(target_fps))

        ret, frame = cap.read()
        if drop_old_frames and max_grabs > 0:
            # Pull a few extra frames to flush internal buffers and keep the newest.
            for _ in range(int(max_grabs)):
                ok, newer = cap.read()
                if not ok:
                    break
                frame = newer
                ret = True
        if not ret:
            continue
        if img_size is None:
            img_size = (frame.shape[1], frame.shape[0])
        
        gray = pre.to_gray(frame)
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
    parser = argparse.ArgumentParser(description="Camera calibration with ChArUco")
    parser.add_argument("source", nargs="?", help="Video source (rtsp://..., device index, or file) OR 'create'")
    parser.add_argument("output", nargs="?", help="Output YAML path (required for calibration)")
    parser.add_argument(
        "--accel",
        choices=["auto", "cpu", "opencl", "cuda"],
        default="auto",
        help="Acceleration for preprocessing. Note: ArUco/ChArUco detection still runs on CPU.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Target processing FPS. Lower values reduce CPU load. Use 0 to disable throttling.",
    )
    parser.add_argument(
        "--no-drop-old-frames",
        action="store_true",
        help="Do not flush buffered frames; may increase latency but keeps all frames.",
    )
    parser.add_argument(
        "--max-grabs",
        type=int,
        default=8,
        help="How many extra reads to flush per loop when dropping old frames.",
    )
    args = parser.parse_args()

    if not args.source:
        parser.print_help()
        raise SystemExit(2)

    if args.source == "create":
        create_board()
    else:
        if not args.output:
            print("Error: output.yml is required when calibrating")
            raise SystemExit(2)
        target_fps = None if args.fps == 0 else float(args.fps)
        calibrate(
            args.source,
            args.output,
            accel=args.accel,
            target_fps=target_fps,
            drop_old_frames=not args.no_drop_old_frames,
            max_grabs=args.max_grabs,
        )