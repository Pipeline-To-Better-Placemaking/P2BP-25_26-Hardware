import cv2
import numpy as np
from ultralytics import YOLO
import torch
import sys
import os
from pathlib import Path
from collections import defaultdict
import time
import json

# Add OSNet-IBN1-Lite to path
OSNET_LITE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OSNet-IBN1-Lite")
sys.path.insert(0, OSNET_LITE_PATH)

# Import OSNet components
from OSNet import osnet_ibn_x1_0
import torch.nn as nn


class OSNetLiteEncoder:
    """OSNet feature extractor."""

    def __init__(self, weight_filepath, patch_height=256, patch_width=128,
                 num_classes=1000, use_gpu=True):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

        print(f"Loading OSNet model from {weight_filepath}")
        self.model = osnet_ibn_x1_0(num_classes=num_classes, loss='softmax')

        if os.path.exists(weight_filepath):
            checkpoint = torch.load(weight_filepath, map_location=self.device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dict[k] = v

            self.model.load_state_dict(new_state_dict, strict=False)
            print(f" Loaded weights from {weight_filepath}")

        self.model.to(self.device)
        self.model.eval()

    def preprocess_crop(self, crop):
        crop = cv2.resize(crop, (self.patch_width, self.patch_height))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - self.norm_mean) / self.norm_std
        crop = torch.from_numpy(crop.transpose(2, 0, 1)).float()
        return crop

    def extract_features(self, crops):
        if len(crops) == 0:
            return np.array([])

        tensors = [self.preprocess_crop(crop) for crop in crops]
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            features = self.model(batch)

        return features.cpu().numpy()

    def extract_single_feature(self, crop):
        features = self.extract_features([crop])
        return features[0] if len(features) > 0 else None


class ImprovedPersonTracker:
    """
    Tracker that collects OSNet feature vectors per track for post-hoc re-identification.
    Live Re-ID is intentionally disabled — identity fusion is handled by a separate script.
    Global IDs are assigned sequentially by first appearance order, not by YOLO ID.
    """

    def __init__(self, yolo_model='yolov8s.pt', osnet_weights=None,
                 feature_interval=1.0, use_gpu=True):
        """
        Args:
            yolo_model: YOLO model (use 's' or 'm' for better accuracy)
            osnet_weights: Path to OSNet weights
            feature_interval: Seconds between feature extractions
            use_gpu: Use GPU if available
        """
        print("Loading YOLO model...")
        self.yolo = YOLO(yolo_model)

        print("Loading OSNet encoder...")
        self.encoder = OSNetLiteEncoder(
            weight_filepath=osnet_weights,
            num_classes=1000,
            use_gpu=use_gpu
        )

        self.feature_interval = feature_interval

        # Track storage keyed by global_id
        self.track_history = defaultdict(lambda: {
            'boxes': [],
            'features': [],
            'feature_vectors': [],  # Store raw numpy vectors for post-hoc Re-ID
            'timestamps': [],
            'frame_ids': [],
            'last_seen': -1
        })

        self.last_feature_time = {}

        # Sequential global ID counter - always increments by 1 regardless of YOLO IDs
        self.global_id_counter = 1

        # Maps YOLO track ID -> global ID (reset each video)
        self.yolo_to_global_id = {}

    def extract_person_crop(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 32 or (y2 - y1) < 32:
            return None

        return frame[y1:y2, x1:x2]

    def process_video(self, video_path, output_path=None, display=False):
        """Process video with YOLO tracking and OSNet feature collection (no live Re-ID)."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nProcessing: {video_path}")
        print(f"FPS: {fps}, Resolution: {width}x{height}, Frames: {total_frames}")

        # Reset state for fresh video
        self.track_history.clear()
        self.last_feature_time.clear()
        self.global_id_counter = 1
        self.yolo_to_global_id.clear()

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_id = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_id / fps

            # YOLO tracking
            results = self.yolo.track(
                frame,
                persist=True,
                classes=[0],  # Person class only
                verbose=False,
                conf=0.5,
                iou=0.5,
                tracker="bytetrack.yaml"
            )

            current_active_global_ids = set()

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()

                for box, yolo_id, conf in zip(boxes, track_ids, confidences):

                    crop = self.extract_person_crop(frame, box)

                    if yolo_id not in self.yolo_to_global_id:
                        # New YOLO ID — assign the next sequential global ID
                        global_id = self.global_id_counter
                        self.global_id_counter += 1
                        self.yolo_to_global_id[yolo_id] = global_id

                    else:
                        # Known YOLO ID — look up existing global ID
                        global_id = self.yolo_to_global_id[yolo_id]

                    current_active_global_ids.add(global_id)

                    # Update track history with this frame's detection
                    self.track_history[global_id]['boxes'].append(box)
                    self.track_history[global_id]['timestamps'].append(current_time)
                    self.track_history[global_id]['frame_ids'].append(frame_id)
                    self.track_history[global_id]['last_seen'] = frame_id

                    # Periodic feature extraction at feature_interval seconds
                    if global_id not in self.last_feature_time:
                        self.last_feature_time[global_id] = -self.feature_interval  # Force first extraction

                    time_since_last = current_time - self.last_feature_time[global_id]

                    if time_since_last >= self.feature_interval and crop is not None:
                        feature = self.encoder.extract_single_feature(crop)

                        if feature is not None:
                            # Save full feature entry for fusion script
                            self.track_history[global_id]['features'].append({
                                'timestamp': current_time,
                                'frame_id': frame_id,
                                'confidence': float(conf),
                                'feature_vector': feature.tolist(),
                                'bbox': box.tolist()
                            })
                            # Also keep raw numpy vector for post-hoc Re-ID in fusion script
                            self.track_history[global_id]['feature_vectors'].append(feature)
                            self.last_feature_time[global_id] = current_time

                    # Draw bounding box and label on frame
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ID:{global_id} ({conf:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # HUD overlay
            info = (f"Frame: {frame_id}/{total_frames} | "
                    f"Active IDs: {len(current_active_global_ids)} | "
                    f"Total IDs: {len(self.track_history)}")
            cv2.putText(frame, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if writer:
                writer.write(frame)

            if display:
                cv2.imshow('Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_id += 1

            if frame_id % 30 == 0:
                elapsed = time.time() - start_time
                fps_proc = frame_id / elapsed
                print(f"Progress: {frame_id}/{total_frames} ({fps_proc:.1f} fps)")

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        print(f"\n Completed")
        print(f"  Unique Global IDs: {len(self.track_history)}")

    def save_features(self, output_file, add_timestamp=True):
        """
        Save extracted features to JSON.
        output_file: Base output filename
        add_timestamp: If True, adds timestamp to avoid overwriting previous runs
        """
        if add_timestamp:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(output_file)
            output_file = output_path.parent / f"{output_path.stem}_{timestamp}{output_path.suffix}"

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        data = {}

        for track_id, history in self.track_history.items():
            if not history['timestamps']:
                continue

            data[f"track_{track_id}"] = {
                'num_detections': len(history['boxes']),
                'duration': (history['timestamps'][-1] - history['timestamps'][0]
                             if len(history['timestamps']) > 1 else 0),
                'num_features': len(history['features']),
                'features': history['features']  # Contains bbox + feature_vector + timestamp
            }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f" Features saved to: {output_file}")
        return str(output_file)

    def get_summary(self):
        """Get tracking summary."""
        summary = []

        for track_id, history in self.track_history.items():
            if not history['timestamps']:
                continue

            summary.append({
                'track_id': track_id,
                'num_detections': len(history['boxes']),
                'duration': (history['timestamps'][-1] - history['timestamps'][0]
                             if len(history['timestamps']) > 1 else 0),
                'num_features': len(history['features']),
            })

        return sorted(summary, key=lambda x: x['track_id'])


if __name__ == "__main__":
    print("Script started!")

    osnet_weights = 'OSNet-IBN1-Lite/weights/osnet_ibn_x1_0.pth'

    tracker = ImprovedPersonTracker(
        yolo_model='yolov8s.pt',
        osnet_weights=osnet_weights,
        feature_interval=1.0,
        use_gpu=True
    )

    print("Processing video!")
    print("Press 'q' to quit")

    tracker.process_video(
        '4p-c0-trim.mp4',
        output_path='tracking_outputs/4p-c0-trim_improved1.mp4',
        display=True
    )

    saved_path = tracker.save_features('tracking_outputs/4p-c0-trim_improved_features222.json')

    print("\nSummary:")
    for item in tracker.get_summary():
        print(f"  ID {item['track_id']}: {item['num_detections']} detections, "
              f"{item['duration']:.1f}s, {item['num_features']} features")

    print(f"\nDone! JSON saved to: {saved_path}")