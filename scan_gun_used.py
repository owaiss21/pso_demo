#!/usr/bin/env python3
"""
custom_yolo11_deepsort_track.py
Detects and tracks objects, assuming a single persistent scan gun.

Usage:
    python custom_yolo11_deepsort_track.py \
        --input input.mp4 \
        --output out.mp4 \
        --model yolov11_custom.pt \
        --conf 0.7 \
        --show
"""

import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Class Configuration ---
CLASS_NAMES = {
    0: "Card Machine", 1: "Cash", 2: "Cash During Trns",
    3: "Cash Register CLOSED", 4: "Cash Register OPEN", 5: "Customer",
    6: "Employee", 7: "Scan Gun",
}
# Define class IDs
CUSTOMER_CLASS_ID = 5
EMPLOYEE_CLASS_ID = 6
SCAN_GUN_CLASS_ID = 7
# We now track Customers, Employees, and Scan Guns
TRACKED_CLASSES = [CUSTOMER_CLASS_ID, EMPLOYEE_CLASS_ID, SCAN_GUN_CLASS_ID]

# --- Scan Gun Logic Configuration ---
MOVEMENT_THRESHOLD_PIXELS = 10
STATIONARY_THRESHOLD_FRAMES = 15
# If gun is not seen for this many frames, reset its state to stationary.
GUN_TIMEOUT_FRAMES = 60 

# --- Visualization Configuration ---
STATE_COLORS = {
    'STATIONARY': (255, 0, 0),    # Blue
    'MOVING': (0, 255, 255),  # Yellow
    'IN_USE': (0, 0, 255),      # Red
}
DEFAULT_COLOR = (0, 255, 0) # Green for other objects


def draw_boxes(frame, tracks, class_names, the_one_gun_state):
    """Draw tracked boxes. For any scan gun, apply the single gun state."""
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        cls = int(track.get_det_class())
        l, t, r, b = map(int, track.to_ltrb())

        label = class_names.get(cls, "Unknown")
        color = DEFAULT_COLOR

        # If this is a scan gun, use the single global state for its label and color
        if cls == SCAN_GUN_CLASS_ID:
            state = the_one_gun_state['state']
            label += f" [{state}]"
            color = STATE_COLORS.get(state, DEFAULT_COLOR)

        # Draw bounding box
        cv2.rectangle(frame, (l, t), (r, b), color, 2)

        # Draw label
        label = f"{label} ID: {track_id}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (l, t - 18), (l + w + 6, t), color, -1)
        cv2.putText(frame, label, (l + 3, t - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame


def main(args):
    model = YOLO(args.model)
    tracker = DeepSort(max_age=args.max_age, n_init=3)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (W, H))

    # --- SINGLE State Management for THE Scan Gun ---
    the_scan_gun_state = {
        'state': 'STATIONARY',
        'position': None,
        'stationary_frames': 0,
        'last_seen_frame': 0
    }

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        res = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        r = res[0]

        detections = []
        if hasattr(r, "boxes") and len(r.boxes) > 0:
            for box in r.boxes:
                cls = int(box.cls)
                if cls not in TRACKED_CLASSES:
                    continue
                
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().reshape(-1))
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, cls))

        # Update DeepSort for all tracked objects
        tracks = tracker.update_tracks(detections, frame=frame)

        # --- Scan Gun Logic (Single Gun Assumption) ---

        # 1. Find the best scan gun detection in the current frame (highest confidence)
        scan_gun_detections = [d for d in detections if d[2] == SCAN_GUN_CLASS_ID]
        best_gun_detection = max(scan_gun_detections, key=lambda d: d[1]) if scan_gun_detections else None
        
        # 2. Check for customer presence
        customer_present = any(
            track.get_det_class() == CUSTOMER_CLASS_ID for track in tracks if track.is_confirmed()
        )

        # 3. Update the single gun state based on the best detection
        if best_gun_detection:
            the_scan_gun_state['last_seen_frame'] = frame_idx
            
            # Calculate current position
            box, _, _ = best_gun_detection
            x, y, w, h = box
            current_pos = np.array([x + w / 2, y + h / 2])

            # If this is the first time we see the gun, initialize its position
            if the_scan_gun_state['position'] is None:
                the_scan_gun_state['position'] = current_pos
            
            # Calculate movement and update state
            movement_dist = np.linalg.norm(current_pos - the_scan_gun_state['position'])
            
            if movement_dist < MOVEMENT_THRESHOLD_PIXELS:
                the_scan_gun_state['stationary_frames'] += 1
                if the_scan_gun_state['stationary_frames'] >= STATIONARY_THRESHOLD_FRAMES:
                    the_scan_gun_state['state'] = 'STATIONARY'
            else:
                the_scan_gun_state['stationary_frames'] = 0
                previous_state = the_scan_gun_state['state']

                if previous_state == 'STATIONARY' and customer_present:
                    the_scan_gun_state['state'] = 'IN_USE'
                    print(f"Event (Frame {frame_idx}): Scan Gun is now IN_USE.")
                elif previous_state != 'IN_USE':
                    the_scan_gun_state['state'] = 'MOVING'

            # Update position for the next frame's comparison
            the_scan_gun_state['position'] = current_pos
        
        else:
            # If gun is not detected, check for timeout
            if frame_idx - the_scan_gun_state['last_seen_frame'] > GUN_TIMEOUT_FRAMES:
                if the_scan_gun_state['state'] != 'STATIONARY':
                     print(f"Event (Frame {frame_idx}): Scan Gun timed out, resetting to STATIONARY.")
                     the_scan_gun_state['state'] = 'STATIONARY'
                     the_scan_gun_state['position'] = None # Forget old position


        # Draw boxes using the single gun state
        annotated = draw_boxes(frame, tracks, CLASS_NAMES, the_scan_gun_state)
        out.write(annotated)

        if args.show:
            cv2.imshow("tracks", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_idx % args.print_every == 0:
            elapsed = time.time() - t0
            print(f"Frame {frame_idx}/{total} processed, elapsed {elapsed:.1f}s")

    cap.release()
    out.release()
    if args.show:
        cv2.destroyAllWindows()

    print("Saved:", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to save output video")
    parser.add_argument("--model", default="yolov11_custom.pt", help="Path to YOLOv11 custom weights")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max_age", type=int, default=30)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--print_every", type=int, default=60)
    args = parser.parse_args()

    main(args)