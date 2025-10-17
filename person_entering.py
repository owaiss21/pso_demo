#!/usr/bin/env python3
"""
Detect people with YOLOv8 + DeepSORT and log when a person enters through an
invisible door positioned around 20% from the left edge of the frame.
A text overlay is displayed on the video when an entry occurs.

Usage:
    python yolov8_deepsort_track.py --input input.mp4 --output out.mp4 --conf 0.6
"""

import argparse
import time
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --------------------------------------------------------------------
# Draw bounding boxes and entry alert
# --------------------------------------------------------------------
def draw_boxes(frame, tracks, entered_ids, entry_message_timer):
    H, W = frame.shape[:2]

    ## MODIFICATION: The two lines for drawing the door line have been removed.

    ## MODIFICATION: Logic to display the "PERSON ENTERED" message.
    if entry_message_timer > 0:
        text = "PERSON ENTERED"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position at the top-center of the frame
        pos_x = (W - text_w) // 2
        pos_y = 50
        
        # Draw a black rectangle background for better visibility
        cv2.rectangle(frame, (pos_x - 10, pos_y - text_h - 10), (pos_x + text_w + 10, pos_y + 10), (0,0,0), -1)
        # Draw the white text
        cv2.putText(frame, text, (pos_x, pos_y), font, font_scale, (0, 255, 0), thickness)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        # Change box color to blue for people who have entered
        color = (0,255,0) if track_id not in entered_ids else (255,0,0)
        cv2.rectangle(frame, (l,t), (r,b), color, 2)
        label = f"ID {track_id}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (l, t - 18), (l + w + 6, t), color, -1)
        cv2.putText(frame, label, (l + 3, t - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return frame

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
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

    # door line (20% from left)
    door_x = int(0.16 * W)

    # keep previous x-centers per track to detect crossing
    last_positions = {}
    entered_ids = set()
    
    ## MODIFICATION: Variables to control the on-screen message timer.
    entry_message_timer = 0
    MESSAGE_DURATION_FRAMES = int(fps * 2) # Show message for 2 seconds

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
                cls = int(box.cls.cpu().numpy())
                if cls != 0:  # person only
                    continue
                x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int).reshape(-1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                conf = float(box.conf.cpu().numpy())
                detections.append(([x1, y1, w, h], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)
        
        ## MODIFICATION: Flag to check if an entry happened in the current frame.
        person_entered_this_frame = False

        # --- Check for entering events ---
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cx = int((l + r) / 2)
            prev_cx = last_positions.get(track_id, None)
            last_positions[track_id] = cx

            # Crossing logic: moved from right side -> left side (crossing door line)
            if prev_cx is not None and prev_cx > door_x and cx <= door_x:
                if track_id not in entered_ids:
                    entered_ids.add(track_id)
                    print(f"[EVENT] Person ID {track_id} ENTERED at frame {frame_idx}")
                    ## MODIFICATION: Set flag to true to trigger the message.
                    person_entered_this_frame = True

        ## MODIFICATION: Update the message timer.
        if person_entered_this_frame:
            # If someone new entered, reset the timer to its full duration
            entry_message_timer = MESSAGE_DURATION_FRAMES
        elif entry_message_timer > 0:
            # If no one new entered, but the timer is active, count it down
            entry_message_timer -= 1
        
        ## MODIFICATION: Pass the message timer to the drawing function.
        annotated = draw_boxes(frame, tracks, entered_ids, entry_message_timer)
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

# --------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.6, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max_age", type=int, default=30)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--print_every", type=int, default=60)
    args = parser.parse_args()
    main(args)