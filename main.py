# main.py
import os, sys, csv, time, math
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from tqdm import tqdm

from tracker_core import TrackerEngine, CLASS_NAMES
from card_logic import CardMachineMonitor, TransactionDecider

def format_hhmmss_msec(seconds: float) -> str:
    if seconds < 0 or math.isnan(seconds) or math.isinf(seconds):
        return "00:00:00.000"
    whole = int(seconds); msec = int((seconds - whole) * 1000)
    hh = whole // 3600; mm = (whole % 3600) // 60; ss = (whole % 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{msec:03d}"

def draw_human_track(frame, xyxy, logical_id: int, track_id: int, role: str):
    x1, y1, x2, y2 = map(int, xyxy)
    color = (0, 255, 0) if role.lower() == "customer" else (255, 128, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{role.upper()} L{logical_id} (T{track_id})"
    cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cx = (x1 + x2) // 2; by = y2
    cv2.circle(frame, (cx, by), 4, color, -1)

def draw_box_with_label(frame, xyxy, label: str):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
    cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

def main(
    video_path: str,
    weights_path: str = "models/weights.pt",
    conf: float = 0.45,
    out_fps: float = 24.0,
):
    # --- Open video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

    # --- Output dirs/files ---
    video_name = Path(video_path).stem
    run_root = Path(f"output/{video_name}")
    frames_dir = run_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = str(run_root / "video_with_detection.mp4")
    events_csv_path = str(run_root / "events.csv")

    # --- Writers ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, float(out_fps), (width, height))

    f_csv = open(events_csv_path, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["event","logical_id","video_time","system_time_utc","frame_idx","note"])

    # --- Engines ---
    tracker = TrackerEngine(
        weights_path=weights_path, conf=conf, frame_w=width, frame_h=height
    )

    # Card machine monitor thresholds: scale by frame diagonal for robustness
    diag = math.hypot(width, height)
    cm_monitor = CardMachineMonitor(
        stable_radius_px=0.02 * diag,        # ~2% diag as "same place" jitter
        stable_confirm_s=2.0,
        moved_threshold_px=0.05 * diag,      # ~5% diag as "moved"
        moved_min_duration_s=0.25,
        rest_min_duration_s=0.8
    )
    decider = TransactionDecider(rearm_gap_s=2.0, enable_cash_heuristic=True)

    start_wallclock = datetime.now(timezone.utc)
    pbar = tqdm(total=frame_count if frame_count > 0 else None, desc="Processing", unit="frm")

    # --- Frame loop ---
    next_write_t = 0.0
    write_dt = 1.0 / max(0.1, float(out_fps))

    orig_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_now = orig_idx / in_fps

            # 1) Run tracker step
            out_dict = tracker.step(frame, t_now)
            human_tracks = out_dict["human_tracks"]
            other_dets = out_dict["nonhuman"]
            new_events = out_dict["new_events"]  # *_at_counter dwell events

            # 2) Update Card Machine state from nonhuman dets
            card_boxes = [d for d in other_dets if d["cls_id"] == 0]
            cm_monitor.update(card_boxes, t_now)
            cm_state = cm_monitor.snapshot()

            # 3) Decide card/cash transactions
            tx_events = decider.update(t_now, human_tracks, cm_state, other_dets)

            # 4) Log events
            def _log(evt_type, lid, note=""):
                vt = format_hhmmss_msec(t_now)
                st = (start_wallclock + timedelta(seconds=t_now)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                writer.writerow([evt_type, lid, vt, st, orig_idx, note])

            for e in new_events:
                _log(e["type"], e["lid"])

            for e in tx_events:
                _log(e["type"], e["lid"])

            # 5) Draw overlays
            # Humans
            for h in human_tracks:
                draw_human_track(frame, h["box"], h["lid"], h["track_id"], h["role"])

            # Non-humans
            for d in other_dets:
                draw_box_with_label(frame, d["box"], f"{CLASS_NAMES.get(d['cls_id'], d['cls_id'])}")

            # Card machine anchors/lines
            pc = cm_state.get("persistent_center")
            cc = cm_state.get("current_center")
            if pc:
                cv2.circle(frame, (int(pc[0]), int(pc[1])), 6, (255, 0, 0), -1)  # persistent place (blue)
            if pc and cc:
                cv2.line(frame, (int(pc[0]), int(pc[1])), (int(cc[0]), int(cc[1])), (255, 0, 0), 2)
                if cm_state.get("is_moved"):
                    cv2.putText(frame, "CARD MACHINE MOVED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

            # 6) Write video and frame dumps
            if t_now + 1e-6 >= next_write_t:
                out.write(frame)
                frame_path = frames_dir / f"frame_{int(round(t_now*1000)):09d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                next_write_t += write_dt

            orig_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        f_csv.close()
        out.release()
        cap.release()

    print(f"Done.\nVideo:  {output_video_path}\nEvents: {events_csv_path}\nFrames: {frames_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Counter analytics: track Customer/Employee and infer card vs cash.")
    parser.add_argument("video_path")
    parser.add_argument("--weights", default="models/weights.pt")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--out-fps", type=float, default=24.0)
    args = parser.parse_args()

    t0 = time.time()
    main(args.video_path, weights_path=args.weights, conf=args.conf, out_fps=args.out_fps)
    print(f"Time: {time.time()-t0:.2f}s")
