# main.py
import os, sys, csv, time, math
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
from ultralytics import YOLO

from tracker_logic import TrackerLogic, CLASS_NAMES, HUMAN_CLASS_IDS
from card_logic import CardMachineMonitor, TransactionDecider

def resolve_tracker_yaml_strict(name: str = "botsort"):
    req = (name or "botsort").lower()
    try:
        import ultralytics as ul
        base = Path(ul.__file__).parent / "cfg" / "trackers"
    except Exception:
        base = Path("ultralytics") / "cfg" / "trackers"
    if req in {"botsort", "bot", "bot-sort"}:
        p = base / "botsort.yaml"
        if not p.exists():
            raise FileNotFoundError(f"BoT-SORT YAML not found at '{p}'.")
        return str(p)
    if req in {"strongsort", "strong", "strong-sort"}:
        p = base / "strongsort.yaml"
        if not p.exists():
            raise FileNotFoundError(f"StrongSORT YAML not found at '{p}'.")
        return str(p)
    p = Path(req)
    if not p.exists():
        raise FileNotFoundError(f"Tracker YAML '{req}' not found.")
    return str(p)

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
    tracker_name: str = "botsort",
    out_fps: float = 24.0,
):
    # 0) quiet ultralytics logging
    try:
        from ultralytics.utils import LOGGER
        LOGGER.setLevel(40)  # ERROR
        from ultralytics import settings
        settings.update({"verbose": False})
    except Exception:
        pass

    # 1) Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}", file=sys.stderr)
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

    # 2) Outputs
    video_name = Path(video_path).stem
    run_root = Path(f"output/{video_name}")
    frames_dir = run_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = str(run_root / "video_with_detection.mp4")
    events_csv_path = str(run_root / "events.csv")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, float(out_fps), (width, height))

    f_csv = open(events_csv_path, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["event","logical_id","video_time","system_time_utc","frame_idx","note"])

    # 3) Inference + tracker logic
    model = YOLO(weights_path)
    tracker_yaml = resolve_tracker_yaml_strict(tracker_name)

    logic = TrackerLogic(
        frame_w=width, frame_h=height,
        dwell_s=1.5, max_time_gap_s=1.5, exit_gap_s=3.5,
        excl_frac_x=0.10, excl_frac_y=0.10,
        min_area_frac=0.0004, min_aspect=1.2, max_aspect=5.0
    )

    diag = math.hypot(width, height)
    cm_monitor = CardMachineMonitor(
        stable_radius_px=0.02 * diag,         # ~2% diag jitter band
        stable_confirm_s=2.0,
        moved_threshold_px=0.05 * diag,       # ~5% diag to consider "moved"
        moved_min_duration_s=0.25,
        rest_min_duration_s=0.8
    )
    decider = TransactionDecider(rearm_gap_s=2.0, enable_cash_heuristic=True)

    start_wallclock = datetime.now(timezone.utc)
    pbar = tqdm(
        total=frame_count if frame_count > 0 else None,
        desc="Processing",
        unit="frm",
        dynamic_ncols=True,
        position=0,
        leave=True,
        mininterval=0.25,
        maxinterval=1.0,
        smoothing=0.05,
        file=sys.stdout,
        disable=not sys.stdout.isatty(),
    )

    next_write_t = 0.0
    write_dt = 1.0 / max(0.1, float(out_fps))

    orig_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_now = orig_idx / in_fps

            # --- Inference happens HERE in main.py ---
            results = model.track(
                frame,
                persist=True,
                tracker=tracker_yaml,
                conf=conf,
                verbose=False
            )

            human_dets, other_dets = [], []
            if results and len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes is not None:
                boxes = results[0].boxes
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
                ids = boxes.id.cpu().numpy().astype(int).flatten() if getattr(boxes, "id", None) is not None else None
                confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else None
                clses = boxes.cls.cpu().numpy().astype(int).flatten() if getattr(boxes, "cls", None) is not None else None

                n = len(xyxy)
                for i in range(n):
                    if confs is not None and confs[i] < conf:
                        continue
                    box_xyxy = xyxy[i]
                    track_id = int(ids[i]) if ids is not None and i < len(ids) else -1
                    cls_id = int(clses[i]) if clses is not None and i < len(clses) else -1
                    label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

                    if cls_id in HUMAN_CLASS_IDS:
                        role = "customer" if cls_id == 5 else "employee"
                        human_dets.append({"track_id": track_id, "box": box_xyxy, "role": role})
                    else:
                        other_dets.append({"cls_id": cls_id, "label": label, "conf": float(confs[i]) if confs is not None else 0.0, "box": box_xyxy})

            # --- Pure logic step (no inference) ---
            human_tracks, at_counter_events = logic.step_from_dets(frame, human_dets, t_now)

            # --- Card machine update & transaction decisions ---
            card_boxes = [d for d in other_dets if d["cls_id"] == 0]
            cm_monitor.update(card_boxes, t_now)
            cm_state = cm_monitor.snapshot()

            tx_events = decider.update(t_now, human_tracks, cm_state, other_dets)

            # --- Log events ---
            def _log(evt_type, lid, note=""):
                vt = format_hhmmss_msec(t_now)
                st = (start_wallclock + timedelta(seconds=t_now)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                writer.writerow([evt_type, lid, vt, st, orig_idx, note])

            for e in at_counter_events:
                _log(e["type"], e["lid"])
            for e in tx_events:
                _log(e["type"], e["lid"])

            # --- Draw overlays ---
            for h in human_tracks:
                draw_human_track(frame, h["box"], h["lid"], h["track_id"], h["role"])

            for d in other_dets:
                draw_box_with_label(frame, d["box"], d["label"])

            pc = cm_state.get("persistent_center")
            cc = cm_state.get("current_center")
            if pc:
                cv2.circle(frame, (int(pc[0]), int(pc[1])), 6, (255, 0, 0), -1)
            if pc and cc:
                cv2.line(frame, (int(pc[0]), int(pc[1])), (int(cc[0]), int(cc[1])), (255, 0, 0), 2)
                if cm_state.get("is_moved"):
                    cv2.putText(frame, "CARD MACHINE MOVED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

            # --- Output video/frames ---
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
    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Counter analytics (inference in main).")
    parser.add_argument("video_path")
    parser.add_argument("--weights", default="models/weights.pt")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--tracker", type=str, default="botsort")
    parser.add_argument("--out-fps", type=float, default=24.0)
    args = parser.parse_args()

    t0 = time.time()
    main(args.video_path, weights_path=args.weights, conf=args.conf, tracker_name=args.tracker, out_fps=args.out_fps)
    print(f"Time: {time.time()-t0:.2f}s")
