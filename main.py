# main.py
import os
import sys
import csv
import time
import math
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
from ultralytics import YOLO

# Local modules you already have in this project:
# - tracker_logic.py  (provides TrackerLogic, CLASS_NAMES, HUMAN_CLASS_IDS)
# - card_logic.py     (provides CardMachineMonitor, TransactionDecider)
# - barcode_logic.py  (provides BarcodeMonitor with gap-reappear robustness, BarcodeDecider)
from tracker_logic import TrackerLogic, CLASS_NAMES, HUMAN_CLASS_IDS
from card_logic import CardMachineMonitor, TransactionDecider
from barcode_logic import BarcodeMonitor, BarcodeDecider


# ----------------------------
# Tracker YAML resolution
# ----------------------------
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

    # If it's not a known name, interpret as a path (validated by caller)
    return req


def resolve_tracker_yaml(user_arg: str):
    """
    If user passes a path that exists, use it.
    Otherwise, try strict resolver for builtin names ('botsort', 'strongsort').
    """
    if user_arg:
        p = Path(user_arg)
        if p.exists():
            return str(p)
    return resolve_tracker_yaml_strict(user_arg or "botsort")


# ----------------------------
# Small utilities
# ----------------------------
def format_hhmmss_msec(seconds: float) -> str:
    if seconds < 0 or math.isnan(seconds) or math.isinf(seconds):
        return "00:00:00.000"
    whole = int(seconds)
    msec = int((seconds - whole) * 1000)
    hh = whole // 3600
    mm = (whole % 3600) // 60
    ss = (whole % 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{msec:03d}"


def draw_human_track(frame, xyxy, logical_id: int, track_id: int, role: str):
    x1, y1, x2, y2 = map(int, xyxy)
    color = (0, 255, 0) if role.lower() == "customer" else (255, 128, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{role.upper()} L{logical_id} (T{track_id})"
    cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cx = (x1 + x2) // 2
    by = y2
    cv2.circle(frame, (cx, by), 4, color, -1)


def draw_box_with_label(frame, xyxy, label: str):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
    cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)


# ----------------------------
# Overlay banners (transient)
# ----------------------------
def add_overlay(active_overlays, text: str, pos: str, now_t: float, dur_s: float = 2.0):
    """
    pos: 'tl' (top-left) or 'tr' (top-right)
    """
    active_overlays.append({"text": text, "pos": pos, "end_t": now_t + dur_s})


def draw_overlays(frame, active_overlays, now_t: float, frame_w: int):
    # Drop expired
    active_overlays[:] = [o for o in active_overlays if o["end_t"] > now_t]
    if not active_overlays:
        return

    margin = 12
    line_gap = 6
    pad = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thick = 2

    next_y = {"tl": margin, "tr": margin}
    for o in active_overlays:
        text = o["text"]
        pos = o["pos"]
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        box_w, box_h = tw + 2 * pad, th + 2 * pad

        if pos == "tl":
            x = margin
            y = next_y["tl"]
            next_y["tl"] += box_h + line_gap
        else:
            x = frame_w - margin - box_w
            y = next_y["tr"]
            next_y["tr"] += box_h + line_gap

        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
        cv2.putText(frame, text, (x + pad, y + th + pad - 2),
                    font, scale, (255, 255, 255), thick, cv2.LINE_AA)


# ----------------------------
# Persistent counter HUD (top-left)
# ----------------------------
def draw_counter_hud(frame, counts, frame_w: int):
    """
    Always-visible HUD at top-left with dynamic counts:
      - Customers at counter (unique sessions)
      - Barcode used
      - Total transactions
      - Cash transactions
      - Card transactions
    """
    margin = 12
    pad_x = 10
    pad_y = 8
    line_gap = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thick = 2

    lines = [
        f"Customers: {counts['customers']}",
        f"Barcode Used: {counts['barcode']}",
        f"Transactions: {counts['transactions_total']}",
        f"  Cash: {counts['cash']}",
        f"  Card: {counts['card']}",
    ]

    # Measure widest line
    sizes = [cv2.getTextSize(s, font, scale, thick)[0] for s in lines]
    max_w = max(w for (w, h) in sizes)
    total_h = sum(h for (w, h) in sizes) + (len(lines) - 1) * line_gap

    x = margin
    y = margin
    box_w = max_w + 2 * pad_x
    box_h = total_h + 2 * pad_y

    # Background
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)

    # Text lines
    cur_y = y + pad_y
    for (s, (tw, th)) in zip(lines, sizes):
        cv2.putText(frame, s, (x + pad_x, cur_y + th),
                    font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        cur_y += th + line_gap


# ----------------------------
# Main
# ----------------------------
def main(
    video_path: str,
    weights_path: str = "models/weights.pt",
    conf: float = 0.45,
    tracker_yaml_arg: str = "trackers/botsort_counter.yaml",
    out_fps: float = 24.0,
    overlay_secs: float = 2.0,
    viz_anchors: bool = True,   # draw anchors/lines for card machine & scan gun
):
    # Silence Ultralytics logs
    try:
        from ultralytics.utils import LOGGER
        LOGGER.setLevel(40)
        from ultralytics import settings
        settings.update({"verbose": False})
    except Exception:
        pass

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}", file=sys.stderr)
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

    # Outputs
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
    writer.writerow(["event", "logical_id", "video_time", "system_time_utc", "frame_idx", "note"])

    # Inference setup
    model = YOLO(weights_path)
    tracker_yaml = resolve_tracker_yaml(tracker_yaml_arg)

    # Logic modules
    logic = TrackerLogic(
        frame_w=width, frame_h=height,
        dwell_s=1.5, max_time_gap_s=1.5, exit_gap_s=3.5,
        excl_frac_x=0.10, excl_frac_y=0.10,
        min_area_frac=0.0004, min_aspect=1.2, max_aspect=5.0
    )

    diag = math.hypot(width, height)

    cm_monitor = CardMachineMonitor(
        stable_radius_px=0.02 * diag,
        stable_confirm_s=2.0,
        moved_threshold_px=0.05 * diag,
        moved_min_duration_s=0.25,
        rest_min_duration_s=0.8
    )
    tx_decider = TransactionDecider(rearm_gap_s=2.0)

    # Barcode (scan gun) monitor with gap-reappear robustness
    bc_monitor = BarcodeMonitor(
        stable_radius_px=0.02 * diag,      # jitter band near cradle
        stable_confirm_s=2.0,               # time to lock initial cradle
        moved_threshold_px=0.06 * diag,     # continuous away threshold
        moved_min_duration_s=0.30,          # continuous away duration
        rest_min_duration_s=0.8,            # rest dwell to adopt new cradle
        gap_min_absent_s=0.10,              # if missing >= this and reappear far -> moved
        reappear_moved_threshold_px=0.06 * diag,
        fast_move_px=0.08 * diag            # optional jerk trigger
    )
    bc_decider = BarcodeDecider(rearm_gap_s=2.0)

    start_wallclock = datetime.now(timezone.utc)

    # Single-line tqdm
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
    active_overlays = []  # transient messages (top-right)
    overlay_duration = float(overlay_secs)

    # Persistent counters
    counts = {
        "customers": 0,
        "barcode": 0,
        "transactions_total": 0,
        "cash": 0,
        "card": 0,
    }

    orig_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_now = orig_idx / in_fps

            # ---- Inference & tracking (Ultralytics) ----
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
                    # confidence check
                    if confs is not None and confs[i] < conf:
                        continue

                    box_xyxy = xyxy[i]
                    track_id = int(ids[i]) if ids is not None and i < len(ids) else -1
                    cls_id = int(clses[i]) if clses is not None and i < len(clses) else -1
                    label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

                    if cls_id in HUMAN_CLASS_IDS:
                        # No diagonal role reassignment — class defines role
                        role = "customer" if cls_id == 5 else "employee"
                        human_dets.append({"track_id": track_id, "box": box_xyxy, "role": role})
                    else:
                        other_dets.append({
                            "cls_id": cls_id,
                            "label": label,
                            "conf": float(confs[i]) if confs is not None else 0.0,
                            "box": box_xyxy
                        })

            # ---- Tracking logic (no model calls) ----
            human_tracks, at_counter_events = logic.step_from_dets(frame, human_dets, t_now)

            # ---- Card machine & transactions ----
            card_boxes = [d for d in other_dets if d["cls_id"] == 0]  # Card Machine
            cm_monitor.update(card_boxes, t_now)
            cm_state = cm_monitor.snapshot()
            tx_events = tx_decider.update(t_now, human_tracks, cm_state, other_dets)

            # ---- Barcode monitor & 'barcode_used' events ----
            scan_boxes = [d for d in other_dets if d["cls_id"] == 7]  # Scan Gun
            bc_monitor.update(scan_boxes, t_now)
            bc_state = bc_monitor.snapshot()
            bc_events = bc_decider.update(t_now, human_tracks, bc_state)

            # ---- CSV logging helper ----
            def _log(evt_type, lid, note=""):
                vt = format_hhmmss_msec(t_now)
                st = (start_wallclock + timedelta(seconds=t_now)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                writer.writerow([evt_type, lid, vt, st, orig_idx, note])

            # ---- Events -> CSV + Overlays + COUNTS ----
            for e in at_counter_events:
                _log(e["type"], e["lid"])
                if e["type"] == "customer_at_counter":
                    counts["customers"] += 1
                    # Move this to top-right (with others)
                    add_overlay(active_overlays, f"CUSTOMER AT COUNTER (L{e['lid']})", "tr", t_now, overlay_duration)

            for e in bc_events:
                _log(e["type"], e["lid"])
                counts["barcode"] += 1
                add_overlay(active_overlays, "BARCODE GUN USED", "tr", t_now, overlay_duration)

            for e in tx_events:
                _log(e["type"], e["lid"])
                if e["type"] == "card_transaction":
                    counts["card"] += 1
                    counts["transactions_total"] += 1
                    add_overlay(active_overlays, "CARD TRANSACTION", "tr", t_now, overlay_duration)
                elif e["type"] == "cash_transaction":
                    counts["cash"] += 1
                    counts["transactions_total"] += 1
                    add_overlay(active_overlays, "CASH TRANSACTION", "tr", t_now, overlay_duration)

            # ---- Draw tracks and other boxes ----
            for h in human_tracks:
                draw_human_track(frame, h["box"], h["lid"], h["track_id"], h["role"])

            for d in other_dets:
                draw_box_with_label(frame, d["box"], d["label"])

            # Optional: visualize anchors/lines for card machine & scan gun
            if viz_anchors:
                # Card machine
                pc = cm_state.get("persistent_center")
                cc = cm_state.get("current_center")
                if pc:
                    cv2.circle(frame, (int(pc[0]), int(pc[1])), 6, (255, 0, 0), -1)  # blue
                if pc and cc:
                    cv2.line(frame, (int(pc[0]), int(pc[1])), (int(cc[0]), int(cc[1])), (255, 0, 0), 2)

                # Scan gun
                bpc = bc_state.get("persistent_center")
                bcc = bc_state.get("current_center")
                if bpc:
                    cv2.circle(frame, (int(bpc[0]), int(bpc[1])), 6, (0, 128, 255), -1)  # orange-ish
                if bpc and bcc:
                    cv2.line(frame, (int(bpc[0]), int(bpc[1])), (int(bcc[0]), int(bcc[1])), (0, 128, 255), 2)

            # ---- Draw persistent HUD (top-left) and transient banners (top-right) ----
            draw_counter_hud(frame, counts, width)
            draw_overlays(frame, active_overlays, t_now, width)

            # ---- Output ----
            if t_now + 1e-6 >= next_write_t:
                out.write(frame)
                frame_path = frames_dir / f"frame_{int(round(t_now * 1000)):09d}.jpg"
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

    parser = argparse.ArgumentParser(description="Counter analytics (YOLOv11 inference in main).")
    parser.add_argument("video_path")
    parser.add_argument("--weights", default="models/weights.pt", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.45, help="YOLO confidence threshold")
    parser.add_argument("--tracker", type=str, default="trackers/botsort_counter.yaml",
                        help="Path or name of tracker yaml (path preferred).")
    parser.add_argument("--out-fps", type=float, default=24.0, help="Output video FPS")
    parser.add_argument("--overlay-secs", type=float, default=2.0, help="Overlay banner duration in seconds")
    parser.add_argument("--no-viz-anchors", action="store_true", help="Disable anchor/line visualization")
    args = parser.parse_args()

    t0 = time.time()
    main(
        video_path=args.video_path,
        weights_path=args.weights,
        conf=args.conf,
        tracker_yaml_arg=args.tracker,
        out_fps=args.out_fps,
        overlay_secs=args.overlay_secs,
        viz_anchors=not args.no_viz_anchors,
    )
    print(f"Time: {time.time()-t0:.2f}s")
