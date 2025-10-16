import os
import csv
import sys
import math
import time
import cv2
import logging
import warnings
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict
from ultralytics import YOLO
from tqdm import tqdm

# Silence warnings and Ultralytics logs
warnings.filterwarnings("ignore")
try:
    from ultralytics.utils import LOGGER
    LOGGER.setLevel(logging.ERROR)
    from ultralytics import settings
    settings.update({"verbose": False})
except Exception:
    pass

# ----------------------------
# Geometry / utility
# ----------------------------
def is_left_of_diagonal(x: float, y: float, width: int, height: int) -> bool:
    return (width * y - height * x) > 0  # workers (left/below)

def classify_role_for_box(box_xyxy, frame_w, frame_h) -> str:
    x1, y1, x2, y2 = map(float, box_xyxy)
    cx = 0.5 * (x1 + x2); by = y2
    return "worker" if is_left_of_diagonal(cx, by, frame_w, frame_h) else "customer"

def format_hhmmss_msec(seconds: float) -> str:
    if seconds < 0 or math.isnan(seconds) or math.isinf(seconds):
        return "00:00:00.000"
    whole = int(seconds); msec = int((seconds - whole) * 1000)
    hh = whole // 3600; mm = (whole % 3600) // 60; ss = (whole % 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{msec:03d}"

# ----------------------------
# Appearance features (HSV hist)
# ----------------------------
def compute_hsv_hist(frame_bgr, box_xyxy):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist

def hist_corr(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))  # [-1..1], 1 best

# ----------------------------
# Drawing (no diagonal drawn)
# ----------------------------
def draw_track(frame, xyxy, logical_id: int, track_id: int, role: str):
    x1, y1, x2, y2 = map(int, xyxy)
    color = (0, 255, 0) if role == "customer" else (255, 128, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{role.upper()} L{logical_id} (T{track_id})"
    cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cx = (x1 + x2) // 2; by = y2
    cv2.circle(frame, (cx, by), 4, color, -1)

# ----------------------------
# ROI / polygon
# ----------------------------
def point_in_triangle(pt, tri):
    (x, y) = pt
    (x1, y1), (x2, y2), (x3, y3) = tri
    denom = ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    if denom == 0:
        return False
    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
    c = 1 - a - b
    return (a >= 0) and (b >= 0) and (c >= 0)

def feet_point_from_box(box):
    x1, y1, x2, y2 = map(float, box)
    return (0.5*(x1+x2), y2)

def build_rois(width, height, excl_frac_x=0.10, excl_frac_y=0.10):
    W, H = width, height
    customer_tri = [(0, 0), (W, 0), (W, H)]   # right/above the main diagonal
    exit_tri = [(int(W*(1-excl_frac_x)), 0), (W, 0), (W, int(H*excl_frac_y))]  # top-right wedge to exclude
    return customer_tri, exit_tri

def in_entry_roi(feet_pt, customer_tri, exit_tri):
    return point_in_triangle(feet_pt, customer_tri) and not point_in_triangle(feet_pt, exit_tri)

# ----------------------------
# Strict tracker YAML resolver
# ----------------------------
def resolve_tracker_yaml_strict(name: Optional[str]) -> str:
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

# ----------------------------
# Logical ID manager with motion + quality gating + exit hysteresis
# ----------------------------
@dataclass
class Logical:
    lid: int
    last_seen_t: float
    last_foot: Tuple[float, float]
    last_box: np.ndarray
    last_size: float
    hist: Optional[np.ndarray]
    role: str
    customer_entry_start_t: Optional[float] = None
    logged: bool = False
    touched: bool = False
    current_track_id: int = -1
    is_exiting: bool = False
    exit_since_t: Optional[float] = None
    vx: float = 0.0
    vy: float = 0.0

class LogicalIDManager:
    def __init__(self, frame_w, frame_h,
                 dwell_s=1.5,
                 max_time_gap_s=1.2,
                 exit_gap_s=3.0,
                 suppress_new_in_exit_s=2.0,
                 max_foot_dist_frac=0.12,
                 size_ratio_tol=(0.5, 2.0),
                 min_hist_corr=0.60,
                 # motion params
                 base_radius_frac=0.06,
                 speed_coef=2.5,
                 vel_alpha=0.3,
                 # low-quality update smoothing
                 lowq_alpha=0.2,
                 box_ema_alpha=0.4):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.diag = math.hypot(frame_w, frame_h)

        self.dwell_s = dwell_s
        self.max_time_gap_s = max_time_gap_s
        self.exit_gap_s = exit_gap_s
        self.suppress_new_in_exit_s = suppress_new_in_exit_s

        self.max_foot_dist = max_foot_dist_frac * self.diag
        self.size_ratio_tol = size_ratio_tol
        self.min_hist_corr = min_hist_corr

        self.base_radius = base_radius_frac * self.diag
        self.speed_coef = speed_coef
        self.vel_alpha = vel_alpha

        self.lowq_alpha = lowq_alpha         # how much to trust low-quality detections
        self.box_ema_alpha = box_ema_alpha   # smooth box updates

        self.next_lid = 1
        self.active: Dict[int, Logical] = {}
        self.recent: Dict[int, Logical] = {}
        self.recent_exit: Dict[int, Tuple[Logical, float]] = {}
        self.trackid_to_lid: Dict[int, int] = {}

    # helpers
    def _feet_point(self, box):
        x1, y1, x2, y2 = map(float, box)
        return (0.5 * (x1 + x2), y2)

    def _box_size(self, box):
        x1, y1, x2, y2 = map(float, box)
        return max(1.0, (x2 - x1) * (y2 - y1))

    def _predict_foot(self, obj: Logical, t_now: float):
        dt = max(0.0, t_now - obj.last_seen_t)
        return (obj.last_foot[0] + obj.vx * dt, obj.last_foot[1] + obj.vy * dt)

    def _search_radius(self, obj: Logical, t_now: float):
        dt = max(0.0, t_now - obj.last_seen_t)
        speed = math.hypot(obj.vx, obj.vy)
        return self.base_radius + self.speed_coef * speed * dt

    def _cost(self, pred_foot, det_foot, size_ratio, hist_c):
        dist = math.hypot(det_foot[0]-pred_foot[0], det_foot[1]-pred_foot[1])
        app_cost = 1.0 - max(-1.0, min(1.0, hist_c)) * 0.5 - 0.5  # [-1,1] -> [1,0]
        size_cost = abs(math.log(max(1e-3, size_ratio)))
        # weights
        return 1.0 * dist + 0.4 * app_cost + 0.2 * size_cost, dist

    # pool matcher
    def _match_pool(self, pool: Dict[int, Logical], det_foot, det_size, det_hist, t_now, relax=False):
        best_lid, best_cost, best_dist = None, 1e18, 1e18
        for lid, obj in pool.items():
            pred = self._predict_foot(obj, t_now)
            radius = 1.5*self._search_radius(obj, t_now) if relax else self._search_radius(obj, t_now)
            size_ratio = det_size / max(1.0, obj.last_size)
            if not (self.size_ratio_tol[0] <= size_ratio <= self.size_ratio_tol[1]):
                continue
            hc = hist_corr(det_hist, obj.hist)
            if (not relax) and (hc < self.min_hist_corr):
                continue
            cost, dist = self._cost(pred, det_foot, size_ratio, hc)
            if dist <= radius and cost < best_cost:
                best_lid, best_cost, best_dist = lid, cost, dist
        return best_lid

    def assign(self, frame_bgr, detections, t_now, customer_tri, exit_tri,
               min_area_frac: float, min_aspect: float, max_aspect: float):
        # mark untouched
        for obj in self.active.values():
            obj.touched = False

        frame_area = float(self.frame_w * self.frame_h)
        new_map = {}
        results = []

        exit_guard_lids = [lid for lid, (_o, t0) in self.recent_exit.items()
                           if t_now - t0 <= self.suppress_new_in_exit_s]
        assigned_lids = set()

        for det in detections:
            tid = det["track_id"]; box = det["box"]; role = det["role"]

            # quality checks
            x1, y1, x2, y2 = map(float, box)
            bw, bh = max(1.0, x2-x1), max(1.0, y2-y1)
            area_frac = (bw * bh) / max(1.0, frame_area)
            aspect = bh / bw
            low_quality = (area_frac < min_area_frac) or not (min_aspect <= aspect <= max_aspect)

            det_foot = self._feet_point(box)
            det_size = self._box_size(box)
            det_hist = compute_hsv_hist(frame_bgr, box)
            in_exit_now = point_in_triangle(det_foot, exit_tri)

            # 1) Try ACTIVE (motion-aware)
            lid = self._match_pool(self.active, det_foot, det_size, det_hist, t_now, relax=False)
            obj = None
            if lid is not None and lid not in assigned_lids:
                obj = self.active[lid]
            else:
                # 2) Try RECENT
                lid = self._match_pool(self.recent, det_foot, det_size, det_hist, t_now, relax=False)
                if lid is not None and lid not in assigned_lids:
                    obj = self.recent.pop(lid)
                    self.active[lid] = obj
                else:
                    # 3) Try EXIT-RECENT (relaxed)
                    if in_exit_now:
                        lid = self._match_pool({k:v for k,(v,_) in self.recent_exit.items()},
                                               det_foot, det_size, det_hist, t_now, relax=True)
                        if lid is not None and lid not in assigned_lids:
                            obj, _ = self.recent_exit.pop(lid)
                            self.active[lid] = obj

            # 4) Create new logical? Only if NOT in exit-grace and NOT low-quality
            if obj is None:
                if in_exit_now and exit_guard_lids:
                    # suppress new IDs in exit wedge during grace
                    continue
                if low_quality:
                    # suppress brand-new IDs from low-quality boxes anywhere
                    continue
                # make a new logical
                lid = lid if lid is not None else None
                lid = lid or (max(self.active.keys() | self.recent.keys() | self.recent_exit.keys(), default=0) + 1)
                obj = Logical(
                    lid=lid, last_seen_t=t_now, last_foot=det_foot,
                    last_box=np.array(box, dtype=float), last_size=det_size,
                    hist=det_hist, role=role, customer_entry_start_t=None,
                    logged=False, touched=False, current_track_id=tid,
                    is_exiting=in_exit_now, exit_since_t=(t_now if in_exit_now else None),
                    vx=0.0, vy=0.0
                )
                self.active[lid] = obj

            # Update velocity (use prediction blending if low-quality)
            dt = max(1e-3, t_now - obj.last_seen_t)
            pred_foot = self._predict_foot(obj, t_now)
            foot_for_update = (
                (1.0 - self.lowq_alpha) * pred_foot[0] + self.lowq_alpha * det_foot[0],
                (1.0 - self.lowq_alpha) * pred_foot[1] + self.lowq_alpha * det_foot[1]
            ) if low_quality else det_foot

            inst_vx = (foot_for_update[0] - obj.last_foot[0]) / dt
            inst_vy = (foot_for_update[1] - obj.last_foot[1]) / dt
            obj.vx = (1 - self.vel_alpha) * obj.vx + self.vel_alpha * inst_vx
            obj.vy = (1 - self.vel_alpha) * obj.vy + self.vel_alpha * inst_vy

            prev_role = obj.role
            obj.last_seen_t = t_now
            obj.last_foot = foot_for_update

            # Smooth box updates to resist arm-only crops
            if obj.last_box is None or low_quality:
                obj.last_box = (1.0 - self.box_ema_alpha) * obj.last_box + self.box_ema_alpha * np.array(box, dtype=float)
            else:
                obj.last_box = (1.0 - self.box_ema_alpha) * obj.last_box + self.box_ema_alpha * np.array(box, dtype=float)

            obj.last_size = self._box_size(obj.last_box)
            if det_hist is not None:
                obj.hist = det_hist if obj.hist is None else (obj.hist * 0.9 + det_hist * 0.1)
            obj.role = role
            obj.current_track_id = tid
            obj.touched = True

            if in_exit_now:
                if not obj.is_exiting:
                    obj.is_exiting, obj.exit_since_t = True, t_now
            else:
                obj.is_exiting = False

            # Dwell only if in ENTRY ROI
            if in_entry_roi(foot_for_update, customer_tri, exit_tri):
                if obj.customer_entry_start_t is None:
                    obj.customer_entry_start_t = t_now
            else:
                obj.customer_entry_start_t = None

            results.append({"logical_id": obj.lid, "track_id": tid, "box": obj.last_box.copy(), "role": obj.role, "prev_role": prev_role})
            new_map[tid] = obj.lid
            assigned_lids.add(obj.lid)

        # Move untouched actives → recent/recent_exit
        for lid, obj in list(self.active.items()):
            if not obj.touched:
                if obj.is_exiting and obj.exit_since_t is not None:
                    self.recent_exit[lid] = (obj, obj.exit_since_t)
                else:
                    self.recent[lid] = obj
                del self.active[lid]

        # Expire recents
        for lid, obj in list(self.recent.items()):
            if t_now - obj.last_seen_t > self.max_time_gap_s:
                del self.recent[lid]
        for lid, (obj, _t0) in list(self.recent_exit.items()):
            if t_now - obj.last_seen_t > self.exit_gap_s:
                del self.recent_exit[lid]

        self.trackid_to_lid = new_map
        return results

    def ready_to_log(self, lid, t_now):
        obj = self.active.get(lid)
        if not obj or obj.logged or obj.role != "customer" or obj.customer_entry_start_t is None:
            return False
        return (t_now - obj.customer_entry_start_t) >= self.dwell_s

    def mark_logged(self, lid):
        if lid in self.active:
            self.active[lid].logged = True

# -------------
# Main
# -------------
def main(
    video_path: str,
    conf: float = 0.5,                       # raised default to reduce spurious boxes
    out_fps: Optional[float] = 2.0,
    tracker_name: str = "botsort",
    infer_fps: Optional[float] = None,
    excl_frac_x: float = 0.10,
    excl_frac_y: float = 0.10,
    dwell_s: float = 1.5,
    max_time_gap_s: float = 1.5,
    exit_gap_s: float = 3.5,
    suppress_new_in_exit_s: float = 2.5,
    max_foot_dist_frac: float = 0.12,
    min_hist_corr: float = 0.65,
    base_radius_frac: float = 0.07,
    speed_coef: float = 3.0,
    vel_alpha: float = 0.35,
    # detection quality gates (important for waving hands)
    min_area_frac: float = 0.0004,           # reject tiny boxes
    min_aspect: float = 1.2,                 # h/w lower bound (person boxes are tall)
    max_aspect: float = 5.0                  # h/w upper bound (avoid ultra-tall glitches)
):
    video_name = Path(video_path).stem
    run_root = Path(f"output/{video_name}")
    frames_dir = run_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = str(run_root / "video_with_detection.mp4")
    events_csv_path = str(run_root / "customer_events.csv")

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

    print(f"Input video FPS: {in_fps:.2f}")

    if out_fps is None or not np.isfinite(out_fps) or out_fps <= 0:
        out_fps = 2.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, float(out_fps), (width, height))

    f_csv = open(events_csv_path, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["event","logical_id","tracker_id","video_time","system_time_utc",
                     "orig_frame_idx","role","x1","y1","x2","y2"])

    start_wallclock = datetime.now(timezone.utc)
    pbar = tqdm(total=frame_count if frame_count > 0 else None, desc="Tracking", unit="frm")

    next_write_t = 0.0
    write_dt = 1.0 / max(0.1, float(out_fps))

    if infer_fps is None or not np.isfinite(infer_fps) or infer_fps <= 0:
        infer_every_frame = True; infer_dt = 0.0; next_infer_t = 0.0
    else:
        infer_every_frame = False; infer_dt = 1.0 / float(infer_fps); next_infer_t = 0.0

    try:
        tracker_yaml = resolve_tracker_yaml_strict(tracker_name)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    customer_tri, exit_tri = build_rois(width, height, excl_frac_x=excl_frac_x, excl_frac_y=excl_frac_y)

    lidm = LogicalIDManager(
        frame_w=width, frame_h=height,
        dwell_s=dwell_s,
        max_time_gap_s=max_time_gap_s,
        exit_gap_s=exit_gap_s,
        suppress_new_in_exit_s=suppress_new_in_exit_s,
        max_foot_dist_frac=max_foot_dist_frac,
        min_hist_corr=min_hist_corr,
        base_radius_frac=base_radius_frac,
        speed_coef=speed_coef,
        vel_alpha=vel_alpha
    )

    orig_idx = 0
    last_infer_t = -1.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_now = orig_idx / in_fps
            need_infer = infer_every_frame or (t_now + 1e-6 >= next_infer_t)
            if (t_now + 1e-6 >= next_write_t) and (t_now - last_infer_t > 1e-6):
                need_infer = True

            if need_infer:
                results = model.track(
                    frame,
                    persist=True,
                    tracker="tracker/botsort_custom.yaml",
                    classes=[0],
                    conf=conf,
                    verbose=False
                )

                detections = []
                if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
                    boxes = results[0].boxes
                    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
                    ids = None
                    if boxes.id is not None:
                        ids = boxes.id.cpu().numpy().astype(int).flatten() if hasattr(boxes.id, "cpu") else boxes.id.astype(int).flatten()
                    # optional per-box conf filtering (already filtered by 'conf' in model.track, but keep for safety)
                    confs = None
                    if hasattr(boxes, "conf") and boxes.conf is not None:
                        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf

                    for i in range(len(xyxy)):
                        if confs is not None and confs[i] < conf:
                            continue
                        box_xyxy = xyxy[i]
                        track_id = int(ids[i]) if ids is not None and i < len(ids) else -1
                        role = classify_role_for_box(box_xyxy, width, height)
                        detections.append({"track_id": track_id, "box": box_xyxy, "role": role})

                assigned = lidm.assign(
                    frame, detections, t_now, customer_tri, exit_tri,
                    min_area_frac=min_area_frac, min_aspect=min_aspect, max_aspect=max_aspect
                )

                for det in assigned:
                    lid = det["logical_id"]; tid = det["track_id"]
                    draw_track(frame, det["box"], lid, tid, det["role"])

                    if lidm.ready_to_log(lid, t_now):
                        x1, y1, x2, y2 = map(int, det["box"])
                        video_time_str = format_hhmmss_msec(t_now)
                        system_time_str = (start_wallclock + timedelta(seconds=t_now)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                        writer.writerow([
                            "customer_at_counter", lid, tid, video_time_str, system_time_str,
                            orig_idx, det["role"], x1, y1, x2, y2
                        ])
                        f_csv.flush()
                        lidm.mark_logged(lid)

                last_infer_t = t_now
                if not infer_every_frame:
                    next_infer_t += infer_dt

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
        cap.release()
        out.release()

    print(f"Processing complete.\nFrames dir: {frames_dir}\nVideo:      {output_video_path}\nEvents:     {events_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="BoT-SORT (strict) + motion-aware logical IDs + quality gating; entry ROI excludes top-right exit wedge."
    )
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--conf", type=float, default=0.45, help="YOLO confidence threshold (default: 0.4)")
    parser.add_argument("--out-fps", type=float, default=24.0)
    parser.add_argument("--tracker", type=str, default="botsort")
    parser.add_argument("--infer-fps", type=float, default=None)
    parser.add_argument("--excl-frac-x", type=float, default=0.10)
    parser.add_argument("--excl-frac-y", type=float, default=0.10)
    parser.add_argument("--dwell-s", type=float, default=1.5)
    parser.add_argument("--max-time-gap-s", type=float, default=1.5)
    parser.add_argument("--exit-gap-s", type=float, default=3.5)
    parser.add_argument("--suppress-new-in-exit-s", type=float, default=2.5)
    parser.add_argument("--max-foot-dist-frac", type=float, default=0.12)
    parser.add_argument("--min-hist-corr", type=float, default=0.65)
    parser.add_argument("--base-radius-frac", type=float, default=0.07)
    parser.add_argument("--speed-coef", type=float, default=3.0)
    parser.add_argument("--vel-alpha", type=float, default=0.35)
    parser.add_argument("--min-area-frac", type=float, default=0.0004, help="Reject boxes smaller than this frame-area fraction")
    parser.add_argument("--min-aspect", type=float, default=1.2, help="Min h/w for person boxes")
    parser.add_argument("--max-aspect", type=float, default=5.0, help="Max h/w for person boxes")
    args = parser.parse_args()

    t0 = time.time()
    main(
        args.video_path,
        conf=args.conf,
        out_fps=args.out_fps,
        tracker_name=args.tracker,
        infer_fps=args.infer_fps,
        excl_frac_x=args.excl_frac_x,
        excl_frac_y=args.excl_frac_y,
        dwell_s=args.dwell_s,
        max_time_gap_s=args.max_time_gap_s,
        exit_gap_s=args.exit_gap_s,
        suppress_new_in_exit_s=args.suppress_new_in_exit_s,
        max_foot_dist_frac=args.max_foot_dist_frac,
        min_hist_corr=args.min_hist_corr,
        base_radius_frac=args.base_radius_frac,
        speed_coef=args.speed_coef,
        vel_alpha=args.vel_alpha,
        min_area_frac=args.min_area_frac,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect
    )
    print(f"Time taken: {time.time() - t0:.2f}s")
