# tracker_core.py
import math
import numpy as np
import cv2
import warnings
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from ultralytics import YOLO

warnings.filterwarnings("ignore")
try:
    from ultralytics.utils import LOGGER
    LOGGER.setLevel(logging.ERROR)
    from ultralytics import settings
    settings.update({"verbose": False})
except Exception:
    pass

CLASS_NAMES = {
    0: "Card Machine",
    1: "Cash",
    2: "Cash During Trns",
    3: "Cash Register CLOSED",
    4: "Cash Register OPEN",
    5: "Customer",
    6: "Employee",
    7: "Scan Gun",
}
HUMAN_CLASS_IDS = {5, 6}  # Customer, Employee

# ---------- Geometry / ROI ----------
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

def build_rois(width, height, excl_frac_x=0.10, excl_frac_y=0.10):
    """
    Two triangles split by diagonal (0,0)->(W,H) for event-side gating only:
      - customer_tri: 'above/right' of diagonal
      - employee_tri: 'below/left' of diagonal
    exit_tri: small top-right wedge to suppress false entries.
    """
    W, H = width, height
    customer_tri = [(0, 0), (W, 0), (W, H)]
    employee_tri = [(0, 0), (0, H), (W, H)]
    exit_tri = [(int(W*(1-excl_frac_x)), 0), (W, 0), (W, int(H*excl_frac_y))]
    return customer_tri, employee_tri, exit_tri

def feet_point_from_box(box):
    x1, y1, x2, y2 = map(float, box)
    return (0.5*(x1+x2), y2)

def in_role_roi(feet_pt, role_lower: str, customer_tri, employee_tri, exit_tri):
    base_tri = customer_tri if role_lower == "customer" else employee_tri
    return point_in_triangle(feet_pt, base_tri) and not point_in_triangle(feet_pt, exit_tri)

# ---------- Appearance features ----------
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
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))

# ---------- Logical IDs ----------
@dataclass
class Logical:
    lid: int
    last_seen_t: float
    last_foot: Tuple[float, float]
    last_box: np.ndarray
    last_size: float
    hist: Optional[np.ndarray]
    role: str
    roi_entry_start_t: Optional[float] = None
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
                 base_radius_frac=0.06,
                 speed_coef=2.5,
                 vel_alpha=0.3,
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

        self.lowq_alpha = lowq_alpha
        self.box_ema_alpha = box_ema_alpha

        self.active: Dict[int, Logical] = {}
        self.recent: Dict[int, Logical] = {}
        self.recent_exit: Dict[int, Tuple[Logical, float]] = {}
        self.trackid_to_lid: Dict[int, int] = {}

    def _feet_point(self, box):
        return feet_point_from_box(box)

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
        app_cost = 1.0 - max(-1.0, min(1.0, hist_c)) * 0.5 - 0.5
        size_cost = abs(math.log(max(1e-3, size_ratio)))
        return 1.0 * dist + 0.4 * app_cost + 0.2 * size_cost, dist

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

    def assign(self, frame_bgr, detections: List[Dict], t_now: float,
               customer_tri, employee_tri, exit_tri,
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

            # 1) Try ACTIVE
            lid = self._match_pool(self.active, det_foot, det_size, det_hist, t_now, relax=False)
            obj = None
            if lid is not None and lid not in assigned_lids:
                obj = self.active[lid]
            else:
                # 2) RECENT
                lid = self._match_pool(self.recent, det_foot, det_size, det_hist, t_now, relax=False)
                if lid is not None and lid not in assigned_lids:
                    obj = self.recent.pop(lid)
                    self.active[lid] = obj
                else:
                    # 3) EXIT-RECENT (relaxed)
                    if in_exit_now:
                        lid = self._match_pool({k:v for k,(v,_) in self.recent_exit.items()},
                                               det_foot, det_size, det_hist, t_now, relax=True)
                        if lid is not None and lid not in assigned_lids:
                            obj, _ = self.recent_exit.pop(lid)
                            self.active[lid] = obj

            # 4) New logical if allowed
            if obj is None:
                if in_exit_now and exit_guard_lids:
                    continue
                if low_quality:
                    continue
                lid = lid if lid is not None else None
                lid = lid or (max(self.active.keys() | self.recent.keys() | self.recent_exit.keys(), default=0) + 1)
                obj = Logical(
                    lid=lid, last_seen_t=t_now, last_foot=det_foot,
                    last_box=np.array(box, dtype=float), last_size=det_size,
                    hist=det_hist, role=role, roi_entry_start_t=None,
                    logged=False, touched=False, current_track_id=tid,
                    is_exiting=in_exit_now, exit_since_t=(t_now if in_exit_now else None),
                    vx=0.0, vy=0.0
                )
                self.active[lid] = obj

            # update velocity & state
            dt = max(1e-3, t_now - obj.last_seen_t)
            pred_foot = self._predict_foot(obj, t_now)
            inst_vx = (det_foot[0] - obj.last_foot[0]) / dt
            inst_vy = (det_foot[1] - obj.last_foot[1]) / dt
            obj.vx = (1 - 0.3) * obj.vx + 0.3 * inst_vx
            obj.vy = (1 - 0.3) * obj.vy + 0.3 * inst_vy

            prev_role = obj.role
            obj.last_seen_t = t_now
            obj.last_foot = det_foot

            obj.last_box = (1.0 - 0.4) * obj.last_box + 0.4 * np.array(box, dtype=float)
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

            # dwell by role-appropriate ROI
            if in_role_roi(det_foot, obj.role.lower(), customer_tri, employee_tri, exit_tri):
                if obj.roi_entry_start_t is None:
                    obj.roi_entry_start_t = t_now
            else:
                obj.roi_entry_start_t = None

            results.append({"logical_id": obj.lid, "track_id": tid, "box": obj.last_box.copy(),
                            "role": obj.role, "prev_role": prev_role,
                            "in_role_roi": obj.roi_entry_start_t is not None})

            new_map[tid] = obj.lid
            assigned_lids.add(obj.lid)

        # Move untouched actives
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
        if not obj or obj.logged or obj.roi_entry_start_t is None:
            return False
        if obj.role.lower() not in {"customer", "employee"}:
            return False
        return (t_now - obj.roi_entry_start_t) >= self.dwell_s

    def mark_logged(self, lid):
        if lid in self.active:
            self.active[lid].logged = True

# ---------- Tracker Engine ----------
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

class TrackerEngine:
    def __init__(self,
                 weights_path: str = "models/weights.pt",
                 conf: float = 0.45,
                 tracker_name: str = "botsort",
                 frame_w: int = 1920,
                 frame_h: int = 1080,
                 excl_frac_x: float = 0.10,
                 excl_frac_y: float = 0.10,
                 dwell_s: float = 1.5,
                 max_time_gap_s: float = 1.5,
                 exit_gap_s: float = 3.5,
                 min_area_frac: float = 0.0004,
                 min_aspect: float = 1.2,
                 max_aspect: float = 5.0):
        self.model = YOLO(weights_path)
        self.conf = conf
        self.tracker_yaml = resolve_tracker_yaml_strict(tracker_name)
        self.frame_w = frame_w
        self.frame_h = frame_h
        (self.customer_tri,
         self.employee_tri,
         self.exit_tri) = build_rois(frame_w, frame_h, excl_frac_x, excl_frac_y)

        self.lidm = LogicalIDManager(
            frame_w=frame_w, frame_h=frame_h,
            dwell_s=dwell_s, max_time_gap_s=max_time_gap_s,
            exit_gap_s=exit_gap_s, suppress_new_in_exit_s=2.5,
            max_foot_dist_frac=0.12, min_hist_corr=0.65,
            base_radius_frac=0.07, speed_coef=3.0, vel_alpha=0.35,
            lowq_alpha=0.2, box_ema_alpha=0.4
        )
        self.min_area_frac = min_area_frac
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

    def step(self, frame_bgr, t_now: float):
        """
        Returns:
          dict with:
            human_tracks: [{lid, role, track_id, box, in_role_roi}]
            nonhuman: [{cls_id, label, conf, box}]
            new_events: [{'type': 'customer_at_counter'|'employee_at_counter', ...}]
        """
        results = self.model.track(
            frame_bgr, persist=True, tracker=self.tracker_yaml, conf=self.conf, verbose=False
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
                if confs is not None and confs[i] < self.conf:
                    continue
                box_xyxy = xyxy[i]
                track_id = int(ids[i]) if ids is not None and i < len(ids) else -1
                cls_id = int(clses[i]) if clses is not None and i < len(clses) else -1
                label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                if cls_id in HUMAN_CLASS_IDS:
                    role = "customer" if cls_id == 5 else "employee"
                    human_dets.append({"track_id": track_id, "box": box_xyxy, "role": role, "cls_id": cls_id})
                else:
                    other_dets.append({"cls_id": cls_id, "label": label, "conf": float(confs[i]) if confs is not None else 0.0, "box": box_xyxy})

        # Assign logical IDs to humans
        assigned = self.lidm.assign(
            frame_bgr, human_dets, t_now,
            self.customer_tri, self.employee_tri, self.exit_tri,
            min_area_frac=self.min_area_frac, min_aspect=self.min_aspect, max_aspect=self.max_aspect
        )

        # Build human tracks and detect dwell events
        human_tracks = []
        new_events = []
        for det in assigned:
            lid = det["logical_id"]
            in_roi = det["in_role_roi"]
            human_tracks.append({
                "lid": lid,
                "role": det["role"],
                "track_id": det["track_id"],
                "box": det["box"],
                "in_role_roi": in_roi
            })
            if self.lidm.ready_to_log(lid, t_now):
                evt_type = f"{det['role'].lower()}_at_counter"
                new_events.append({
                    "type": evt_type,
                    "lid": lid,
                    "track_id": det["track_id"],
                    "box": det["box"],
                    "t": t_now
                })
                self.lidm.mark_logged(lid)

        return {
            "human_tracks": human_tracks,
            "nonhuman": other_dets,
            "new_events": new_events
        }
