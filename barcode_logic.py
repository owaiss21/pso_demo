# barcode_logic.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math

CLASS_SCAN_GUN = 7  # must match your model

def _center(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

def _feet_center(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    return (0.5*(x1+x2), y2)

def _dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

@dataclass
class BarcodeMonitor:
    """
    Tracks the persistent resting place of the Scan Gun and raises 'is_moved' when:
      1) Continuous: center leaves the resting place by > moved_threshold_px for >= moved_min_duration_s, OR
      2) Gap-reappear: after a detection GAP of >= gap_min_absent_s, it reappears >= reappear_moved_threshold_px away
         from the resting place (or from last_center if resting not yet established).
      3) Fast jerk (optional): single-frame displacement from resting place exceeds fast_move_px.

    All thresholds are in pixels; main.py should pass diag-scaled numbers.
    """
    stable_radius_px: float
    stable_confirm_s: float = 2.0
    moved_threshold_px: float = 80.0
    moved_min_duration_s: float = 0.30
    rest_min_duration_s: float = 0.80

    # New robustness knobs for detection gaps / fast pick-ups
    gap_min_absent_s: float = 0.10
    reappear_moved_threshold_px: Optional[float] = None  # default: moved_threshold_px if None
    fast_move_px: Optional[float] = None  # if not None and distance > fast_move_px -> moved immediately

    # Internal state
    persistent_center: Optional[Tuple[float, float]] = None
    persistent_establish_start_t: Optional[float] = None
    candidate_center: Optional[Tuple[float, float]] = None

    last_center: Optional[Tuple[float, float]] = None
    last_seen_t: Optional[float] = None

    move_start_t: Optional[float] = None
    is_moved: bool = False

    rest_candidate: Optional[Tuple[float, float]] = None
    rest_start_t: Optional[float] = None

    missing_since_t: Optional[float] = None  # when detection first went missing

    def _reappear_check(self, center, t_now):
        """Trigger moved immediately if we had a gap and reappeared far from resting (or last seen)."""
        if self.missing_since_t is None or self.last_seen_t is None:
            return
        absent_dt = t_now - self.missing_since_t
        if absent_dt < max(0.0, self.gap_min_absent_s):
            return

        anchor = self.persistent_center if self.persistent_center is not None else self.last_center
        if anchor is None:
            return

        thr = self.reappear_moved_threshold_px or self.moved_threshold_px
        if _dist(center, anchor) >= thr:
            # Instant move due to reappear far away
            self.is_moved = True
            # Clear any continuous timers; we’re already “moved”
            self.move_start_t = None

    def update(self, scan_boxes: List[Dict], t_now: float):
        """
        scan_boxes: list of { 'box': [x1,y1,x2,y2], 'conf': float, 'cls_id': int==7 }
        Picks highest-confidence detection this frame (if any) and updates state.
        """
        if not scan_boxes:
            # track missing
            if self.missing_since_t is None:
                self.missing_since_t = t_now
            return

        # We have a detection this frame
        best = max(scan_boxes, key=lambda d: d.get("conf", 0.0))
        center = _center(best["box"])

        # If we were missing, run the gap-reappear trigger
        if self.missing_since_t is not None:
            self._reappear_check(center, t_now)
            self.missing_since_t = None  # reset missing flag

        # Update last seen
        self.last_center = center
        self.last_seen_t = t_now

        # Establish initial persistent place
        if self.persistent_center is None:
            if self.candidate_center is None:
                self.candidate_center = center
                self.persistent_establish_start_t = t_now
            else:
                if _dist(center, self.candidate_center) <= self.stable_radius_px:
                    if (t_now - (self.persistent_establish_start_t or t_now)) >= self.stable_confirm_s:
                        self.persistent_center = self.candidate_center
                        self.rest_candidate = self.candidate_center
                        self.rest_start_t = t_now
                else:
                    self.candidate_center = center
                    self.persistent_establish_start_t = t_now
            return

        # With a persistent center, evaluate displacement
        d = _dist(center, self.persistent_center)

        # Optional “fast jerk” immediate trigger
        if self.fast_move_px is not None and d > self.fast_move_px:
            self.is_moved = True
            self.move_start_t = None
        else:
            # Continuous movement logic
            if d > self.moved_threshold_px:
                if self.move_start_t is None:
                    self.move_start_t = t_now
                elif (t_now - self.move_start_t) >= self.moved_min_duration_s:
                    self.is_moved = True
            else:
                # Close to persistent again → consider resting
                self.move_start_t = None
                if self.rest_candidate is None or _dist(center, self.rest_candidate) > self.stable_radius_px:
                    self.rest_candidate = center
                    self.rest_start_t = t_now
                else:
                    if (t_now - (self.rest_start_t or t_now)) >= self.rest_min_duration_s:
                        # If it had moved earlier and now rests elsewhere, adopt new persistent place
                        if self.is_moved and _dist(self.rest_candidate, self.persistent_center) > self.stable_radius_px:
                            self.persistent_center = self.rest_candidate
                        # Reset move flag after a proper rest
                        self.is_moved = False

    def snapshot(self):
        return {
            "persistent_center": self.persistent_center,
            "current_center": self.last_center,
            "is_moved": self.is_moved
        }

@dataclass
class BarcodeDecider:
    """
    Emits one 'barcode_used' per customer session WHEN the scan gun is 'moved'
    while a customer is at the counter. Sessions keyed by logical customer ID (lid).
    """
    rearm_gap_s: float = 2.0
    sessions: Dict[int, Dict] = field(default_factory=dict)  # lid -> {last_seen_t, marked}

    def _ensure_session(self, lid: int, t_now: float):
        s = self.sessions.get(lid)
        if s is None:
            s = dict(last_seen_t=t_now, marked=False)
            self.sessions[lid] = s
        else:
            s["last_seen_t"] = t_now
        return s

    def _nearest_customer(self, point_xy, customers: List[Dict]):
        return min(
            customers,
            key=lambda h: _dist(point_xy, _feet_center(h["box"]))
        )

    def update(self, t_now: float, human_tracks: List[Dict], barcode_state: Dict):
        """
        human_tracks: [{lid, role, box, in_role_roi, ...}], only role='customer' & in_role_roi matter
        barcode_state: {'persistent_center', 'current_center', 'is_moved'}
        """
        events = []

        present_customers = [h for h in human_tracks if h["role"] == "customer" and h.get("in_role_roi")]
        present_lids = set()
        for h in present_customers:
            self._ensure_session(h["lid"], t_now)
            present_lids.add(h["lid"])

        # Barcode event: gun moved while customer present
        if barcode_state.get("is_moved") and present_customers:
            bc_c = barcode_state.get("current_center")
            if bc_c is not None:
                closest = self._nearest_customer(bc_c, present_customers)
                s = self.sessions[closest["lid"]]
                if not s["marked"]:
                    events.append({"type": "barcode_used", "lid": closest["lid"], "t": t_now})
                    s["marked"] = True

        # Cleanup sessions (re-arm)
        to_delete = []
        for lid, s in self.sessions.items():
            if (lid not in present_lids) and ((t_now - s["last_seen_t"]) >= self.rearm_gap_s):
                to_delete.append(lid)
        for lid in to_delete:
            self.sessions.pop(lid, None)

        return events
