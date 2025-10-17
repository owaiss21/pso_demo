# barcode_logic.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math

CLASS_SCAN_GUN = 6  # must match your model

# ------------- helpers -------------
def _center(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

def _feet_center(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    return (0.5*(x1+x2), y2)

def _outside_square(pt, ctr, half_side):
    if pt is None or ctr is None:
        return False
    dx = abs(pt[0] - ctr[0])
    dy = abs(pt[1] - ctr[1])
    return (dx > half_side) or (dy > half_side)

# ------------- square monitor -------------
@dataclass
class BarcodeMonitor:
    """
    Square-anchor monitor for the Scan Gun.

    - Establish a persistent resting square after the gun center remains within that
      same square for `lock_confirm_s` seconds.
    - If persistent exists and the current center is OUTSIDE the square, set `outside_now=True`.
    - When the gun is put down (stays within a small square around its current center
      for `adopt_rest_s` seconds), adopt that as the new persistent square (center changes).
    """
    rest_half_side_px: float            # half side of the resting square
    lock_confirm_s: float = 1.5         # time to confirm initial resting square
    adopt_rest_s: float = 0.8           # time to adopt a new resting place

    # state
    persistent_center: Optional[Tuple[float, float]] = None
    candidate_center: Optional[Tuple[float, float]] = None
    candidate_start_t: Optional[float] = None

    last_center: Optional[Tuple[float, float]] = None
    last_seen_t: Optional[float] = None

    outside_now: bool = False

    adopt_center: Optional[Tuple[float, float]] = None
    adopt_start_t: Optional[float] = None

    def update(self, scan_boxes: List[Dict], t_now: float):
        if not scan_boxes:
            return

        best = max(scan_boxes, key=lambda d: d.get("conf", 0.0))
        c = _center(best["box"])
        self.last_center = c
        self.last_seen_t = t_now

        # Lock initial square
        if self.persistent_center is None:
            if self.candidate_center is None:
                self.candidate_center = c
                self.candidate_start_t = t_now
            else:
                # still within candidate square?
                if not _outside_square(c, self.candidate_center, self.rest_half_side_px):
                    if (t_now - (self.candidate_start_t or t_now)) >= self.lock_confirm_s:
                        self.persistent_center = self.candidate_center
                        # seed adopt state
                        self.adopt_center = self.persistent_center
                        self.adopt_start_t = t_now
                else:
                    # restart candidate
                    self.candidate_center = c
                    self.candidate_start_t = t_now
            self.outside_now = False
            return

        # We have a persistent square: are we outside it now?
        self.outside_now = _outside_square(c, self.persistent_center, self.rest_half_side_px)

        # Adoption: when it rests (wherever it is), adopt a new center after `adopt_rest_s`
        # Track stability around current center using a small square with same half-side
        if self.adopt_center is None or _outside_square(c, self.adopt_center, self.rest_half_side_px):
            self.adopt_center = c
            self.adopt_start_t = t_now
        else:
            if (t_now - (self.adopt_start_t or t_now)) >= self.adopt_rest_s:
                # adopt new resting center
                self.persistent_center = self.adopt_center
                # keep adopt window rolling
                self.adopt_start_t = t_now

    def snapshot(self):
        return {
            "persistent_center": self.persistent_center,
            "rest_half_side_px": self.rest_half_side_px,
            "current_center": self.last_center,
            "outside_now": self.outside_now,
            "locked": self.persistent_center is not None
        }

# ------------- once-per-session decider -------------
@dataclass
class BarcodeDecider:
    """
    Emit one 'barcode_used' per customer session when:
    - a customer is at the counter (role='customer' and in_role_roi=True), AND
    - the scan gun center is OUTSIDE the resting square in that frame.
    """
    rearm_gap_s: float = 2.0
    sessions: Dict[int, Dict] = field(default_factory=dict)  # lid -> {last_seen_t, marked}

    def _ensure(self, lid: int, t_now: float):
        s = self.sessions.get(lid)
        if s is None:
            s = dict(last_seen_t=t_now, marked=False)
            self.sessions[lid] = s
        else:
            s["last_seen_t"] = t_now
        return s

    def _nearest_customer(self, point_xy, customers: List[Dict]):
        # we still pick nearest to have a deterministic choice if multiple present
        return min(customers, key=lambda h: math.hypot(point_xy[0] - _feet_center(h["box"])[0],
                                                       point_xy[1] - _feet_center(h["box"])[1]))

    def update(self, t_now: float, human_tracks: List[Dict], bc_state: Dict):
        events = []

        present = [h for h in human_tracks if h["role"] == "customer" and h.get("in_role_roi")]
        present_lids = set()
        for h in present:
            self._ensure(h["lid"], t_now)
            present_lids.add(h["lid"])

        if bc_state.get("outside_now") and present:
            cc = bc_state.get("current_center")
            if cc is None:
                # fallback: just pick the first present customer
                target = present[0]
            else:
                target = self._nearest_customer(cc, present)
            s = self.sessions[target["lid"]]
            if not s["marked"]:
                events.append({"type": "barcode_used", "lid": target["lid"], "t": t_now})
                s["marked"] = True

        # rearm
        to_del = []
        for lid, s in self.sessions.items():
            if lid not in present_lids and (t_now - s["last_seen_t"]) >= self.rearm_gap_s:
                to_del.append(lid)
        for lid in to_del:
            self.sessions.pop(lid, None)

        return events
