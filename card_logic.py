# card_logic.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math

def _center(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

def _dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

@dataclass
class CardMachineMonitor:
    """
    Tracks the persistent resting place of the Card Machine and detects meaningful moves.
    All thresholds are in pixels.
    """
    stable_radius_px: float
    stable_confirm_s: float = 2.0
    moved_threshold_px: float = 80.0
    moved_min_duration_s: float = 0.3
    rest_min_duration_s: float = 1.0

    # State
    persistent_center: Optional[Tuple[float, float]] = None
    persistent_establish_start_t: Optional[float] = None
    candidate_center: Optional[Tuple[float, float]] = None

    last_center: Optional[Tuple[float, float]] = None
    last_seen_t: float = 0.0

    move_start_t: Optional[float] = None
    is_moved: bool = False

    rest_candidate: Optional[Tuple[float, float]] = None
    rest_start_t: Optional[float] = None

    def update(self, card_boxes: List[Dict], t_now: float):
        """
        card_boxes: list of { 'box': [x1,y1,x2,y2], 'conf': float }
        Picks highest-conf detection this frame.
        """
        if not card_boxes:
            return

        # Choose the most confident detection
        best = max(card_boxes, key=lambda d: d.get("conf", 0.0))
        center = _center(best["box"])
        self.last_center = center
        self.last_seen_t = t_now

        # Initialization of persistent center
        if self.persistent_center is None:
            # Use a stability window to establish the first persistent spot
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

        # Detect moving away from persistent
        d = _dist(center, self.persistent_center)
        if d > self.moved_threshold_px:
            if self.move_start_t is None:
                self.move_start_t = t_now
            elif (t_now - self.move_start_t) >= self.moved_min_duration_s:
                self.is_moved = True
        else:
            self.move_start_t = None
            # Not moved beyond threshold; consider this as part of resting
            # Track rest candidate to allow updating persistent place
            if self.rest_candidate is None or _dist(center, self.rest_candidate) > self.stable_radius_px:
                self.rest_candidate = center
                self.rest_start_t = t_now
            else:
                if (t_now - (self.rest_start_t or t_now)) >= self.rest_min_duration_s:
                    # If we had moved before, adopt new persistent place
                    if self.is_moved and _dist(self.rest_candidate, self.persistent_center) > self.stable_radius_px:
                        self.persistent_center = self.rest_candidate
                    self.is_moved = False  # reset after rest

    def snapshot(self):
        return {
            "persistent_center": self.persistent_center,
            "current_center": self.last_center,
            "is_moved": self.is_moved
        }

@dataclass
class TransactionDecider:
    """
    Emits a 'card_transaction' once per customer session if card machine moves while customer present.
    Optional cash heuristic: if session ends with no card and we ever saw 'Cash' (1) or 'Cash During Trns' (2),
    emit 'cash_transaction'.
    """
    rearm_gap_s: float = 2.0  # if a customer leaves for this long, we forget their session
    enable_cash_heuristic: bool = True

    # lid -> session state
    sessions: Dict[int, Dict] = field(default_factory=dict)
    last_t: float = 0.0

    def _ensure_session(self, lid: int, t_now: float):
        s = self.sessions.get(lid)
        if s is None:
            s = dict(
                started_t=t_now, last_seen_t=t_now,
                card_marked=False,
                saw_cash_token=False
            )
            self.sessions[lid] = s
        else:
            s["last_seen_t"] = t_now
        return s

    def update(self,
               t_now: float,
               human_tracks: List[Dict],
               card_state: Dict,
               other_dets: List[Dict]):
        """
        human_tracks: [{lid, role, box, in_role_roi, ...}]
        card_state:   {'persistent_center', 'current_center', 'is_moved'}
        other_dets:   [{cls_id, label, conf, box}]
        """
        events = []

        # Track presence and optional cash tokens within session
        present_customers = [h for h in human_tracks if h["role"] == "customer" and h.get("in_role_roi")]
        present_lids = set()
        for h in present_customers:
            s = self._ensure_session(h["lid"], t_now)
            present_lids.add(h["lid"])

        if self.enable_cash_heuristic:
            cash_seen = any(d["cls_id"] in (1, 2) for d in other_dets)
            if cash_seen:
                for h in present_customers:
                    self.sessions[h["lid"]]["saw_cash_token"] = True

        # Card transaction decision: only when moved and at least one customer present
        if card_state.get("is_moved") and present_customers:
            # choose nearest present customer to current card center
            card_c = card_state.get("current_center")
            if card_c is not None:
                closest = min(
                    present_customers,
                    key=lambda h: _dist(card_c, ( (h["box"][0]+h["box"][2]) * 0.5, h["box"][3] ))
                )
                s = self.sessions[closest["lid"]]
                if not s["card_marked"]:
                    events.append({
                        "type": "card_transaction",
                        "lid": closest["lid"],
                        "t": t_now
                    })
                    s["card_marked"] = True

        # End/cleanup sessions (and maybe cash) for customers not present
        to_delete = []
        for lid, s in self.sessions.items():
            inactive = (t_now - s["last_seen_t"]) >= self.rearm_gap_s
            if lid not in present_lids and inactive:
                if self.enable_cash_heuristic and not s["card_marked"] and s["saw_cash_token"]:
                    events.append({
                        "type": "cash_transaction",
                        "lid": lid,
                        "t": t_now
                    })
                to_delete.append(lid)
        for lid in to_delete:
            self.sessions.pop(lid, None)

        self.last_t = t_now
        return events
