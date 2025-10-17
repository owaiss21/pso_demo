# card_logic.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math

# Class IDs
CLASS_CARD_MACHINE = 0
CLASS_CASH_REGISTER_OPEN = 3

def _center(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

def _feet_center(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    return (0.5*(x1+x2), y2)

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
        if not card_boxes:
            return

        best = max(card_boxes, key=lambda d: d.get("conf", 0.0))
        center = _center(best["box"])
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

        # Detect “moved away” from persistent place
        d = _dist(center, self.persistent_center)
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
                    self.is_moved = False  # reset after proper rest

    def snapshot(self):
        return {
            "persistent_center": self.persistent_center,
            "current_center": self.last_center,
            "is_moved": self.is_moved
        }

@dataclass
class TransactionDecider:
    """
    Emits exactly one of {'card_transaction','cash_transaction'} per customer session.

    Primary signals:
      - Card: when the Card Machine is moved while a customer is at the counter. (Optional/redundant if machine is fixed.)
      - Cash: when 'Cash Register OPEN' (class 4) is detected while a customer is at the counter.

    Fallback:
      - If a customer session lasts >= fallback_min_dwell_s and ends with neither cash nor card marked,
        mark it as 'card_transaction' on session end.

    Sessions reset after the customer is absent for rearm_gap_s seconds.
    """
    rearm_gap_s: float = 2.0
    fallback_min_dwell_s: float = 4.5  # 4–5s per your requirement

    # lid -> session state
    sessions: Dict[int, Dict] = field(default_factory=dict)
    last_t: float = 0.0

    def _ensure_session(self, lid: int, t_now: float):
        s = self.sessions.get(lid)
        if s is None:
            s = dict(
                started_t=t_now,     # when customer_at_counter fired for this LID
                last_seen_t=t_now,   # last frame we saw this customer in ROI
                card_marked=False,
                cash_marked=False
            )
            self.sessions[lid] = s
        else:
            s["last_seen_t"] = t_now
        return s

    def _nearest_customer(self, point_xy, customers: List[Dict]):
        return min(
            customers,
            key=lambda h: _dist(point_xy, _feet_center(h["box"]))
        )

    def update(self,
               t_now: float,
               human_tracks: List[Dict],
               card_state: Dict,
               other_dets: List[Dict]):
        events = []

        # Sessions for customers currently at counter
        present_customers = [h for h in human_tracks if h["role"] == "customer" and h.get("in_role_roi")]
        present_lids = set()
        for h in present_customers:
            self._ensure_session(h["lid"], t_now)
            present_lids.add(h["lid"])

        # --- PRIMARY: CARD by movement (kept for completeness; can be disabled if undesired) ---
        if card_state.get("is_moved") and present_customers:
            card_c = card_state.get("current_center")
            if card_c is not None:
                closest = self._nearest_customer(card_c, present_customers)
                s = self.sessions[closest["lid"]]
                if not s["card_marked"] and not s["cash_marked"]:
                    events.append({"type": "card_transaction", "lid": closest["lid"], "t": t_now, "note": "card_move"})
                    s["card_marked"] = True

        # --- PRIMARY: CASH by 'Cash Register OPEN' ---
        cash_open_dets = [d for d in other_dets if d.get("cls_id") == CLASS_CASH_REGISTER_OPEN]
        if cash_open_dets and present_customers:
            best_open = max(cash_open_dets, key=lambda d: d.get("conf", 0.0))
            reg_c = _center(best_open["box"])
            closest = self._nearest_customer(reg_c, present_customers)
            s = self.sessions[closest["lid"]]
            if not s["card_marked"] and not s["cash_marked"]:
                events.append({"type": "cash_transaction", "lid": closest["lid"], "t": t_now, "note": "cash_open"})
                s["cash_marked"] = True

        # --- CLEANUP & FALLBACK on session end ---
        # Any sessions whose customer is no longer present and has been absent long enough?
        to_delete = []
        for lid, s in self.sessions.items():
            inactive = (t_now - s["last_seen_t"]) >= self.rearm_gap_s
            if lid not in present_lids and inactive:
                # Fallback: long-enough session with no cash or card → mark card
                dwell = max(0.0, s["last_seen_t"] - s["started_t"])
                if (not s["cash_marked"]) and (not s["card_marked"]) and (dwell >= self.fallback_min_dwell_s):
                    events.append({"type": "card_transaction", "lid": lid, "t": s["last_seen_t"], "note": "fallback_dwell"})
                    # No need to set card_marked (we are deleting), but harmless if we do:
                    s["card_marked"] = True
                to_delete.append(lid)

        for lid in to_delete:
            self.sessions.pop(lid, None)

        self.last_t = t_now
        return events
