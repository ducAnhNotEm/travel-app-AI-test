"""Constraint Engine & Route Optimizer for TripMate Hybrid Planning Pipeline.

Responsibilities:
  1. Haversine distance calculation between GPS coordinates.
  2. Nearest-Neighbor route optimisation for POIs within the same half-day.
  3. Constraint Engine — maps POIs to daily time slots and locks
     user-pinned / custom places to their target slots.

Slot schedule (fixed):
  breakfast   07:30 - 08:30   (restaurant | cafe)
  morning     09:00 - 11:30   (attraction)
  lunch       12:00 - 13:30   (restaurant)
  afternoon   14:30 - 17:00   (attraction)
  dusk        17:30 - 18:30   (ACTIVITY)
  dinner      19:00 - 20:30   (restaurant)
  evening     21:00 - 22:30   (cafe | nightlife | ACTIVITY)
"""

import math
import logging
from typing import List, Dict, Optional, Tuple, Any

from .models import NormalizedPlace, CustomPlace, ItineraryItem, ItemType, PlaceStatus

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Slot definitions
# ──────────────────────────────────────────────────────────────────────────────

# Each slot: (slot_name, time_range, accepted_categories, item_type)
DAILY_SLOTS: List[Tuple[str, str, List[str], ItemType]] = [
    ("breakfast", "07:30 - 08:30", ["restaurant", "cafe"], ItemType.PLACE),
    ("morning",   "09:00 - 11:30", ["attraction", "museum", "park"], ItemType.PLACE),
    ("lunch",     "12:00 - 13:30", ["restaurant", "cafe"], ItemType.PLACE),
    ("afternoon", "14:30 - 17:00", ["attraction", "museum", "park"], ItemType.PLACE),
    ("dusk",      "17:30 - 18:30", [], ItemType.ACTIVITY),      # always ACTIVITY
    ("dinner",    "19:00 - 20:30", ["restaurant"], ItemType.PLACE),
    ("evening",   "21:00 - 22:30", ["cafe", "nightlife"], ItemType.PLACE),
]

_SLOT_INDEX: Dict[str, int] = {s[0]: i for i, s in enumerate(DAILY_SLOTS)}

# Activity titles per slot used when no POI is available
_ACTIVITY_FALLBACKS: Dict[str, str] = {
    "breakfast": "Ăn sáng tại quán địa phương",
    "morning":   "Dạo bộ khám phá khu vực xung quanh",
    "lunch":     "Thưởng thức đặc sản địa phương bữa trưa",
    "afternoon": "Tự do khám phá, chụp ảnh lưu niệm",
    "dusk":      "Ngắm hoàng hôn / Dạo biển / Nghỉ ngơi",
    "dinner":    "Ăn tối và ngắm cảnh đêm",
    "evening":   "Khám phá phố đêm và thưởng thức đồ uống",
}


# ──────────────────────────────────────────────────────────────────────────────
# Haversine distance
# ──────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres between two GPS points."""
    R = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ──────────────────────────────────────────────────────────────────────────────
# Nearest-Neighbor route optimiser
# ──────────────────────────────────────────────────────────────────────────────

def nearest_neighbor_sort(places: List[NormalizedPlace]) -> List[NormalizedPlace]:
    """Sort a list of POIs using the Nearest-Neighbor heuristic.

    Starts from the first place in the list (preserves pinned/custom priority)
    and greedily picks the geographically closest unvisited POI next.

    Places without valid coordinates (lat=0, lng=0) are appended last.
    """
    if len(places) <= 1:
        return list(places)

    def _has_coords(p: NormalizedPlace) -> bool:
        return p.latitude != 0.0 or p.longitude != 0.0

    with_coords = [p for p in places if _has_coords(p)]
    without_coords = [p for p in places if not _has_coords(p)]

    if not with_coords:
        return list(places)

    ordered: List[NormalizedPlace] = [with_coords[0]]
    remaining = with_coords[1:]

    while remaining:
        last = ordered[-1]
        closest = min(
            remaining,
            key=lambda p: haversine_km(last.latitude, last.longitude, p.latitude, p.longitude),
        )
        ordered.append(closest)
        remaining.remove(closest)

    return ordered + without_coords


# ──────────────────────────────────────────────────────────────────────────────
# Constraint Engine
# ──────────────────────────────────────────────────────────────────────────────

class ConstraintEngine:
    """Assigns POIs to daily time slots, honouring user-requested locks.

    For each day, the engine:
      1. Pre-fills "locked" slots from custom/pinned places with a target_day
         and/or target_slot.
      2. Fills remaining slots from the pool of available POIs by category.
      3. Inserts ACTIVITY items for dusk and any slot without a matching POI.
    """

    def build_day_schedule(
        self,
        day: int,
        destination_name: str,
        available_places: List[NormalizedPlace],
        locked_places: List[CustomPlace],
    ) -> List[ItineraryItem]:
        """Return an ordered list of ItineraryItem for a single day.

        Args:
            day:               1-based day number in the full trip.
            destination_name:  City / destination label.
            available_places:  Pool of NormalizedPlace objects for this destination.
            locked_places:     CustomPlace objects explicitly pinned to this day.
        """
        items: List[ItineraryItem] = []
        # Track which places have already been used across slots today
        used_ids: set = set()

        # Build a mapping slot_name → locked NormalizedPlace (if any)
        slot_locks: Dict[str, Optional[NormalizedPlace]] = {}
        for cp in locked_places:
            if cp.target_slot and cp.target_slot in _SLOT_INDEX:
                np = _custom_to_norm(cp)
                slot_locks[cp.target_slot] = np
                used_ids.add(np.id)

        for slot_name, time_range, accepted_cats, default_type in DAILY_SLOTS:
            locked_np = slot_locks.get(slot_name)

            if locked_np is not None:
                # ── Locked place ──────────────────────────────────
                status = (
                    PlaceStatus.VERIFIED
                    if locked_np.source in ("custom_verified", "google_places", "pinned")
                    else PlaceStatus.UNVERIFIED
                )
                items.append(ItineraryItem(
                    item_type=ItemType.PLACE,
                    title=locked_np.name,
                    time_slot=time_range,
                    day=day,
                    destination_name=destination_name,
                    description=_build_description(locked_np, status),
                    place_details=_norm_to_dict(locked_np),
                    status=status,
                ))

            elif default_type == ItemType.ACTIVITY:
                # ── Fixed ACTIVITY slot (dusk) ────────────────────
                items.append(ItineraryItem(
                    item_type=ItemType.ACTIVITY,
                    title=_ACTIVITY_FALLBACKS[slot_name],
                    time_slot=time_range,
                    day=day,
                    destination_name=destination_name,
                    description="Hoạt động tự do, tận hưởng bầu không khí địa phương.",
                ))

            else:
                # ── Find best unused place for this slot ──────────
                candidate = self._pick_place(available_places, accepted_cats, used_ids)
                if candidate:
                    used_ids.add(candidate.id)
                    items.append(ItineraryItem(
                        item_type=ItemType.PLACE,
                        title=candidate.name,
                        time_slot=time_range,
                        day=day,
                        destination_name=destination_name,
                        description=_build_description(candidate, PlaceStatus.VERIFIED),
                        place_details=_norm_to_dict(candidate),
                        status=PlaceStatus.VERIFIED,
                    ))
                else:
                    # Fallback ACTIVITY for slots with no available POI
                    items.append(ItineraryItem(
                        item_type=ItemType.ACTIVITY,
                        title=_ACTIVITY_FALLBACKS[slot_name],
                        time_slot=time_range,
                        day=day,
                        destination_name=destination_name,
                        description="Khám phá tự do khu vực xung quanh.",
                    ))

        return items

    # ------------------------------------------------------------------ #

    @staticmethod
    def _pick_place(
        pool: List[NormalizedPlace],
        accepted_cats: List[str],
        used_ids: set,
    ) -> Optional[NormalizedPlace]:
        """Return the highest-rated unused place from the pool matching accepted_cats."""
        candidates = [
            p for p in pool
            if p.id not in used_ids and p.category in accepted_cats
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.rating or 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _custom_to_norm(cp: CustomPlace) -> NormalizedPlace:
    """Lightweight conversion for locked custom places."""
    from .fusion import _custom_place_to_normalized
    return _custom_place_to_normalized(cp)


def _norm_to_dict(p: NormalizedPlace) -> Dict[str, Any]:
    """Convert NormalizedPlace to a plain dict for ItineraryItem.place_details."""
    return {
        "id": p.id,
        "name": p.name,
        "category": p.category,
        "address": p.address,
        "latitude": p.latitude,
        "longitude": p.longitude,
        "rating": p.rating,
        "user_rating_count": p.user_rating_count,
        "google_maps_uri": p.google_maps_uri,
        "source": p.source,
    }


def _build_description(p: NormalizedPlace, status: PlaceStatus) -> str:
    """Build a short description string for an ItineraryItem."""
    parts: List[str] = []
    if p.address:
        parts.append(f"Địa chỉ: {p.address}")
    if p.rating:
        parts.append(f"Đánh giá: {p.rating}⭐ ({p.user_rating_count} lượt)")
    if status == PlaceStatus.UNVERIFIED:
        parts.append("[Địa điểm do bạn đề xuất - chưa xác thực trên bản đồ]")
    return " | ".join(parts)
