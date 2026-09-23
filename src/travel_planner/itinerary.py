"""Deterministic Itinerary Builder for TripMate Hybrid Planning Pipeline.

Assembles the final ordered list of ItineraryItem objects by:
  1. Walking through each DestinationItem in order.
  2. Inserting a TRANSIT item at the start of each new destination
     (except the very first).
  3. Calling ConstraintEngine.build_day_schedule() for each day,
     passing the fused place pool and locked custom places.
  4. Applying Nearest-Neighbor route optimisation to POIs within
     each day's pool before scheduling.

The LLM receives this structured list for formatting only — it CANNOT
add, remove, or rename any ItineraryItem produced here.
"""

import logging
from typing import List, Dict, Optional

from .models import (
    DestinationItem,
    CustomPlace,
    NormalizedPlace,
    ItineraryItem,
    ItemType,
    TravelIntent,
)
from .optimizer import ConstraintEngine, nearest_neighbor_sort, haversine_km
from .destinations import get_destination_info

logger = logging.getLogger(__name__)

# Rough driving speeds by transport mode (km/h) for transit time estimation
_SPEED_KMH: Dict[str, float] = {
    "car":        70.0,
    "motorcycle": 55.0,
    "public":     45.0,
    "flight":    700.0,
}


class ItineraryBuilder:
    """Builds the deterministic itinerary from a TravelIntent + fused places.

    Args:
        places_by_destination:  Dict mapping destination name → list of
                                NormalizedPlace for that destination.
        intent:                 TravelIntent produced by extract_travel_intent.
    """

    def __init__(self):
        self._constraint_engine = ConstraintEngine()

    def build(
        self,
        intent: TravelIntent,
        places_by_destination: Dict[str, List[NormalizedPlace]],
    ) -> List[ItineraryItem]:
        """Return the complete, ordered list of ItineraryItem for the trip."""
        items: List[ItineraryItem] = []
        day_counter = 0

        for i, dest in enumerate(sorted(intent.destinations, key=lambda d: d.order)):
            # ── Inject TRANSIT at the start of a new destination ──
            if i > 0:
                prev_dest = sorted(intent.destinations, key=lambda d: d.order)[i - 1]
                transit = self._build_transit(
                    from_dest=prev_dest,
                    to_dest=dest,
                    day=day_counter + 1,
                    transportation=intent.transportation,
                )
                items.append(transit)

            # Resolve destination lat/lng from catalog if not already set
            if dest.latitude is None or dest.longitude is None:
                info = get_destination_info(dest.name)
                if info:
                    dest.latitude = info["latitude"]
                    dest.longitude = info["longitude"]

            # Pool of fused places for this destination
            dest_places = places_by_destination.get(dest.name, [])

            # Optimise route within the destination
            optimised_places = nearest_neighbor_sort(dest_places)

            # Locked custom places for this destination
            locked_for_dest = [
                cp for cp in intent.custom_places
                if (cp.target_destination or "").lower() == dest.name.lower()
                   or cp.target_destination is None
            ]

            # Build schedule for each day at this destination
            for _day_offset in range(dest.days):
                day_counter += 1

                # Locked places for *this specific day*
                locked_today = [
                    cp for cp in locked_for_dest
                    if cp.target_day is None or cp.target_day == day_counter
                ]

                day_items = self._constraint_engine.build_day_schedule(
                    day=day_counter,
                    destination_name=dest.name,
                    available_places=optimised_places,
                    locked_places=locked_today,
                )
                items.extend(day_items)

        return items

    # ------------------------------------------------------------------ #
    # Transit helper                                                        #
    # ------------------------------------------------------------------ #

    def _build_transit(
        self,
        from_dest: DestinationItem,
        to_dest: DestinationItem,
        day: int,
        transportation: str,
    ) -> ItineraryItem:
        """Create a TRANSIT ItineraryItem between two destinations."""
        # Resolve coordinates
        from_lat, from_lng = self._get_coords(from_dest)
        to_lat, to_lng = self._get_coords(to_dest)

        distance_km: Optional[float] = None
        travel_time_min: Optional[int] = None

        if from_lat is not None and to_lat is not None:
            distance_km = haversine_km(from_lat, from_lng, to_lat, to_lng)
            speed = _SPEED_KMH.get(transportation, 70.0)
            travel_time_min = max(10, int(distance_km / speed * 60))

        # Build human-readable summary
        if distance_km is not None and travel_time_min is not None:
            dist_str = f"~{distance_km:.0f} km"
            if travel_time_min < 60:
                time_str = f"~{travel_time_min} phút"
            else:
                hours = travel_time_min // 60
                minutes = travel_time_min % 60
                time_str = f"~{hours}h{minutes:02d} phút" if minutes else f"~{hours} giờ"
            transport_label = _transport_label(transportation)
            title = (
                f"Di chuyển: {from_dest.name} → {to_dest.name} "
                f"({dist_str} · {time_str} {transport_label})"
            )
        else:
            title = f"Di chuyển: {from_dest.name} → {to_dest.name}"

        return ItineraryItem(
            item_type=ItemType.TRANSIT,
            title=title,
            time_slot="08:00 - " + _estimate_arrival_slot(travel_time_min),
            day=day,
            destination_name=to_dest.name,
            description=(
                f"Khởi hành từ {from_dest.name}, đến {to_dest.name}. "
                "Nhớ mang theo hành lý và xác nhận chỗ ở trước."
            ),
            transit_info={
                "from": from_dest.name,
                "to": to_dest.name,
                "distance_km": round(distance_km, 1) if distance_km else None,
                "travel_time_min": travel_time_min,
                "transportation": transportation,
            },
        )

    @staticmethod
    def _get_coords(dest: DestinationItem):
        """Return (lat, lng) or (None, None) for a DestinationItem."""
        if dest.latitude is not None and dest.longitude is not None:
            return dest.latitude, dest.longitude
        info = get_destination_info(dest.name)
        if info:
            return info.get("latitude"), info.get("longitude")
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _transport_label(mode: str) -> str:
    labels = {
        "car": "ô tô",
        "motorcycle": "xe máy",
        "public": "xe buýt/công cộng",
        "flight": "máy bay",
    }
    return labels.get(mode, mode)


def _estimate_arrival_slot(travel_time_min: Optional[int]) -> str:
    """Return a rough HH:MM arrival based on 08:00 departure."""
    if travel_time_min is None:
        return "10:00"
    total_minutes = 8 * 60 + travel_time_min
    h = total_minutes // 60
    m = total_minutes % 60
    return f"{min(h, 23):02d}:{m:02d}"
