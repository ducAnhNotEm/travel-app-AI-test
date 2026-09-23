"""Place Fusion Engine for TripMate Hybrid Planning Pipeline.

Merges three sources of POI data into a single deduplicated list, ranked
by priority:

  1. Pinned Places   — user-pinned directly from the UI (highest priority)
  2. Custom Places   — extracted from the prompt (both verified & unverified)
  3. Google Places   — AI-recommended from coordinate-based search (fill-in)

Deduplication is performed by Google place_id first, then by normalised
name similarity as a fallback.
"""

import logging
from typing import List, Optional, Dict, Any

from .models import CustomPlace, NormalizedPlace, PlaceStatus

logger = logging.getLogger(__name__)

_DEDUP_SIMILARITY_THRESHOLD = 0.8


def _norm_name(name: str) -> str:
    """Lowercase + strip Vietnamese diacritics-aware normalisation."""
    return " ".join(name.lower().split())


def _name_overlap(a: str, b: str) -> float:
    a_tok = set(_norm_name(a).split())
    b_tok = set(_norm_name(b).split())
    if not a_tok or not b_tok:
        return 0.0
    return len(a_tok & b_tok) / max(len(a_tok), len(b_tok))


class PlaceFusionEngine:
    """Merge pinned, custom, and Google-recommended places into one list.

    Args:
        pinned_places:  List of NormalizedPlace (pinned via UI).
        custom_places:  List of CustomPlace (extracted from user prompt).
        google_places:  List of raw Google Places dicts (from PlacesService).
    """

    def fuse(
        self,
        pinned_places: List[NormalizedPlace],
        custom_places: List[CustomPlace],
        google_places: List[Dict[str, Any]],
    ) -> List[NormalizedPlace]:
        """Return merged, deduplicated, priority-ordered list of NormalizedPlace."""

        merged: List[NormalizedPlace] = []
        seen_ids: set = set()
        seen_names: List[str] = []

        def _is_duplicate(np: NormalizedPlace) -> bool:
            """Return True if this place is already in merged."""
            if np.id and np.id in seen_ids:
                return True
            for existing_name in seen_names:
                if _name_overlap(np.name, existing_name) >= _DEDUP_SIMILARITY_THRESHOLD:
                    return True
            return False

        def _register(np: NormalizedPlace) -> None:
            merged.append(np)
            if np.id:
                seen_ids.add(np.id)
            seen_names.append(np.name)

        # ── Priority 1: Pinned places ──────────────────────────────
        for p in pinned_places:
            if not _is_duplicate(p):
                _register(p)

        # ── Priority 2: Custom places (verified first, then unverified)
        verified_custom = [c for c in custom_places if c.status == PlaceStatus.VERIFIED]
        unverified_custom = [c for c in custom_places if c.status == PlaceStatus.UNVERIFIED]

        for cp in verified_custom + unverified_custom:
            np = _custom_place_to_normalized(cp)
            if not _is_duplicate(np):
                _register(np)

        # ── Priority 3: Google recommended places ─────────────────
        for raw in google_places:
            np = _google_raw_to_normalized(raw)
            if not _is_duplicate(np):
                _register(np)

        logger.debug(
            "PlaceFusion: %d pinned + %d custom + %d google → %d unique",
            len(pinned_places), len(custom_places), len(google_places), len(merged),
        )
        return merged


# ──────────────────────────────────────────────────────────────────────────────
# Conversion helpers
# ──────────────────────────────────────────────────────────────────────────────

def _custom_place_to_normalized(cp: CustomPlace) -> NormalizedPlace:
    """Convert a CustomPlace to a NormalizedPlace."""
    source = (
        "custom_verified" if cp.status == PlaceStatus.VERIFIED else "custom_unverified"
    )
    return NormalizedPlace(
        id=cp.place_id or f"custom_{_norm_name(cp.name).replace(' ', '_')}",
        name=cp.name,
        category=cp.category,
        address=cp.address or "",
        latitude=cp.latitude or 0.0,
        longitude=cp.longitude or 0.0,
        rating=cp.rating or 0.0,
        user_rating_count=0,
        google_maps_uri=None,
        source=source,
    )


def _google_raw_to_normalized(raw: Dict[str, Any]) -> NormalizedPlace:
    """Convert a raw Google Places API dict to a NormalizedPlace."""
    return NormalizedPlace(
        id=raw.get("id", ""),
        name=raw.get("name", ""),
        category=raw.get("category", "attraction"),
        address=raw.get("address", ""),
        latitude=raw.get("latitude") or 0.0,
        longitude=raw.get("longitude") or 0.0,
        rating=raw.get("rating") or 0.0,
        user_rating_count=raw.get("user_rating_count", 0),
        google_maps_uri=raw.get("google_maps_uri"),
        source="google_places",
    )
