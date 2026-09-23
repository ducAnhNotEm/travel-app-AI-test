"""Place Resolver for TripMate Hybrid Planning Pipeline.

Resolves user-supplied CustomPlace names against the Google Places API
(or Geoapify fallback). Verified places gain coordinates, rating, and
a Google place_id; unverified places are kept as-is with a warning flag
so they still appear in the final itinerary.

Usage:
    resolver = PlaceResolver(places_service)
    resolved = resolver.resolve_custom_places(custom_places, destination)
"""

import logging
from typing import List, Optional

from .models import CustomPlace, PlaceStatus
from .places import PlacesService, PlacesAPIError

logger = logging.getLogger(__name__)

# Minimum similarity score to accept a search result as a verified match
_MIN_MATCH_CONFIDENCE = 0.4


def _name_similarity(a: str, b: str) -> float:
    """Simple token-overlap similarity (0.0 – 1.0) between two names."""
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))


class PlaceResolver:
    """Resolves CustomPlace objects against the Places API.

    Priority:
      1. Exact / high-confidence name match in Places search results
      2. First returned result if the search query is very specific
      3. UNVERIFIED fallback — keep the user's name as-is
    """

    def __init__(self, places_service: Optional[PlacesService] = None):
        self.places = places_service or PlacesService()

    def resolve_custom_places(
        self,
        custom_places: List[CustomPlace],
        destination: str,
    ) -> List[CustomPlace]:
        """Attempt to verify each CustomPlace via the Places API.

        Mutates and returns the same list with status / coordinates updated.
        """
        for cp in custom_places:
            try:
                self._resolve_one(cp, destination)
            except Exception as e:
                logger.debug(
                    "Could not resolve '%s' in %s: %s — kept UNVERIFIED.",
                    cp.name, destination, e,
                )
                # Leave cp.status as UNVERIFIED
        return custom_places

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _resolve_one(self, cp: CustomPlace, destination: str) -> None:
        """Try to verify a single CustomPlace.  Modifies cp in-place."""
        # Build a targeted query: "<place name> <destination>"
        dest_part = cp.target_destination or destination
        query = f"{cp.name} {dest_part}"

        results: list = []
        try:
            results = self.places.search_text(query, max_results=5)
        except PlacesAPIError as e:
            if e.error_type in ("missing_api_key", "auth_error", "quota_exceeded"):
                logger.debug("Places API unavailable for resolver: %s", e)
                return  # graceful — keep UNVERIFIED
            raise

        if not results:
            logger.debug("No results for '%s' — kept UNVERIFIED.", cp.name)
            return

        # Pick best match by name similarity
        best = max(
            results,
            key=lambda r: _name_similarity(cp.name, r.get("name", "")),
        )
        score = _name_similarity(cp.name, best.get("name", ""))

        if score >= _MIN_MATCH_CONFIDENCE:
            cp.status = PlaceStatus.VERIFIED
            cp.place_id = best.get("id")
            cp.address = best.get("address") or cp.address
            cp.rating = best.get("rating") or cp.rating
            cp.latitude = best.get("latitude") or cp.latitude
            cp.longitude = best.get("longitude") or cp.longitude
            logger.debug(
                "Resolved '%s' → '%s' (score=%.2f, status=VERIFIED)",
                cp.name, best.get("name"), score,
            )
        else:
            logger.debug(
                "Best match for '%s' was '%s' (score=%.2f < %.2f) — kept UNVERIFIED.",
                cp.name, best.get("name"), score, _MIN_MATCH_CONFIDENCE,
            )
