"""Google Places API (New) service for TripMate foundation.

Provides:
- Text Search (places:searchText)
- Nearby Search (places:searchNearby)
- Place Details (places/{placeId})
- Geoapify adapter / fallback when Geoapify key is configured
"""

import os
import logging
from typing import Optional, Dict, Any, List
import requests

logger = logging.getLogger(__name__)

GOOGLE_PLACES_BASE_URL = "https://places.googleapis.com/v1"
GEOAPIFY_BASE_URL = "https://api.geoapify.com/v2/places"

DEFAULT_FIELD_MASK = (
    "places.id,"
    "places.displayName,"
    "places.formattedAddress,"
    "places.location,"
    "places.rating,"
    "places.userRatingCount,"
    "places.priceLevel,"
    "places.googleMapsUri"
)

DETAIL_FIELD_MASK = (
    "id,"
    "displayName,"
    "formattedAddress,"
    "location,"
    "rating,"
    "userRatingCount,"
    "priceLevel,"
    "googleMapsUri,"
    "currentOpeningHours,"
    "websiteUri,"
    "nationalPhoneNumber"
)


class PlacesAPIError(Exception):
    """Base exception for Places API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, error_type: str = "api_error"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type


def _load_env_if_needed():
    """Load .env file if present."""
    from pathlib import Path
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass

_load_env_if_needed()


def get_google_api_key() -> Optional[str]:
    """Retrieve Google Maps / Places API key from environment."""
    _load_env_if_needed()
    return os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("GOOGLE_PLACES_API_KEY")


def get_geoapify_api_key() -> Optional[str]:
    """Retrieve Geoapify API key from environment if configured."""
    _load_env_if_needed()
    return os.environ.get("GEOAPIFY_API_KEY")


class PlacesService:
    """Service to interact with Google Places API (New)."""

    def __init__(self, api_key: Optional[str] = None, geoapify_key: Optional[str] = None):
        # If explicitly passed (including empty string ""), respect it; otherwise load from env
        self.api_key = api_key if api_key is not None else get_google_api_key()
        self.geoapify_key = geoapify_key if geoapify_key is not None else get_geoapify_api_key()
        self.session = requests.Session()

    def _normalize_place(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Google Places (New) result into standard dictionary."""
        display_name = raw.get("displayName", {})
        name = display_name.get("text") if isinstance(display_name, dict) else str(display_name or "")

        loc = raw.get("location", {})
        lat = loc.get("latitude") if isinstance(loc, dict) else None
        lng = loc.get("longitude") if isinstance(loc, dict) else None

        return {
            "id": raw.get("id", ""),
            "name": name,
            "address": raw.get("formattedAddress", ""),
            "latitude": lat,
            "longitude": lng,
            "rating": raw.get("rating"),
            "user_rating_count": raw.get("userRatingCount", 0),
            "price_level": raw.get("priceLevel"),
            "google_maps_uri": raw.get("googleMapsUri", ""),
        }

    def _safe_request(self, method: str, url: str, headers: Dict[str, str], **kwargs) -> requests.Response:
        """Perform HTTP request with sanitized logging & error handling."""
        try:
            resp = self.session.request(method, url, headers=headers, timeout=12, **kwargs)
        except requests.exceptions.Timeout:
            raise PlacesAPIError("Request to Places API timed out. Please try again.", status_code=504, error_type="timeout")
        except requests.exceptions.ConnectionError:
            raise PlacesAPIError("Unable to connect to Places API. Check network connection.", status_code=503, error_type="network_error")
        except Exception as e:
            raise PlacesAPIError(f"Network error during Places API request: {str(e)}", error_type="network_error")

        if resp.status_code == 400:
            raise PlacesAPIError("Invalid Places API request. Please verify search parameters.", status_code=400, error_type="invalid_request")
        elif resp.status_code in (401, 403):
            raise PlacesAPIError("Invalid or unauthorized Google Maps API key. Check GOOGLE_MAPS_API_KEY.", status_code=resp.status_code, error_type="auth_error")
        elif resp.status_code == 429:
            raise PlacesAPIError("Google Maps API quota exceeded. Please try again later.", status_code=429, error_type="quota_exceeded")
        elif resp.status_code >= 500:
            raise PlacesAPIError("Google Maps service returned an internal error.", status_code=resp.status_code, error_type="server_error")

        return resp

    def search_text(
        self,
        query: str,
        location_bias: Optional[Dict[str, Any]] = None,
        max_results: int = 8,
        field_mask: str = DEFAULT_FIELD_MASK,
    ) -> List[Dict[str, Any]]:
        """
        Text Search using Google Places API (New):
        POST https://places.googleapis.com/v1/places:searchText
        """
        if not query or not query.strip():
            return []

        if not self.api_key:
            # Check if Geoapify key is present or return informative error
            if self.geoapify_key:
                return self.search_geoapify(query=query)
            raise PlacesAPIError(
                "Missing Google Maps API key. Please configure GOOGLE_MAPS_API_KEY.",
                status_code=401,
                error_type="missing_api_key"
            )

        url = f"{GOOGLE_PLACES_BASE_URL}/places:searchText"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": field_mask,
        }
        payload: Dict[str, Any] = {
            "textQuery": query.strip(),
            "pageSize": min(max(1, max_results), 20),
        }

        if location_bias:
            payload["locationBias"] = location_bias

        resp = self._safe_request("POST", url, headers=headers, json=payload)
        data = resp.json()
        raw_places = data.get("places", [])
        return [self._normalize_place(p) for p in raw_places]

    def search_nearby(
        self,
        latitude: float,
        longitude: float,
        radius_meters: float = 3000.0,
        included_types: Optional[List[str]] = None,
        max_results: int = 8,
        field_mask: str = DEFAULT_FIELD_MASK,
    ) -> List[Dict[str, Any]]:
        """
        Nearby Search using Google Places API (New):
        POST https://places.googleapis.com/v1/places:searchNearby
        """
        if not self.api_key:
            if self.geoapify_key:
                cat = (included_types[0] if included_types else "catering.restaurant")
                return self.search_geoapify_nearby(latitude, longitude, radius_meters, category=cat)
            raise PlacesAPIError(
                "Missing Google Maps API key. Please configure GOOGLE_MAPS_API_KEY.",
                status_code=401,
                error_type="missing_api_key"
            )

        url = f"{GOOGLE_PLACES_BASE_URL}/places:searchNearby"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": field_mask,
        }
        payload: Dict[str, Any] = {
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": latitude,
                        "longitude": longitude,
                    },
                    "radius": float(radius_meters),
                }
            },
            "maxResultCount": min(max(1, max_results), 20),
        }

        if included_types:
            payload["includedTypes"] = included_types

        resp = self._safe_request("POST", url, headers=headers, json=payload)
        data = resp.json()
        raw_places = data.get("places", [])
        return [self._normalize_place(p) for p in raw_places]

    def get_place_details(
        self,
        place_id: str,
        field_mask: str = DETAIL_FIELD_MASK,
    ) -> Dict[str, Any]:
        """
        Get Place Details using Google Places API (New):
        GET https://places.googleapis.com/v1/places/{placeId}
        """
        if not place_id:
            raise PlacesAPIError("place_id is required.", status_code=400, error_type="invalid_request")

        if not self.api_key:
            raise PlacesAPIError(
                "Missing Google Maps API key. Please configure GOOGLE_MAPS_API_KEY.",
                status_code=401,
                error_type="missing_api_key"
            )

        url = f"{GOOGLE_PLACES_BASE_URL}/places/{place_id}"
        headers = {
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": field_mask,
        }

        resp = self._safe_request("GET", url, headers=headers)
        data = resp.json()
        normalized = self._normalize_place(data)
        normalized.update({
            "website_uri": data.get("websiteUri"),
            "phone_number": data.get("nationalPhoneNumber"),
            "opening_hours": data.get("currentOpeningHours", {}).get("weekdayDescriptions", []),
        })
        return normalized

    # ── Geoapify Adapter Support ──
    def search_geoapify_nearby(
        self,
        latitude: float,
        longitude: float,
        radius_meters: float = 3000,
        category: str = "catering.restaurant"
    ) -> List[Dict[str, Any]]:
        """Search places using Geoapify API if configured."""
        if not self.geoapify_key:
            raise PlacesAPIError("Missing GEOAPIFY_API_KEY.", status_code=401, error_type="missing_api_key")

        url = f"{GEOAPIFY_BASE_URL}?categories={category}&filter=circle:{longitude},{latitude},{int(radius_meters)}&apiKey={self.geoapify_key}"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            features = resp.json().get("features", [])
            results = []
            for f in features:
                props = f.get("properties", {})
                results.append({
                    "id": props.get("place_id", ""),
                    "name": props.get("name", props.get("formatted", "Unnamed place")),
                    "address": props.get("formatted", ""),
                    "latitude": props.get("lat"),
                    "longitude": props.get("lon"),
                    "rating": props.get("rating"),
                    "user_rating_count": 0,
                    "price_level": None,
                    "google_maps_uri": f"https://www.google.com/maps/search/?api=1&query={props.get('lat')},{props.get('lon')}",
                })
            return results
        except Exception as e:
            logger.warning("Geoapify request failed: %s", e)
            raise PlacesAPIError(f"Geoapify request failed: {e}")

    def search_geoapify(self, query: str) -> List[Dict[str, Any]]:
        """Geoapify text search fallback."""
        if not self.geoapify_key:
            raise PlacesAPIError("Missing GEOAPIFY_API_KEY.", status_code=401, error_type="missing_api_key")
        url = f"https://api.geoapify.com/v1/geocode/search?text={query}&apiKey={self.geoapify_key}"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            features = resp.json().get("features", [])
            results = []
            for f in features:
                props = f.get("properties", {})
                results.append({
                    "id": props.get("place_id", ""),
                    "name": props.get("name", props.get("formatted", query)),
                    "address": props.get("formatted", ""),
                    "latitude": props.get("lat"),
                    "longitude": props.get("lon"),
                    "rating": 4.5,
                    "user_rating_count": 50,
                    "price_level": None,
                    "google_maps_uri": f"https://www.google.com/maps/search/?api=1&query={props.get('lat')},{props.get('lon')}",
                })
            return results
        except Exception as e:
            raise PlacesAPIError(f"Geoapify text search failed: {e}")
