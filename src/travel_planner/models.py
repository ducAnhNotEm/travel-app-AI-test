"""Data models for TripMate Hybrid AI Planning Pipeline.

Defines all shared dataclasses and enums used across the pipeline:
  - ItemType / PlaceStatus (enums)
  - DestinationItem, CustomPlace, NormalizedPlace
  - ItineraryItem, TravelIntent
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class ItemType(str, Enum):
    """Type of a single item inside a generated itinerary."""
    PLACE = "PLACE"        # Specific POI: restaurant, attraction, cafe
    ACTIVITY = "ACTIVITY"  # Experience activity: sunset stroll, beach swim
    TRANSIT = "TRANSIT"    # Inter-city / long-distance travel segment


class PlaceStatus(str, Enum):
    """Verification status for a user-supplied custom place."""
    VERIFIED = "verified"      # Confirmed via Google Places (has place_id, lat/lng, rating)
    UNVERIFIED = "unverified"  # User-suggested; not found / not searched on map


@dataclass
class DestinationItem:
    """One stop in a multi-destination trip."""
    name: str
    days: int
    order: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass
class CustomPlace:
    """A specific place explicitly mentioned by the user in their prompt."""
    name: str
    category: str          # restaurant | attraction | cafe | hotel
    source: str = "user_prompt"
    status: PlaceStatus = PlaceStatus.UNVERIFIED
    place_id: Optional[str] = None
    address: Optional[str] = None
    rating: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    target_destination: Optional[str] = None
    target_day: Optional[int] = None
    # breakfast | lunch | dinner | morning | afternoon | evening
    target_slot: Optional[str] = None


@dataclass
class NormalizedPlace:
    """Unified POI record from Google Places / Geoapify / catalog."""
    id: str
    name: str
    category: str
    address: str
    latitude: float
    longitude: float
    rating: float
    user_rating_count: int
    google_maps_uri: Optional[str] = None
    # google_places | pinned | custom_verified | custom_unverified | catalog
    source: str = "google_places"


@dataclass
class ItineraryItem:
    """A single time-block in the deterministic itinerary."""
    item_type: ItemType
    title: str
    # e.g. "07:30 - 08:30", "09:00 - 11:30"
    time_slot: str
    day: int
    destination_name: str
    description: str = ""
    # Full NormalizedPlace info when item_type == PLACE
    place_details: Optional[Dict[str, Any]] = None
    # verified / unverified when item_type == PLACE
    status: Optional[PlaceStatus] = None
    # From/To/Distance/Time when item_type == TRANSIT
    transit_info: Optional[Dict[str, Any]] = None


@dataclass
class TravelIntent:
    """Structured representation of a user's travel request."""
    destinations: List[DestinationItem]
    total_days: int
    travelers: int
    transportation: str
    tier: str
    custom_places: List[CustomPlace] = field(default_factory=list)
    preferences: List[str] = field(default_factory=list)
    budget_vnd: Optional[float] = None
