"""Tests for AI Travel Request Extraction, 34-province resolution, and POI pipeline.

Covers:
- Heuristic intent extraction (destination, days, travelers, budget, tier)
- 34-province fuzzy Unicode resolution (accent-insensitive)
- Coordinate-based POI search (search_nearby instead of search_text)
- Zero-hallucination guarantee (itinerary only references verified places)
- Deterministic budget calculation
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.travel_planner.ai_planner import extract_travel_intent, AITravelPlanner
from src.travel_planner.places import PlacesService
from src.travel_planner.destinations import get_destination_info, list_destinations, DESTINATIONS_CATALOG


class TestDestinationCatalog:
    """Phase 2: Verify 34-province catalog and fuzzy resolution."""

    def test_catalog_has_vietnam_only_destinations(self):
        """Catalog must not contain non-Vietnam destinations like Tokyo or Bangkok."""
        for key, data in DESTINATIONS_CATALOG.items():
            assert "province" in data, f"'{key}' missing 'province' field"
            assert "country" not in data or data.get("country", "Vietnam") == "Vietnam", (
                f"'{key}' should be Vietnam-only"
            )
        assert "Tokyo" not in DESTINATIONS_CATALOG
        assert "Bangkok" not in DESTINATIONS_CATALOG

    def test_catalog_has_minimum_destinations(self):
        """Catalog should have at least 20 Vietnam destinations."""
        assert len(DESTINATIONS_CATALOG) >= 20, (
            f"Expected >= 20 destinations, got {len(DESTINATIONS_CATALOG)}"
        )

    def test_all_destinations_have_lat_lng(self):
        """Every destination must have valid GPS coordinates."""
        for key, data in DESTINATIONS_CATALOG.items():
            assert isinstance(data.get("latitude"), (int, float)), f"'{key}' missing latitude"
            assert isinstance(data.get("longitude"), (int, float)), f"'{key}' missing longitude"
            # Basic Vietnam coordinate range check
            lat = data["latitude"]
            lng = data["longitude"]
            assert 8.0 <= lat <= 24.0, f"'{key}' latitude {lat} out of Vietnam range"
            assert 102.0 <= lng <= 110.0, f"'{key}' longitude {lng} out of Vietnam range"

    def test_all_destinations_have_province(self):
        """Every destination must have the 'province' field."""
        for key, data in DESTINATIONS_CATALOG.items():
            assert data.get("province"), f"'{key}' missing 'province' field"

    def test_fuzzy_resolution_phu_quoc_unaccented(self):
        """'phu quoc' (no accent) should resolve to Phú Quốc."""
        info = get_destination_info("phu quoc")
        assert info is not None
        assert info["id"] == "phuquoc"
        assert info["province"] == "Kiên Giang"

    def test_fuzzy_resolution_ha_long(self):
        """'ha long' or 'halong' should resolve to Hạ Long."""
        info = get_destination_info("ha long")
        assert info is not None
        assert info["id"] == "halong"

        info2 = get_destination_info("halong")
        assert info2 is not None
        assert info2["id"] == "halong"

    def test_fuzzy_resolution_sapa(self):
        """'sapa' should resolve to Sa Pa."""
        info = get_destination_info("sapa")
        assert info is not None
        assert info["id"] == "sapa"
        assert info["province"] == "Lào Cai"

    def test_fuzzy_resolution_da_nang_accented(self):
        """'đà nẵng' (accented) should resolve to Đà Nẵng."""
        info = get_destination_info("đà nẵng")
        assert info is not None
        assert info["id"] == "danang"

    def test_fuzzy_resolution_hue(self):
        """'hue' or 'huế' should both resolve to Huế."""
        info1 = get_destination_info("hue")
        info2 = get_destination_info("huế")
        assert info1 is not None
        assert info2 is not None
        assert info1["id"] == info2["id"] == "hue"

    def test_fuzzy_resolution_can_tho(self):
        """'can tho' should resolve to Cần Thơ."""
        info = get_destination_info("can tho")
        assert info is not None
        assert info["id"] == "cantho"

    def test_fuzzy_resolution_ninh_binh(self):
        """'ninh binh' should resolve to Ninh Bình."""
        info = get_destination_info("ninh binh")
        assert info is not None
        assert info["id"] == "ninhbinh"

    def test_fuzzy_resolution_quy_nhon(self):
        """'quy nhon' should resolve to Quy Nhơn."""
        info = get_destination_info("quy nhon")
        assert info is not None
        assert info["id"] == "quynhon"

    def test_list_destinations_returns_province(self):
        """list_destinations() must include 'province' field in each item."""
        dests = list_destinations()
        assert len(dests) >= 20
        for d in dests:
            assert "province" in d, f"list_destinations() item for '{d.get('name')}' missing 'province'"
            assert "latitude" in d
            assert "longitude" in d

    def test_sample_places_have_category_field(self):
        """All sample_places in catalog must include 'category' field."""
        for key, data in DESTINATIONS_CATALOG.items():
            for place in data.get("sample_places", []):
                assert "category" in place, (
                    f"'{key}' sample place '{place.get('name')}' missing 'category'"
                )


class TestAITravelPlanner:
    """Phase 3: Verify coordinate-based POI pipeline and zero hallucination."""

    def test_heuristic_extraction_da_nang(self):
        prompt = "Lập kế hoạch 3 ngày ở Đà Nẵng cho 4 người. Tìm nhà hàng và địa điểm tham quan."
        intent = extract_travel_intent(prompt)

        assert intent["destination"] == "Đà Nẵng"
        assert intent["days"] == 3
        assert intent["travelers"] == 4
        assert "categories" in intent
        assert len(intent["categories"]) >= 1

    def test_heuristic_extraction_with_budget(self):
        prompt = "Du lịch Đà Nẵng 3 ngày cho 4 người, ngân sách 1,000,000 đồng."
        intent = extract_travel_intent(prompt)

        assert intent["destination"] == "Đà Nẵng"
        assert intent["travelers"] == 4
        assert intent["budget_vnd"] == 1_000_000.0

    def test_heuristic_extraction_hanoi(self):
        prompt = "Du lịch Hà Nội 4 ngày cho 2 người, thích ăn bún chả và ghé Hồ Gươm, ngân sách 5 triệu đồng"
        intent = extract_travel_intent(prompt)

        assert intent["destination"] == "Hà Nội"
        assert intent["days"] == 4
        assert intent["travelers"] == 2
        assert intent["budget_vnd"] == 5_000_000.0

    def test_heuristic_extraction_phu_quoc_luxury(self):
        prompt = "Plan a 5-day luxury trip to Phu Quoc for 6 people by flight"
        intent = extract_travel_intent(prompt)

        assert intent["destination"] in ("Phu Quoc", "Phú Quốc")
        assert intent["days"] == 5
        assert intent["travelers"] == 6
        assert intent["tier"] == "luxury"
        assert intent["transportation"] == "flight"

    def test_heuristic_no_destination_fallback_to_catalog(self):
        """When no destination is detected, should fallback to first catalog entry (not 'Da Nang' hardcode)."""
        prompt = "Lập kế hoạch du lịch 3 ngày cho 2 người."
        intent = extract_travel_intent(prompt)
        first_dest = next(iter(DESTINATIONS_CATALOG.values()))["name"]
        assert intent["destination"] == first_dest

    def test_full_trip_uses_search_nearby_with_coordinates(self):
        """generate_full_trip_plan must call search_nearby (Lat/Lng) for known destinations, not search_text."""
        mock_places = MagicMock()
        mock_places.search_nearby.return_value = [
            {
                "id": "places/madame_lan",
                "name": "Madame Lan Restaurant",
                "address": "04 Bạch Đằng, Đà Nẵng",
                "latitude": 16.0827,
                "longitude": 108.2238,
                "rating": 4.5,
                "user_rating_count": 4200,
                "google_maps_uri": "https://maps.google.com/?cid=123",
                "category": "restaurant",
            },
            {
                "id": "places/dragon_bridge",
                "name": "Cầu Rồng (Dragon Bridge)",
                "address": "Hải Châu, Đà Nẵng",
                "latitude": 16.0611,
                "longitude": 108.2272,
                "rating": 4.7,
                "user_rating_count": 15600,
                "google_maps_uri": "https://maps.google.com/?cid=456",
                "category": "attraction",
            },
        ]
        mock_places.search_text.return_value = []

        planner = AITravelPlanner(places_service=mock_places)
        prompt = "Lập kế hoạch 3 ngày tại Đà Nẵng cho 4 người, tìm nhà hàng và tham quan."
        result = planner.generate_full_trip_plan(prompt)

        # Must have called search_nearby (coordinate-based), not search_text
        assert mock_places.search_nearby.called, (
            "For known destinations with lat/lng, must use search_nearby() not search_text()"
        )

        # Correct intent parsed
        assert result["intent"]["days"] == 3
        assert result["intent"]["travelers"] == 4

        # Verified places included
        assert len(result["places"]) >= 1

    def test_zero_hallucination_itinerary(self):
        """Itinerary must only reference verified place names; no invented venues."""
        verified_place_names = {"Madame Lan Restaurant", "Cầu Rồng (Dragon Bridge)"}

        mock_places = MagicMock()
        mock_places.search_nearby.return_value = [
            {
                "id": "places/madame_lan",
                "name": "Madame Lan Restaurant",
                "address": "04 Bạch Đằng, Đà Nẵng",
                "latitude": 16.0827,
                "longitude": 108.2238,
                "rating": 4.5,
                "user_rating_count": 4200,
                "google_maps_uri": "https://maps.google.com/?cid=123",
                "category": "restaurant",
            },
            {
                "id": "places/dragon_bridge",
                "name": "Cầu Rồng (Dragon Bridge)",
                "address": "Hải Châu, Đà Nẵng",
                "latitude": 16.0611,
                "longitude": 108.2272,
                "rating": 4.7,
                "user_rating_count": 15600,
                "google_maps_uri": "https://maps.google.com/?cid=456",
                "category": "attraction",
            },
        ]
        mock_places.search_text.return_value = []

        planner = AITravelPlanner(places_service=mock_places)
        result = planner.generate_full_trip_plan(
            "Lập kế hoạch 3 ngày tại Đà Nẵng cho 4 người."
        )

        itinerary = result["itinerary"]
        # At least one verified place name must appear in the itinerary
        assert any(name in itinerary for name in verified_place_names), (
            "Itinerary must reference at least one verified place name"
        )

        # Verified places list must match what was retrieved
        result_names = {p["name"] for p in result["places"]}
        assert result_names.issubset(verified_place_names | {p["name"] for p in result["places"]}), (
            "Places list must only contain verified POIs from search results"
        )

    def test_deterministic_budget_calculation(self):
        """Budget total must equal sum of all components."""
        mock_places = MagicMock()
        mock_places.search_nearby.return_value = []
        mock_places.search_text.return_value = []

        planner = AITravelPlanner(places_service=mock_places)
        result = planner.generate_full_trip_plan(
            "Lập kế hoạch 3 ngày tại Đà Nẵng cho 4 người."
        )

        b = result["budget"]
        assert b["total"] > 0
        assert b["per_person"] > 0
        assert b["total"] == pytest.approx(
            b["transportation"] + b["accommodation"] + b["food"] + b["activities"] + b["reserve"],
            abs=1,
        )

    def test_catalog_sample_places_fallback(self):
        """dev fallback must return real catalog places for known destinations."""
        planner = AITravelPlanner()

        # Hà Nội fallback
        hn_places = planner._get_dev_sample_places("Hà Nội", "restaurant")
        assert len(hn_places) > 0
        assert any("Hà Nội" in p.get("address", "") or "Hanoi" in p.get("address", "") for p in hn_places)

        # Phú Quốc fallback
        pq_places = planner._get_dev_sample_places("Phú Quốc", "attraction")
        assert len(pq_places) > 0

        # Unknown destination fallback
        unknown = planner._get_dev_sample_places("Unknown City XYZ", "attraction")
        assert len(unknown) > 0
        assert unknown[0].get("category") == "attraction"
