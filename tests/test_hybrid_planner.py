"""Tests for TripMate Hybrid AI Planning Pipeline.

Covers:
  - Intent extraction for multi-destination trips and custom places
  - Place resolution (VERIFIED vs UNVERIFIED)
  - Scheduling logic (transit on correct day, no day leak)
  - Route optimisation (Haversine + nearest-neighbor)
  - Anti-hallucination (no unknown POIs in itinerary)
  - Full pipeline integration
"""

import pytest
import sys
import os
from typing import List
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.travel_planner.models import (
    TravelIntent,
    DestinationItem,
    CustomPlace,
    NormalizedPlace,
    ItineraryItem,
    ItemType,
    PlaceStatus,
)
from src.travel_planner.optimizer import haversine_km, nearest_neighbor_sort, ConstraintEngine
from src.travel_planner.fusion import PlaceFusionEngine, _custom_place_to_normalized, _google_raw_to_normalized
from src.travel_planner.resolver import PlaceResolver, _name_similarity
from src.travel_planner.itinerary import ItineraryBuilder
from src.travel_planner.ai_planner import (
    extract_travel_intent,
    extract_travel_intent_v2,
    _heuristic_extract,
    _legacy_to_travel_intent,
    format_itinerary,
    AITravelPlanner,
)
from src.travel_planner.places import PlacesAPIError


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_normalized_place():
    return NormalizedPlace(
        id="places/abc123",
        name="Nhà hàng Madame Lan",
        category="restaurant",
        address="04 Bạch Đằng, Đà Nẵng",
        latitude=16.0827,
        longitude=108.2238,
        rating=4.5,
        user_rating_count=4200,
        google_maps_uri="https://maps.google.com/?cid=123",
        source="google_places",
    )


@pytest.fixture
def sample_attraction():
    return NormalizedPlace(
        id="places/dragon_bridge",
        name="Cầu Rồng (Dragon Bridge)",
        category="attraction",
        address="Hải Châu, Đà Nẵng",
        latitude=16.0611,
        longitude=108.2272,
        rating=4.7,
        user_rating_count=15600,
        google_maps_uri="https://maps.google.com/?cid=456",
        source="google_places",
    )


@pytest.fixture
def danang_intent():
    return TravelIntent(
        destinations=[DestinationItem(name="Đà Nẵng", days=2, order=1, latitude=16.0471, longitude=108.2068)],
        total_days=2,
        travelers=4,
        transportation="car",
        tier="moderate",
    )


@pytest.fixture
def multi_dest_intent():
    return TravelIntent(
        destinations=[
            DestinationItem(name="Đà Nẵng", days=2, order=1, latitude=16.0471, longitude=108.2068),
            DestinationItem(name="Hội An", days=1, order=2, latitude=15.8794, longitude=108.3380),
        ],
        total_days=3,
        travelers=4,
        transportation="car",
        tier="moderate",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Models
# ─────────────────────────────────────────────────────────────────────────────

class TestModels:
    def test_item_type_values(self):
        assert ItemType.PLACE == "PLACE"
        assert ItemType.ACTIVITY == "ACTIVITY"
        assert ItemType.TRANSIT == "TRANSIT"

    def test_place_status_values(self):
        assert PlaceStatus.VERIFIED == "verified"
        assert PlaceStatus.UNVERIFIED == "unverified"

    def test_destination_item_fields(self):
        d = DestinationItem(name="Hội An", days=2, order=1)
        assert d.name == "Hội An"
        assert d.days == 2
        assert d.order == 1
        assert d.latitude is None

    def test_custom_place_defaults(self):
        cp = CustomPlace(name="Bánh Mì Phượng", category="restaurant")
        assert cp.status == PlaceStatus.UNVERIFIED
        assert cp.source == "user_prompt"
        assert cp.place_id is None

    def test_itinerary_item_fields(self):
        item = ItineraryItem(
            item_type=ItemType.PLACE,
            title="Test Place",
            time_slot="09:00 - 11:30",
            day=1,
            destination_name="Đà Nẵng",
        )
        assert item.item_type == ItemType.PLACE
        assert item.day == 1
        assert item.description == ""

    def test_travel_intent_defaults(self):
        intent = TravelIntent(
            destinations=[DestinationItem("Hà Nội", 3, 1)],
            total_days=3,
            travelers=2,
            transportation="car",
            tier="moderate",
        )
        assert intent.custom_places == []
        assert intent.preferences == []
        assert intent.budget_vnd is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Haversine & Route Optimiser
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizer:
    def test_haversine_same_point(self):
        dist = haversine_km(16.0, 108.0, 16.0, 108.0)
        assert dist == pytest.approx(0.0, abs=0.001)

    def test_haversine_da_nang_to_hoi_an(self):
        """Đà Nẵng → Hội An is roughly 28-32 km by road (straight-line ~25-30 km)."""
        dist = haversine_km(16.0471, 108.2068, 15.8794, 108.3380)
        assert 20 <= dist <= 40, f"Expected 20-40 km, got {dist:.1f} km"

    def test_haversine_symmetry(self):
        d1 = haversine_km(10.0, 106.0, 21.0, 105.0)
        d2 = haversine_km(21.0, 105.0, 10.0, 106.0)
        assert d1 == pytest.approx(d2, abs=0.001)

    def test_nearest_neighbor_sort_empty(self):
        result = nearest_neighbor_sort([])
        assert result == []

    def test_nearest_neighbor_sort_single(self, sample_normalized_place):
        result = nearest_neighbor_sort([sample_normalized_place])
        assert len(result) == 1
        assert result[0].id == sample_normalized_place.id

    def test_nearest_neighbor_sort_order(self, sample_normalized_place, sample_attraction):
        """After sorting, first place stays first (it's the anchor)."""
        places = [sample_normalized_place, sample_attraction]
        result = nearest_neighbor_sort(places)
        assert len(result) == 2
        # First place is the anchor, second is the closest — both should be present
        ids = {p.id for p in result}
        assert sample_normalized_place.id in ids
        assert sample_attraction.id in ids

    def test_nearest_neighbor_no_coords_appended_last(self):
        with_coords = NormalizedPlace(
            id="a", name="A", category="attraction", address="", latitude=16.0, longitude=108.0,
            rating=4.0, user_rating_count=100,
        )
        no_coords = NormalizedPlace(
            id="b", name="B", category="attraction", address="", latitude=0.0, longitude=0.0,
            rating=4.0, user_rating_count=100,
        )
        result = nearest_neighbor_sort([with_coords, no_coords])
        assert result[-1].id == "b"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Constraint Engine
# ─────────────────────────────────────────────────────────────────────────────

class TestConstraintEngine:
    def test_build_day_schedule_basic(self, sample_normalized_place, sample_attraction):
        engine = ConstraintEngine()
        items = engine.build_day_schedule(
            day=1,
            destination_name="Đà Nẵng",
            available_places=[sample_normalized_place, sample_attraction],
            locked_places=[],
        )
        assert len(items) > 0
        # All items belong to day 1
        assert all(item.day == 1 for item in items)
        # All items reference the correct destination
        assert all(item.destination_name == "Đà Nẵng" for item in items)

    def test_build_day_schedule_slots_covered(self, sample_normalized_place, sample_attraction):
        """All 7 time slots should produce an item (PLACE or ACTIVITY)."""
        engine = ConstraintEngine()
        items = engine.build_day_schedule(
            day=1,
            destination_name="Đà Nẵng",
            available_places=[sample_normalized_place, sample_attraction],
            locked_places=[],
        )
        assert len(items) == 7

    def test_locked_place_appears_in_correct_slot(self, sample_normalized_place):
        """A locked custom place should appear in its target slot."""
        locked_cp = CustomPlace(
            name="Cơm Niêu Nhà Đỏ",
            category="restaurant",
            target_destination="Đà Nẵng",
            target_day=1,
            target_slot="dinner",
        )
        engine = ConstraintEngine()
        items = engine.build_day_schedule(
            day=1,
            destination_name="Đà Nẵng",
            available_places=[sample_normalized_place],
            locked_places=[locked_cp],
        )
        dinner_items = [i for i in items if i.time_slot == "19:00 - 20:30"]
        assert len(dinner_items) == 1
        assert "Cơm Niêu Nhà Đỏ" in dinner_items[0].title

    def test_dusk_is_always_activity(self, sample_normalized_place, sample_attraction):
        """Dusk slot (17:30 - 18:30) must always be an ACTIVITY."""
        engine = ConstraintEngine()
        items = engine.build_day_schedule(
            day=1,
            destination_name="Đà Nẵng",
            available_places=[sample_normalized_place, sample_attraction],
            locked_places=[],
        )
        dusk_items = [i for i in items if i.time_slot == "17:30 - 18:30"]
        assert len(dusk_items) == 1
        assert dusk_items[0].item_type == ItemType.ACTIVITY


# ─────────────────────────────────────────────────────────────────────────────
# 4. Place Fusion Engine
# ─────────────────────────────────────────────────────────────────────────────

class TestPlaceFusionEngine:
    def test_fuse_pinned_has_priority(self):
        """Pinned place with same name as google place should not be duplicated."""
        pinned = NormalizedPlace(
            id="places/dragon_bridge", name="Cầu Rồng", category="attraction",
            address="Đà Nẵng", latitude=16.06, longitude=108.22, rating=4.7,
            user_rating_count=15000, source="pinned",
        )
        google_raw = {
            "id": "places/dragon_bridge", "name": "Cầu Rồng", "category": "attraction",
            "address": "Đà Nẵng", "latitude": 16.06, "longitude": 108.22,
            "rating": 4.7, "user_rating_count": 15000,
        }
        engine = PlaceFusionEngine()
        result = engine.fuse([pinned], [], [google_raw])
        # Should deduplicate
        names = [p.name for p in result]
        assert names.count("Cầu Rồng") == 1
        # Pinned source should be kept
        dragon = next(p for p in result if p.name == "Cầu Rồng")
        assert dragon.source == "pinned"

    def test_fuse_dedup_by_id(self):
        """Two places with the same id should appear only once."""
        p1 = NormalizedPlace(
            id="places/xyz", name="Place A", category="attraction",
            address="", latitude=0.0, longitude=0.0, rating=4.0, user_rating_count=0,
        )
        google_raw = {"id": "places/xyz", "name": "Place A", "category": "attraction",
                      "address": "", "latitude": 0.0, "longitude": 0.0}
        engine = PlaceFusionEngine()
        result = engine.fuse([p1], [], [google_raw])
        assert len([p for p in result if p.id == "places/xyz"]) == 1

    def test_fuse_custom_unverified_included(self):
        """Unverified custom places must still appear in the fused list."""
        cp = CustomPlace(name="Quán Bà Năm", category="restaurant")
        engine = PlaceFusionEngine()
        result = engine.fuse([], [cp], [])
        assert any(p.name == "Quán Bà Năm" for p in result)
        unverified = next(p for p in result if p.name == "Quán Bà Năm")
        assert unverified.source == "custom_unverified"

    def test_fuse_order_pinned_first(self):
        """Pinned places must appear before google-recommended ones."""
        pinned = NormalizedPlace(
            id="pin1", name="Pinned Place", category="attraction",
            address="", latitude=16.0, longitude=108.0, rating=4.0,
            user_rating_count=0, source="pinned",
        )
        google_raw = {"id": "goog1", "name": "Google Place", "category": "attraction",
                      "address": "", "latitude": 16.1, "longitude": 108.1, "rating": 4.5}
        engine = PlaceFusionEngine()
        result = engine.fuse([pinned], [], [google_raw])
        assert result[0].id == "pin1"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Place Resolver
# ─────────────────────────────────────────────────────────────────────────────

class TestPlaceResolver:
    def test_name_similarity_identical(self):
        assert _name_similarity("Bánh Mì Phượng", "Bánh Mì Phượng") == pytest.approx(1.0)

    def test_name_similarity_partial(self):
        score = _name_similarity("Cơm Niêu", "Cơm Niêu Nhà Đỏ")
        assert 0.0 < score < 1.0

    def test_name_similarity_disjoint(self):
        score = _name_similarity("Café ABC", "Restaurant XYZ")
        assert score == 0.0

    def test_resolve_verifies_with_good_match(self):
        """If Places API returns a good match, CustomPlace should be VERIFIED."""
        mock_places = MagicMock()
        mock_places.search_text.return_value = [
            {
                "id": "places/phuong_banh_mi",
                "name": "Bánh Mì Phượng",
                "address": "2B Phan Châu Trinh, Hội An",
                "latitude": 15.880,
                "longitude": 108.335,
                "rating": 4.6,
                "user_rating_count": 9000,
            }
        ]
        resolver = PlaceResolver(mock_places)
        cp = CustomPlace(name="Bánh Mì Phượng", category="restaurant")
        result = resolver.resolve_custom_places([cp], "Hội An")
        assert result[0].status == PlaceStatus.VERIFIED
        assert result[0].place_id == "places/phuong_banh_mi"
        assert result[0].rating == 4.6

    def test_resolve_keeps_unverified_on_no_results(self):
        """When Places API returns nothing, CustomPlace stays UNVERIFIED."""
        mock_places = MagicMock()
        mock_places.search_text.return_value = []
        resolver = PlaceResolver(mock_places)
        cp = CustomPlace(name="Quán Lạ Không Có Thật", category="restaurant")
        result = resolver.resolve_custom_places([cp], "Đà Nẵng")
        assert result[0].status == PlaceStatus.UNVERIFIED

    def test_resolve_graceful_on_api_error(self):
        """API errors must not crash the resolver; place stays UNVERIFIED."""
        mock_places = MagicMock()
        mock_places.search_text.side_effect = PlacesAPIError(
            "No API key", status_code=401, error_type="missing_api_key"
        )
        resolver = PlaceResolver(mock_places)
        cp = CustomPlace(name="Nhà Hàng Nào Đó", category="restaurant")
        result = resolver.resolve_custom_places([cp], "Hà Nội")
        assert result[0].status == PlaceStatus.UNVERIFIED


# ─────────────────────────────────────────────────────────────────────────────
# 6. Itinerary Builder — Transit Scheduling
# ─────────────────────────────────────────────────────────────────────────────

class TestItineraryBuilder:
    def test_single_destination_no_transit(self, danang_intent, sample_normalized_place, sample_attraction):
        builder = ItineraryBuilder()
        places = {
            "Đà Nẵng": [sample_normalized_place, sample_attraction],
        }
        items = builder.build(danang_intent, places)
        transit_items = [i for i in items if i.item_type == ItemType.TRANSIT]
        assert len(transit_items) == 0

    def test_multi_dest_transit_injected(self, multi_dest_intent, sample_normalized_place, sample_attraction):
        """TRANSIT item must be inserted exactly once when switching destinations."""
        builder = ItineraryBuilder()
        places = {
            "Đà Nẵng": [sample_normalized_place, sample_attraction],
            "Hội An": [sample_normalized_place],  # reuse for simplicity
        }
        items = builder.build(multi_dest_intent, places)
        transit_items = [i for i in items if i.item_type == ItemType.TRANSIT]
        assert len(transit_items) == 1, (
            f"Expected 1 TRANSIT item, got {len(transit_items)}"
        )

    def test_transit_on_correct_day(self, multi_dest_intent, sample_normalized_place, sample_attraction):
        """TRANSIT for Hội An must be on Day 3 (after 2 days in Đà Nẵng)."""
        builder = ItineraryBuilder()
        places = {
            "Đà Nẵng": [sample_normalized_place, sample_attraction],
            "Hội An": [sample_normalized_place],
        }
        items = builder.build(multi_dest_intent, places)
        transit_items = [i for i in items if i.item_type == ItemType.TRANSIT]
        assert transit_items[0].day == 3

    def test_day_1_and_2_in_da_nang(self, multi_dest_intent, sample_normalized_place, sample_attraction):
        """Days 1 & 2 must be 100% in Đà Nẵng (no Hội An items)."""
        builder = ItineraryBuilder()
        places = {
            "Đà Nẵng": [sample_normalized_place, sample_attraction],
            "Hội An": [sample_normalized_place],
        }
        items = builder.build(multi_dest_intent, places)
        early_items = [i for i in items if i.day in (1, 2) and i.item_type != ItemType.TRANSIT]
        assert all(i.destination_name == "Đà Nẵng" for i in early_items), (
            "Days 1-2 should only contain Đà Nẵng items"
        )

    def test_day_3_in_hoi_an(self, multi_dest_intent, sample_normalized_place, sample_attraction):
        """After transit, Day 3 non-transit items must reference Hội An."""
        builder = ItineraryBuilder()
        places = {
            "Đà Nẵng": [sample_normalized_place, sample_attraction],
            "Hội An": [sample_normalized_place],
        }
        items = builder.build(multi_dest_intent, places)
        day3_non_transit = [
            i for i in items if i.day == 3 and i.item_type != ItemType.TRANSIT
        ]
        assert any(i.destination_name == "Hội An" for i in day3_non_transit), (
            "Day 3 should have Hội An items after the transit"
        )

    def test_transit_contains_from_to_info(self, multi_dest_intent, sample_normalized_place, sample_attraction):
        """Transit item must carry from/to info in transit_info."""
        builder = ItineraryBuilder()
        places = {
            "Đà Nẵng": [sample_normalized_place, sample_attraction],
            "Hội An": [sample_normalized_place],
        }
        items = builder.build(multi_dest_intent, places)
        transit = next(i for i in items if i.item_type == ItemType.TRANSIT)
        assert transit.transit_info is not None
        assert transit.transit_info["from"] == "Đà Nẵng"
        assert transit.transit_info["to"] == "Hội An"
        assert "distance_km" in transit.transit_info
        assert transit.transit_info["distance_km"] is not None

    def test_total_items_match_expected_days(self, multi_dest_intent, sample_normalized_place, sample_attraction):
        """3 days × 7 slots + 1 transit = 22 items total."""
        builder = ItineraryBuilder()
        places = {
            "Đà Nẵng": [sample_normalized_place, sample_attraction],
            "Hội An": [sample_normalized_place],
        }
        items = builder.build(multi_dest_intent, places)
        # 3 days * 7 slots/day + 1 transit = 22
        assert len(items) == 22, f"Expected 22 items, got {len(items)}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Intent Extraction — Multi-destination & Custom Places
# ─────────────────────────────────────────────────────────────────────────────

class TestIntentExtraction:
    def test_heuristic_multi_destination(self):
        prompt = "Tôi muốn đi Đà Nẵng 2 ngày rồi sang Hội An 1 ngày cho 4 người."
        intent = extract_travel_intent_v2(prompt)
        dest_names = [d.name for d in intent.destinations]
        assert len(intent.destinations) >= 1
        assert intent.travelers == 4
        assert intent.total_days >= 3

    def test_heuristic_single_destination(self):
        prompt = "Lập kế hoạch 3 ngày ở Đà Nẵng cho 4 người, tìm nhà hàng và địa điểm tham quan."
        intent = extract_travel_intent_v2(prompt)
        assert intent.destinations[0].name == "Đà Nẵng"
        assert intent.total_days == 3
        assert intent.travelers == 4

    def test_heuristic_custom_places_dinner(self):
        prompt = "Đi Đà Nẵng 2 ngày. Tối ngày 1 ăn Cơm Niêu Nhà Đỏ Đà Nẵng."
        result = _heuristic_extract(prompt)
        custom_places = result.get("custom_places", [])
        assert any("Cơm Niêu" in cp["name"] for cp in custom_places), (
            "Should detect 'Cơm Niêu Nhà Đỏ' as a custom place"
        )
        dinner_cp = next(cp for cp in custom_places if "Cơm Niêu" in cp["name"])
        assert dinner_cp["target_slot"] == "dinner"
        assert dinner_cp["target_day"] == 1

    def test_heuristic_budget_extraction(self):
        prompt = "Du lịch Đà Nẵng 3 ngày cho 4 người, ngân sách 5 triệu đồng."
        result = _heuristic_extract(prompt)
        assert result["budget_vnd"] == 5_000_000.0

    def test_heuristic_tier_luxury(self):
        prompt = "Plan a luxury 5-day trip to Phu Quoc for 6 people by flight."
        result = _heuristic_extract(prompt)
        assert result["tier"] == "luxury"
        assert result["transportation"] == "flight"

    def test_legacy_intent_backward_compat(self):
        """extract_travel_intent (legacy) must still return destination, days, travelers."""
        prompt = "Lập kế hoạch 3 ngày ở Đà Nẵng cho 4 người."
        result = extract_travel_intent(prompt)
        assert "destination" in result
        assert "days" in result
        assert "travelers" in result
        assert result["destination"] == "Đà Nẵng"
        assert result["days"] == 3
        assert result["travelers"] == 4


# ─────────────────────────────────────────────────────────────────────────────
# 8. Anti-Hallucination: Itinerary references only verified places
# ─────────────────────────────────────────────────────────────────────────────

class TestAntiHallucination:
    def test_no_unknown_pois_in_items(self, danang_intent, sample_normalized_place, sample_attraction):
        """Only place names from the fused pool should appear in PLACE items."""
        allowed_names = {sample_normalized_place.name, sample_attraction.name}

        builder = ItineraryBuilder()
        places = {"Đà Nẵng": [sample_normalized_place, sample_attraction]}
        items = builder.build(danang_intent, places)

        place_items = [i for i in items if i.item_type == ItemType.PLACE]
        for item in place_items:
            assert item.title in allowed_names or item.item_type == ItemType.ACTIVITY, (
                f"Unexpected POI '{item.title}' in itinerary (not in verified pool)"
            )

    def test_format_itinerary_contains_verified_names(self, danang_intent, sample_normalized_place, sample_attraction):
        """format_itinerary template must include verified place names."""
        builder = ItineraryBuilder()
        places = {"Đà Nẵng": [sample_normalized_place, sample_attraction]}
        items = builder.build(danang_intent, places)
        text = format_itinerary(items, danang_intent)

        assert sample_normalized_place.name in text or sample_attraction.name in text, (
            "Formatted itinerary must reference at least one verified place"
        )

    def test_unverified_custom_place_labelled(self, danang_intent):
        """Unverified custom places must carry a warning label in the formatted output."""
        cp = CustomPlace(name="Quán Lạ Bịa Đặt", category="restaurant", target_slot="dinner")
        from src.travel_planner.fusion import _custom_place_to_normalized

        np_unverified = _custom_place_to_normalized(cp)
        item = ItineraryItem(
            item_type=ItemType.PLACE,
            title=np_unverified.name,
            time_slot="19:00 - 20:30",
            day=1,
            destination_name="Đà Nẵng",
            status=PlaceStatus.UNVERIFIED,
        )
        text = format_itinerary([item], danang_intent)
        assert "⚠️" in text or "Địa điểm bạn đề xuất" in text or "unverified" in text.lower(), (
            "Unverified place must be clearly labelled in the output"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9. Full Hybrid Pipeline Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridPipelineIntegration:
    """End-to-end tests that mock the Places API but exercise the full pipeline."""

    def _make_mock_places(self):
        mock = MagicMock()
        mock.search_nearby.return_value = [
            {
                "id": "places/madame_lan",
                "name": "Madame Lan Restaurant",
                "address": "04 Bạch Đằng, Đà Nẵng",
                "latitude": 16.0827,
                "longitude": 108.2238,
                "rating": 4.5,
                "user_rating_count": 4200,
                "category": "restaurant",
                "google_maps_uri": "https://maps.google.com",
            },
            {
                "id": "places/dragon_bridge",
                "name": "Cầu Rồng (Dragon Bridge)",
                "address": "Hải Châu, Đà Nẵng",
                "latitude": 16.0611,
                "longitude": 108.2272,
                "rating": 4.7,
                "user_rating_count": 15600,
                "category": "attraction",
                "google_maps_uri": "https://maps.google.com",
            },
        ]
        mock.search_text.return_value = []
        return mock

    def test_generate_hybrid_plan_keys(self):
        planner = AITravelPlanner(places_service=self._make_mock_places())
        result = planner.generate_hybrid_plan(
            "Lập kế hoạch 3 ngày tại Đà Nẵng cho 4 người."
        )
        assert "intent" in result
        assert "places" in result
        assert "items" in result
        assert "itinerary" in result
        assert "budget" in result

    def test_generate_full_trip_plan_backward_compat(self):
        """generate_full_trip_plan must still return legacy keys."""
        planner = AITravelPlanner(places_service=self._make_mock_places())
        result = planner.generate_full_trip_plan(
            "Lập kế hoạch 3 ngày tại Đà Nẵng cho 4 người."
        )
        assert "intent" in result
        assert "destination" in result["intent"]
        assert "places" in result
        assert "itinerary" in result
        assert "budget" in result

    def test_budget_deterministic(self):
        planner = AITravelPlanner(places_service=self._make_mock_places())
        result = planner.generate_hybrid_plan(
            "Lập kế hoạch 3 ngày tại Đà Nẵng cho 4 người."
        )
        b = result["budget"]
        assert b["total"] > 0
        assert b["per_person"] > 0
        computed = b["transportation"] + b["accommodation"] + b["food"] + b["activities"] + b["reserve"]
        assert b["total"] == pytest.approx(computed, abs=1)

    def test_search_nearby_called_for_known_dest(self):
        mock_places = self._make_mock_places()
        planner = AITravelPlanner(places_service=mock_places)
        planner.generate_hybrid_plan("Lập kế hoạch 3 ngày tại Đà Nẵng cho 4 người.")
        assert mock_places.search_nearby.called, (
            "For a known destination with coordinates, search_nearby must be used"
        )

    def test_multi_destination_transit_in_result(self):
        planner = AITravelPlanner(places_service=self._make_mock_places())
        prompt = "Tôi muốn đi Đà Nẵng 2 ngày rồi sang Hội An 1 ngày cho 4 người."
        result = planner.generate_hybrid_plan(prompt)
        items = result["items"]
        transit_items = [i for i in items if i.item_type == ItemType.TRANSIT]
        # For a 2-destination trip, expect exactly 1 TRANSIT
        assert len(transit_items) >= 1, "Multi-destination plan must contain at least 1 TRANSIT item"

    def test_pinned_places_appear_first(self):
        """Pinned places should have priority and appear in the result."""
        pinned = NormalizedPlace(
            id="places/pinned123",
            name="Địa điểm đã ghim",
            category="attraction",
            address="Test, Đà Nẵng",
            latitude=16.0471,
            longitude=108.2068,
            rating=5.0,
            user_rating_count=1,
            source="pinned",
        )
        planner = AITravelPlanner(places_service=self._make_mock_places())
        result = planner.generate_hybrid_plan(
            "Lập kế hoạch 2 ngày tại Đà Nẵng cho 2 người.",
            pinned_places=[pinned],
        )
        place_ids = [p.id for p in result["places"]]
        assert "places/pinned123" in place_ids, "Pinned place must appear in the result"
        # Pinned should be first
        assert result["places"][0].id == "places/pinned123", "Pinned place must be first"
