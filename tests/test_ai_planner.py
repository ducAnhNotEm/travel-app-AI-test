"""Tests for AI Travel Request Extraction and Places Planning flow."""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.travel_planner.ai_planner import extract_travel_intent, AITravelPlanner
from src.travel_planner.places import PlacesService


class TestAITravelPlanner:
    def test_heuristic_extraction_da_nang(self):
        prompt = "Plan a 3-day trip to Da Nang for 4 people. Find restaurants and attractions and estimate the budget."
        intent = extract_travel_intent(prompt)

        assert intent["destination"] == "Da Nang"
        assert intent["days"] == 3
        assert intent["travelers"] == 4
        assert len(intent["places_queries"]) >= 1

    def test_heuristic_extraction_with_budget(self):
        prompt = "I want dinner near Dragon Bridge in Da Nang for 4 people, budget around 1,000,000 VND."
        intent = extract_travel_intent(prompt)

        assert intent["destination"] == "Da Nang"
        assert intent["travelers"] == 4
        assert intent["budget_vnd"] == 1_000_000.0
        # Category restaurant search query must be present
        queries = [q["query"].lower() for q in intent["places_queries"]]
        assert any("dragon bridge" in q or "restaurant" in q for q in queries)

    def test_full_trip_planner_flow_with_verified_places(self):
        # Mock places service returning specific verified places
        mock_places = MagicMock()
        mock_places.search_text.return_value = [
            {
                "id": "places/madame_lan",
                "name": "Madame Lan",
                "address": "04 Bach Dang, Da Nang",
                "latitude": 16.0827,
                "longitude": 108.2238,
                "rating": 4.5,
                "user_rating_count": 3500,
                "google_maps_uri": "https://maps.google.com/?cid=123",
                "category": "restaurant"
            },
            {
                "id": "places/dragon_bridge",
                "name": "Dragon Bridge (Cau Rong)",
                "address": "Son Tra, Da Nang",
                "latitude": 16.0611,
                "longitude": 108.2272,
                "rating": 4.7,
                "user_rating_count": 14000,
                "google_maps_uri": "https://maps.google.com/?cid=456",
                "category": "attraction"
            }
        ]

        planner = AITravelPlanner(places_service=mock_places)
        prompt = "Plan a 3-day trip to Da Nang for 4 people. Find restaurants and attractions and estimate the budget."
        result = planner.generate_full_trip_plan(prompt)

        # 1. Check intent
        assert result["intent"]["destination"] == "Da Nang"
        assert result["intent"]["days"] == 3
        assert result["intent"]["travelers"] == 4

        # 2. Check places are verified (no hallucinations)
        assert len(result["places"]) == 2
        place_names = [p["name"] for p in result["places"]]
        assert "Madame Lan" in place_names
        assert "Dragon Bridge (Cau Rong)" in place_names

        # 3. Check itinerary was synthesized
        assert "Madame Lan" in result["itinerary"] or "Dragon Bridge" in result["itinerary"]

        # 4. Check deterministic budget was generated
        assert result["budget"]["total"] > 0
        assert result["budget"]["per_person"] > 0
        assert result["budget"]["total"] == (
            result["budget"]["transportation"]
            + result["budget"]["accommodation"]
            + result["budget"]["food"]
            + result["budget"]["activities"]
            + result["budget"]["reserve"]
        )
