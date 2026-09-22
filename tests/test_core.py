"""Tests for Travel Itinerary Bot core logic."""

import pytest
from unittest.mock import patch

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.travel_planner.core import generate_itinerary, get_place_details, BUDGETS, SYSTEM_PROMPT
from src.travel_planner.utils import parse_destinations, parse_budget_items


class TestConfiguration:
    def test_budget_levels(self):
        assert "budget" in BUDGETS
        assert "moderate" in BUDGETS
        assert "luxury" in BUDGETS

    def test_system_prompt_travel_expert(self):
        assert "travel" in SYSTEM_PROMPT.lower()


class TestGenerateItinerary:
    @patch("src.travel_planner.core.chat")
    def test_basic_itinerary(self, mock_chat):
        mock_chat.return_value = "Day 1: Arrive in Tokyo\n- Visit Shibuya"
        result = generate_itinerary("Tokyo", 5, "moderate")
        assert "Day 1" in result
        mock_chat.assert_called_once()

    @patch("src.travel_planner.core.chat")
    def test_itinerary_includes_destination(self, mock_chat):
        mock_chat.return_value = "Itinerary..."
        generate_itinerary("Paris", 3, "luxury")
        messages = mock_chat.call_args[0][0]
        assert "Paris" in messages[0]["content"]

    @patch("src.travel_planner.core.chat")
    def test_itinerary_with_interests(self, mock_chat):
        mock_chat.return_value = "Itinerary..."
        generate_itinerary("Rome", 4, "budget", interests="history,food")
        messages = mock_chat.call_args[0][0]
        assert "history" in messages[0]["content"].lower()

    @patch("src.travel_planner.core.chat")
    def test_itinerary_with_travelers(self, mock_chat):
        mock_chat.return_value = "Itinerary..."
        generate_itinerary("London", 3, "moderate", travelers=4)
        messages = mock_chat.call_args[0][0]
        assert "4" in messages[0]["content"]

    @patch("src.travel_planner.core.chat")
    def test_uses_system_prompt(self, mock_chat):
        mock_chat.return_value = "Plan..."
        generate_itinerary("Tokyo", 2, "budget")
        assert mock_chat.call_args[1]["system_prompt"] == SYSTEM_PROMPT


class TestGetPlaceDetails:
    @patch("src.travel_planner.core.chat")
    def test_returns_details(self, mock_chat):
        mock_chat.return_value = "The Eiffel Tower is a landmark..."
        result = get_place_details("Eiffel Tower", "Paris")
        assert "Eiffel Tower" in result

    @patch("src.travel_planner.core.chat")
    def test_includes_destination(self, mock_chat):
        mock_chat.return_value = "Details..."
        get_place_details("Colosseum", "Rome")
        messages = mock_chat.call_args[0][0]
        assert "Rome" in messages[0]["content"]


class TestMultiDestination:
    def test_parse_single(self):
        assert parse_destinations("Tokyo") == ["Tokyo"]

    def test_parse_multiple(self):
        result = parse_destinations("Tokyo, Kyoto, Osaka")
        assert result == ["Tokyo", "Kyoto", "Osaka"]

    def test_parse_empty(self):
        assert parse_destinations("") == []


class TestBudgetParsing:
    def test_parse_budget_items(self):
        text = "- Accommodation: $500\n- Food: $300\n- Transport: $150"
        items = parse_budget_items(text)
        assert len(items) == 3
        assert items[0]["amount"] == 500.0

    def test_empty_text(self):
        assert parse_budget_items("no numbers here") == []


class TestLLMClientResilience:
    @patch("requests.post")
    def test_chat_raises_ollama_connection_error_without_exit(self, mock_post):
        import requests
        from common.llm_client import chat, OllamaConnectionError
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        with pytest.raises(OllamaConnectionError):
            chat([{"role": "user", "content": "hi"}])

