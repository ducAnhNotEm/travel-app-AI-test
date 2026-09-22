"""Tests for Google Places API (New) integration and Geoapify fallback."""

import os
import pytest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.travel_planner.places import PlacesService, PlacesAPIError, DEFAULT_FIELD_MASK


class TestPlacesService:
    def test_missing_api_key_raises_error(self):
        service = PlacesService(api_key="", geoapify_key="")
        with pytest.raises(PlacesAPIError) as exc:
            service.search_text("restaurants near Dragon Bridge Da Nang")
        assert "Missing Google Maps API key" in str(exc.value)
        assert exc.value.error_type == "missing_api_key"

    def test_empty_query_returns_empty_list(self):
        service = PlacesService(api_key="dummy_key")
        res = service.search_text("")
        assert res == []

    @patch("requests.Session.request")
    def test_successful_text_search(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "places": [
                {
                    "id": "places/ChIJ123",
                    "displayName": {"text": "Madame Lan"},
                    "formattedAddress": "04 Bach Dang, Da Nang",
                    "location": {"latitude": 16.0827, "longitude": 108.2238},
                    "rating": 4.5,
                    "userRatingCount": 3500,
                    "priceLevel": "PRICE_LEVEL_MODERATE",
                    "googleMapsUri": "https://maps.google.com/?cid=123"
                }
            ]
        }
        mock_request.return_value = mock_resp

        service = PlacesService(api_key="test_valid_key")
        results = service.search_text("Madame Lan Da Nang")
        
        assert len(results) == 1
        place = results[0]
        assert place["name"] == "Madame Lan"
        assert place["latitude"] == 16.0827
        assert place["rating"] == 4.5
        assert place["id"] == "places/ChIJ123"
        
        # Verify field mask sent
        args, kwargs = mock_request.call_args
        headers = kwargs.get("headers", {})
        assert headers.get("X-Goog-FieldMask") == DEFAULT_FIELD_MASK
        assert headers.get("X-Goog-Api-Key") == "test_valid_key"

    @patch("requests.Session.request")
    def test_invalid_api_key_error(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_request.return_value = mock_resp

        service = PlacesService(api_key="invalid_key")
        with pytest.raises(PlacesAPIError) as exc:
            service.search_text("Da Nang")
        assert exc.value.error_type == "auth_error"
        assert exc.value.status_code == 403

    @patch("requests.Session.request")
    def test_quota_exceeded_error(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_request.return_value = mock_resp

        service = PlacesService(api_key="test_key")
        with pytest.raises(PlacesAPIError) as exc:
            service.search_text("Da Nang")
        assert exc.value.error_type == "quota_exceeded"
        assert exc.value.status_code == 429

    @patch("requests.Session.request")
    def test_nearby_search(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "places": [
                {
                    "id": "places/ChIJ_nearby1",
                    "displayName": {"text": "Dragon Bridge"},
                    "formattedAddress": "An Hai Tay, Da Nang",
                    "location": {"latitude": 16.0611, "longitude": 108.2272},
                    "rating": 4.7,
                }
            ]
        }
        mock_request.return_value = mock_resp

        service = PlacesService(api_key="test_key")
        results = service.search_nearby(16.0611, 108.2272, radius_meters=2000)
        assert len(results) == 1
        assert results[0]["name"] == "Dragon Bridge"

    @patch("requests.Session.request")
    def test_place_details(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "places/ChIJ123",
            "displayName": {"text": "Madame Lan"},
            "formattedAddress": "04 Bach Dang, Da Nang",
            "websiteUri": "http://madamelan.vn",
            "nationalPhoneNumber": "0905 123 456"
        }
        mock_request.return_value = mock_resp

        service = PlacesService(api_key="test_key")
        details = service.get_place_details("places/ChIJ123")
        assert details["name"] == "Madame Lan"
        assert details["website_uri"] == "http://madamelan.vn"
        assert details["phone_number"] == "0905 123 456"

    @patch("requests.Session.get")
    def test_geoapify_nearby_search(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "features": [
                {
                    "properties": {
                        "place_id": "geo_123",
                        "name": "Pho 10 Ly Quoc Su",
                        "formatted": "10 Ly Quoc Su, Hanoi",
                        "lat": 21.0285,
                        "lon": 105.8542,
                        "rating": 4.6
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp

        service = PlacesService(api_key=None, geoapify_key="test_geo_key")
        results = service.search_geoapify_nearby(21.0285, 105.8542, radius_meters=3000)
        assert len(results) == 1
        assert results[0]["name"] == "Pho 10 Ly Quoc Su"
        assert results[0]["latitude"] == 21.0285
