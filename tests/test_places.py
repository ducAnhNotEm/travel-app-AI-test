"""Tests for Google Places API (New) integration and Geoapify fallback.

Covers:
- POI coordinate-based (Lat/Lng) search enforcement
- Vietnam countrycode filter (countrycode:vn, lang:vi)
- Normalized POI schema with 'category' field
- regionCode VN in Google Places calls
- Deprecated text geocoding warning
"""

import os
import logging
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
                    "displayName": {"text": "Madame Lan Restaurant"},
                    "formattedAddress": "04 Bạch Đằng, Đà Nẵng",
                    "location": {"latitude": 16.0827, "longitude": 108.2238},
                    "rating": 4.5,
                    "userRatingCount": 3500,
                    "googleMapsUri": "https://maps.google.com/?cid=123"
                }
            ]
        }
        mock_request.return_value = mock_resp

        service = PlacesService(api_key="test_valid_key")
        results = service.search_text("Madame Lan Da Nang")

        assert len(results) == 1
        place = results[0]
        assert place["name"] == "Madame Lan Restaurant"
        assert place["latitude"] == 16.0827
        assert place["rating"] == 4.5
        assert place["id"] == "places/ChIJ123"
        # category field must exist in normalized schema
        assert "category" in place

        # Verify field mask + API key sent
        args, kwargs = mock_request.call_args
        headers = kwargs.get("headers", {})
        assert headers.get("X-Goog-FieldMask") == DEFAULT_FIELD_MASK
        assert headers.get("X-Goog-Api-Key") == "test_valid_key"

    @patch("requests.Session.request")
    def test_search_text_sends_region_code_vn(self, mock_request):
        """search_text must include regionCode: VN to bias Google Places results to Vietnam."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"places": []}
        mock_request.return_value = mock_resp

        service = PlacesService(api_key="test_key")
        service.search_text("nhà hàng Hà Nội")

        args, kwargs = mock_request.call_args
        payload = kwargs.get("json", {})
        assert payload.get("regionCode") == "VN", (
            "search_text must send regionCode='VN' to bias results toward Vietnam"
        )

    @patch("requests.Session.request")
    def test_invalid_api_key_error(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_request.return_value = mock_resp

        service = PlacesService(api_key="invalid_key")
        with pytest.raises(PlacesAPIError) as exc:
            service.search_text("Đà Nẵng")
        assert exc.value.error_type == "auth_error"
        assert exc.value.status_code == 403

    @patch("requests.Session.request")
    def test_quota_exceeded_error(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_request.return_value = mock_resp

        service = PlacesService(api_key="test_key")
        with pytest.raises(PlacesAPIError) as exc:
            service.search_text("Đà Nẵng")
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
                    "displayName": {"text": "Cầu Rồng (Dragon Bridge)"},
                    "formattedAddress": "An Hải Tây, Đà Nẵng",
                    "location": {"latitude": 16.0611, "longitude": 108.2272},
                    "rating": 4.7,
                }
            ]
        }
        mock_request.return_value = mock_resp

        service = PlacesService(api_key="test_key")
        results = service.search_nearby(16.0611, 108.2272, radius_meters=2000)
        assert len(results) == 1
        assert results[0]["name"] == "Cầu Rồng (Dragon Bridge)"
        assert "category" in results[0]

    @patch("requests.Session.request")
    def test_place_details(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "places/ChIJ123",
            "displayName": {"text": "Madame Lan Restaurant"},
            "formattedAddress": "04 Bạch Đằng, Đà Nẵng",
            "websiteUri": "http://madamelan.vn",
            "nationalPhoneNumber": "0905 123 456"
        }
        mock_request.return_value = mock_resp

        service = PlacesService(api_key="test_key")
        details = service.get_place_details("places/ChIJ123")
        assert details["name"] == "Madame Lan Restaurant"
        assert details["website_uri"] == "http://madamelan.vn"
        assert details["phone_number"] == "0905 123 456"

    @patch("requests.Session.get")
    def test_geoapify_nearby_enforces_countrycode_vn(self, mock_get):
        """Geoapify /v2/places calls must include filter=countrycode:vn and lang=vi."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "features": [
                {
                    "properties": {
                        "place_id": "geo_001",
                        "name": "Phở 10 Lý Quốc Sứ",
                        "formatted": "10 Lý Quốc Sứ, Hà Nội",
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

        # Check URL enforced countrycode:vn and lang=vi
        call_url = mock_get.call_args[0][0]
        assert "countrycode:vn" in call_url, "Must enforce filter=countrycode:vn"
        assert "lang=vi" in call_url, "Must enforce lang=vi"
        assert "/v2/places" in call_url, "Must use /v2/places endpoint, not /v1/geocode/search"

        assert len(results) == 1
        assert results[0]["name"] == "Phở 10 Lý Quốc Sứ"
        assert results[0]["latitude"] == 21.0285
        # category field should be set
        assert "category" in results[0]

    @patch("requests.Session.get")
    def test_geoapify_text_search_deprecated_logs_warning(self, mock_get, caplog):
        """search_geoapify() without coordinates must log a deprecation warning."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"features": []}
        mock_get.return_value = mock_resp

        service = PlacesService(api_key=None, geoapify_key="test_geo_key")
        with caplog.at_level(logging.WARNING):
            service.search_geoapify(query="nhà hàng Đà Nẵng")

        assert any("deprecated" in r.message.lower() for r in caplog.records), (
            "Calling search_geoapify() without coordinates must emit a deprecation warning"
        )

    @patch("requests.Session.get")
    def test_geoapify_text_search_enforces_vn_filter(self, mock_get):
        """search_geoapify() text fallback must still enforce countrycode:vn + lang=vi."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"features": []}
        mock_get.return_value = mock_resp

        service = PlacesService(api_key=None, geoapify_key="test_geo_key")
        service.search_geoapify(query="restaurant Da Nang")

        call_url = mock_get.call_args[0][0]
        assert "countrycode:vn" in call_url
        assert "lang=vi" in call_url

    @patch("requests.Session.request")
    def test_normalized_schema_has_category_field(self, mock_request):
        """Normalized POI dict must always include 'category' key (unified schema)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "places": [
                {
                    "id": "places/abc",
                    "displayName": {"text": "Test Place"},
                    "formattedAddress": "123 Street, Hà Nội",
                    "location": {"latitude": 21.0, "longitude": 105.8},
                    "rating": 4.0,
                }
            ]
        }
        mock_request.return_value = mock_resp

        service = PlacesService(api_key="test_key")
        results = service.search_text("test")
        assert len(results) == 1
        assert "category" in results[0], "Normalized POI schema must include 'category' field"
        assert "google_maps_uri" in results[0]
        # price_level should NOT be in the schema (replaced by category)
        assert "price_level" not in results[0]
