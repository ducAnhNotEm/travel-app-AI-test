# Final Refactored Master Plan: TripMate Precision POI Pipeline & 34-Province Architecture

**Goal**: Transform TripMate into a 100% robust, Vietnam-centric travel AI system without rewriting. Refactor in strict architectural dependency order: `places.py` (Phase 1) ➔ `destinations.py` (Phase 2) ➔ `ai_planner.py` (Phase 3) ➔ `web_ui.py` (Phase 4) ➔ `tests` (Phase 5).

---

## 🏛️ Executive Summary & Key Architectural Refinements

1. **Strict Dependency Execution Order**:
   - **Phase 1: `places.py`**: Fix POI Geocoding flaw by replacing `/v1/geocode/search` with Geoapify `/v2/places` category searches centered on Lat/Lng coordinates. Enforce `countrycode=vn` and `lang=vi`.
   - **Phase 2: `destinations.py`**: Structure catalog around **34 Merged Provinces/Cities of Vietnam + Key Travel Hubs** (Phú Quốc, Hạ Long, Sa Pa, Hội An) with exact Lat/Lng coordinates.
   - **Phase 3: `ai_planner.py`**: Implement strict POI RAG pipeline:
     $$\text{USER INPUT} \longrightarrow \text{STRUCTURED INTENT} \longrightarrow \text{POI SERVICE (Lat/Lng + Categories)} \longrightarrow \text{VERIFIED REAL POIs} \longrightarrow \text{AI SYNTHESIZER}$$
     *AI is strictly forbidden from inventing venue names; it selects ONLY from verified POIs.*
   - **Phase 4: `web_ui.py`**: Dropdown selectbox UI for destinations (no manual typing required), removing all Đà Nẵng hardcode fallbacks.
   - **Phase 5: `tests/`**: Automated verification.

2. **Refined Localization Rules**:
   - **UI + AI Explanations + Logistics + Addresses**: Rendered in Vietnamese.
   - **Official Brand Names / Proper Nouns**: Retained in official registered names (e.g. *Madame Lan Restaurant*, *Cầu Rồng*, *VinWonders Phú Quốc*, *Bún Chả Hương Liên*) — NO weird translations.

---

## 🔍 User Review Required

> [!IMPORTANT]
> - **Zero Geocoding Flaws**: Eliminates text geocoding queries ("top attractions in Phu Quoc") and enforces Lat/Lng + POI Category searching (`catering.restaurant`, `tourism.sights`, `accommodation.hotel`).
> - **Strict POI Injection**: The AI synthesizer is passed a verified array of real POIs and cannot invent un-retrieved venue names.
> - **Preserve Brand Names**: Proper nouns and venue titles remain in their canonical form.

---

## 🛠️ Detailed Phase Execution Scope

### Phase 1: Core POI Service (`src/travel_planner/places.py`)
- **Replace Geoapify Geocoder**: Deprecate `/v1/geocode/search`. Implement `/v2/places` endpoint with `categories` (`catering.restaurant`, `tourism.sights`, `accommodation.hotel`), `filter=circle:{lng},{lat},{radius_meters}`, `filter=countrycode:vn`, and `lang=vi`.
- **Google Places (New) API**: Enforce `searchNearby` and `searchText` with `locationBias` (Coordinates + Circle radius) and `regionCode: "VN"`.
- **OpenStreetMap POI Fallback**: Implement Nominatim/Overpass POI query with bounding box and `countrycodes=vn`.
- **Normalized POI Schema**: Output uniform POI dictionary across all providers:
  `{id, name, address, latitude, longitude, rating, user_rating_count, category, google_maps_uri}`

---

### Phase 2: Destinations Catalog (`src/travel_planner/destinations.py`)
- **34 Merged Provinces + Tourism Hubs Structure**:
  ```python
  DESTINATIONS_CATALOG = {
      "Phú Quốc": {
          "id": "phuquoc",
          "name": "Phú Quốc",
          "province": "Kiên Giang",
          "latitude": 10.2899,
          "longitude": 103.9840,
          "tags": ["🏖️ Biển Bãi Sao", "🦀 Hải sản Hàm Ninh", "🎡 VinWonders"],
          "sample_places": [...]
      },
      "Hạ Long": {
          "id": "halong",
          "name": "Hạ Long",
          "province": "Quảng Ninh",
          "latitude": 20.9599,
          "longitude": 107.0425,
          "tags": ["🛥️ Vịnh Hạ Long", "🏔️ Hang Đầu Gỗ", "🦀 Hải sản"],
          "sample_places": [...]
      },
      ... # 34 official Vietnam administrative units + hubs
  }
  ```
- **Fuzzy Unicode Resolution**: Accent-insensitive matcher (`phu quoc` -> `Phú Quốc`).

---

### Phase 3: AI Planner & Intent Pipeline (`src/travel_planner/ai_planner.py`)
- **Structured Intent Pipeline**:
  Extract `{destination, days, travelers, categories}` from prompt.
- **POI Retrieval**: Query `PlacesService` using destination's Lat/Lng + requested categories.
- **Strict RAG Itinerary Synthesizer**: Prompt LLM with retrieved real POIs. Mandate that venue names in itinerary match the verified POI list without hallucinating fictitious shops.

---

### Phase 4: Streamlit UI (`src/travel_planner/web_ui.py`)
- Replace text inputs with structured `st.selectbox` for 34 Provinces & Tourism Hubs.
- Remove hardcoded `"danang-2026"` trip ID and defaults.

---

### Phase 5: Automated Verification (`tests/`)
- Run `pytest` on all test files.
- Write tests confirming POI Lat/Lng queries, 34-province resolution, and zero hallucination.

---

## 🧪 Verification Plan

### Automated Tests
- Execute `pytest` across `tests/test_places.py`, `tests/test_ai_planner.py`, `tests/test_core.py`.

### Manual Verification
- Test selecting "Phú Quốc", "Hạ Long", "Cần Thơ" via UI selectbox and generate 3-day plans.
- Confirm 100% of generated places are real, located in the selected region, and rendered in Vietnamese with proper brand names.
