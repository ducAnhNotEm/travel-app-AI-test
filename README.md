# ✈️ Travel Itinerary Bot

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Local LLM](https://img.shields.io/badge/LLM-Gemma%204-FF6F00?logo=google&logoColor=white)
![Privacy First](https://img.shields.io/badge/Privacy-First-blueviolet?logo=lock&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Powered-black?logo=ollama&logoColor=white)

**AI-powered vacation planner that builds day-by-day itineraries with activities, restaurants, costs, and packing lists — running 100% locally with Gemma 4.**

```
+---------------------------------------------------------+
|                TRAVEL ITINERARY BOT                     |
|                                                         |
|  +----------+    +---------------+    +--------------+  |
|  | Traveler  |--->|  CLI / Web UI |--->|  Planner     |  |
|  | Input     |<---|  (Rich /      |<---|  Core        |  |
|  | (dest,    |    |  Streamlit)   |    +------+-------+  |
|  |  days,    |    +---------------+           |          |
|  |  budget)  |                                v          |
|  +----------+    +---------------+    +--------------+  |
|                  | Budget        |<-->|  Ollama API  |  |
|  +----------+    | Breakdown &   |    |  (Gemma 4)   |  |
|  | Saved     |<--| Packing List  |    |  :11434      |  |
|  | Itiner.   |   | Generator     |    +--------------+  |
|  | (JSON)    |   +---------------+                      |
|  +----------+          |                                |
|                  +-----v--------+                       |
|                  | Multi-Dest.  |                       |
|                  | Planner      |                       |
|                  | (A -> B -> C)|                       |
|                  +--------------+                       |
+---------------------------------------------------------+
```

## Features

- **Day-by-Day Itineraries** — Morning, afternoon, and evening activities with time estimates and logistics
- **Multi-Destination Trips** — Chain multiple cities into a single itinerary with transit planning
- **3 Budget Tiers** — Budget, moderate, and luxury options with estimated costs per activity
- **Restaurant Picks** — Local food recommendations for every meal slot
- **Place Details** — Drill into any attraction for hours, fees, tips, and nearby sights
- **Budget Breakdown** — Detailed cost estimate covering accommodation, food, transport, and activities
- **Packing List Generator** — Weather-appropriate packing lists tailored to your destination and interests
- **Save & Compare** — Store itineraries locally as JSON for future reference
- **Web UI + CLI** — Streamlit dashboard for visual planning or Rich terminal for quick generation
- **100% Local & Private** — Your travel plans never leave your machine; no cloud APIs

## Quick Start

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | 3.11+   |
| Ollama      | latest  |
| Gemma 4     | via Ollama |

### Install & Run

```bash
# 1. Clone the repository
git clone https://github.com/kennedyraju55/travel-itinerary-bot.git
cd travel-itinerary-bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama and pull Gemma 4
ollama serve &
ollama pull gemma4

# 4a. Launch the Web UI
streamlit run src/travel_planner/web_ui.py

# 4b. Or use the CLI
python -m travel_planner.cli plan --destination "Tokyo" --days 5 --budget moderate
```

### Docker

```bash
docker-compose up
# Web UI at http://localhost:8501
```

## Tech Stack

| Layer        | Technology                          |
|-------------|--------------------------------------|
| LLM          | Gemma 4 via Ollama                  |
| Backend      | Python 3.11, Click CLI              |
| Web UI       | Streamlit                           |
| API          | FastAPI / Uvicorn                   |
| Terminal UI  | Rich (panels, tables, progress)     |
| Config       | PyYAML                              |
| Data         | pandas                              |
| Testing      | pytest                              |
| Containers   | Docker, Docker Compose              |

## Project Structure

```
travel-itinerary-bot/
├── src/travel_planner/
│   ├── core.py         # Itinerary, budget, packing list logic
│   ├── cli.py          # Click CLI with Rich output
│   ├── web_ui.py       # Streamlit web dashboard
│   ├── api.py          # FastAPI REST endpoints
│   ├── config.py       # YAML configuration loader
│   └── utils.py        # LLM client helpers & utilities
├── common/
│   └── llm_client.py   # Shared Ollama/Gemma 4 client
├── tests/
│   ├── test_core.py    # Unit tests for core logic
│   └── test_cli.py     # CLI integration tests
├── config.yaml         # Travel defaults, model settings
├── requirements.txt    # Python dependencies
├── Dockerfile          # Multi-stage Docker build
├── docker-compose.yml  # Full stack with Ollama
├── Makefile            # Dev shortcuts (install, test, run)
└── setup.py            # Package setup with entry points
```

## 🌟 TripMate Foundation & Extensions

This repository serves as the AI travel-planning foundation for **TripMate** — a collaborative university group travel planner.

### TripMate Architecture

```
                      +-----------------------------+
                      |     Frontend (Streamlit)    |
                      |   or External React/Vue App |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     Backend / FastAPI API   |
                      +--------------+--------------+
                                     |
              +----------------------+----------------------+
              |                      |                      |
              v                      v                      v
     +-----------------+    +-----------------+    +-----------------+
     | AI Planner      |    |  Budget Engine  |    | Temporary Trip  |
     | (Intent Extract)|    | (Deterministic) |    | Fund Manager    |
     +--------+--------+    +--------+--------+    +-----------------+
              |                      |                      ^
              v                      |                      |
     +-----------------+             |              (Fundraising ->
     | Google Places   |             |               Fully Funded ->
     | API (New)       |             |               Confirmed)
     +--------+--------+             |
              |                      |
              v                      v
     +-----------------+    +-----------------+
     | Real Places     |--->| Authoritative   |
     | Verified        |    | Financial Plan  |
     +-----------------+    +-----------------+
```

### Core TripMate Additions
1. **Google Places API (New)**: `src/travel_planner/places.py`
   - Text Search (`/places:searchText`)
   - Nearby Search (`/places:searchNearby`)
   - Place Details (`/places/{placeId}`)
   - Field masks used to minimize quota and avoid wildcards.
   - Fallback adapter for Geoapify API.
2. **AI Intent Extraction & Anti-Hallucination**: `src/travel_planner/ai_planner.py`
   - Maps natural-language user prompt to structured JSON search queries.
   - AI is strictly constrained to recommend only real places returned by Google Places.
3. **Deterministic Budget Engine**: `src/travel_planner/budget.py`
   - Authoritative backend arithmetic (the LLM is never the source of truth for calculations).
   - Generates breakdown for accommodation, food, transportation, activities, reserve buffer (10%), total, and per person.
4. **Temporary Trip Fund**: `src/travel_planner/fund.py`
   - Lifecycle: `DRAFT` → `FUNDRAISING` → `FULLY_FUNDED` → `CONFIRMED`.
   - Member contribution tracking. Auto-transitions to `FULLY_FUNDED` when `total_contributed >= target_amount`.

---

## Quick Start & Setup

### 1. Install Dependencies

```bash
git clone https://github.com/kennedyraju55/travel-itinerary-bot.git
cd travel-itinerary-bot

# Install requirements + API server dependencies
pip install -r requirements.txt fastapi uvicorn
```

### 2. Configure Environment Variables

Create or update your `.env` file:

```bash
# Model Settings (Ollama / Gemma 4)
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=gemma4

# Google Maps Platform (Places API New)
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here

# Optional: Geoapify API Key fallback
GEOAPIFY_API_KEY=your_geoapify_api_key_here
```

### 3. Google Cloud Platform Setup

To use Google Places API (New):
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Select or create your project.
3. In **APIs & Services > Library**, search for and enable:
   - **Places API (New)**
   - (Optional) **Geocoding API**
4. In **APIs & Services > Credentials**, create an API key.
5. (Recommended) Restrict the API key to "Places API (New)".
6. Set `GOOGLE_MAPS_API_KEY=your_key` in your environment or `.env`.

### 4. Run the Application

#### Streamlit Web UI (Recommended for visual experience)
```bash
streamlit run src/travel_planner/web_ui.py
# Access at: http://localhost:8501
```

#### FastAPI Backend (REST endpoints)
```bash
python -m uvicorn src.travel_planner.api:app --reload --port 8004
# API Documentation (Swagger): http://localhost:8004/docs
```

#### CLI
```bash
python -m travel_planner.cli --destination "Da Nang" --days 3 --budget moderate --travelers 4
```

#### Run All Tests
```bash
python -m pytest
```

---

## Example API Requests & Responses

### 1. Full TripMate AI Plan
**POST** `/tripmate/plan`
```json
{
  "prompt": "Plan a 3-day trip to Da Nang for 4 people. Find restaurants and attractions and estimate the budget."
}
```
**Response Preview**:
```json
{
  "status": "success",
  "plan": {
    "intent": {
      "destination": "Da Nang",
      "days": 3,
      "travelers": 4,
      "transportation": "car",
      "tier": "moderate"
    },
    "places": [
      {
        "id": "places/ChIJ...",
        "name": "Dragon Bridge (Cầu Rồng)",
        "address": "An Hải Tây, Sơn Trà, Đà Nẵng",
        "rating": 4.7
      }
    ],
    "itinerary": "# 🗺️ Da Nang — 3-Day Trip Plan...",
    "budget": {
      "transportation": 1050000.0,
      "accommodation": 3200000.0,
      "food": 5400000.0,
      "activities": 1800000.0,
      "reserve": 1145000.0,
      "total": 12595000.0,
      "per_person": 3149000.0
    }
  }
}
```

### 2. Temporary Fund Contribution
**POST** `/tripmate/fund/danang-2026/contribute`
```json
{
  "member_name": "Anh",
  "amount": 3149000
}
```
**Response**:
```json
{
  "status": "success",
  "fund": {
    "trip_id": "danang-2026",
    "target_amount": 12595000.0,
    "total_contributed": 3149000.0,
    "progress_percent": 25.0,
    "status": "FUNDRAISING",
    "members": [
      {"name": "Anh", "target": 3149000.0, "paid": 3149000.0, "status": "paid"}
    ]
  }
}
```

---

## Author & Project Credit

- Original Bot Author: **Nrk Raju Guthikonda**
- TripMate University Project Extension: TripMate Foundation Integration
