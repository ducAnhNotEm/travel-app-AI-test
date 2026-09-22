"""FastAPI REST API for Travel Itinerary Bot & TripMate Foundation."""
import os
import sys
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.travel_planner.core import (
    generate_itinerary,
    generate_multi_destination_itinerary,
    get_place_details,
    generate_budget_breakdown,
    generate_packing_list,
    BUDGETS,
)
from src.travel_planner.places import PlacesService, PlacesAPIError
from src.travel_planner.budget import calculate_budget
from src.travel_planner.fund import (
    fund_manager,
    TripFundError,
    STATUS_FUNDRAISING,
)
from src.travel_planner.ai_planner import AITravelPlanner, extract_travel_intent

app = FastAPI(
    title="TripMate - Travel Itinerary Bot API",
    description="Collaborative Group Travel Planner REST API with Google Places (New), AI Intent Parsing, Budget Engine, and Temporary Trip Funds",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

places_service = PlacesService()
ai_planner = AITravelPlanner(places_service=places_service)


# ── Pydantic Request Models ──

class ItineraryRequest(BaseModel):
    destination: str
    days: int = 5
    budget: str = "moderate"
    interests: Optional[str] = None
    travelers: int = 1
    model: str = "gemma4"
    temperature: float = 0.7


class MultiDestRequest(BaseModel):
    destinations: list[str]
    days_per_dest: int = 3
    budget: str = "moderate"
    interests: Optional[str] = None
    travelers: int = 1
    model: str = "gemma4"
    temperature: float = 0.7


class PlaceRequest(BaseModel):
    place: str
    destination: str
    model: str = "gemma4"
    temperature: float = 0.7


class BudgetRequest(BaseModel):
    itinerary: str
    budget: str = "moderate"
    travelers: int = 1
    model: str = "gemma4"


class PackingListRequest(BaseModel):
    destination: str
    days: int = 5
    interests: Optional[str] = None
    model: str = "gemma4"


# TripMate Specific Models
class PlacesTextSearchRequest(BaseModel):
    query: str
    max_results: int = 8


class PlacesNearbyRequest(BaseModel):
    latitude: float
    longitude: float
    radius_meters: float = 3000.0
    included_types: Optional[List[str]] = None
    max_results: int = 8


class TripMatePlanRequest(BaseModel):
    prompt: str = Field(
        ...,
        example="Plan a 3-day trip to Da Nang for 4 people. Find restaurants and attractions and estimate the budget."
    )


class DeterministicBudgetRequest(BaseModel):
    travelers: int = 4
    days: int = 3
    transportation: str = "car"
    tier: str = "moderate"
    selected_places: Optional[List[Dict[str, Any]]] = None
    activities: Optional[List[str]] = None


class CreateFundRequest(BaseModel):
    trip_id: Optional[str] = None
    target_amount: float = 6_600_000
    members: Optional[List[str]] = ["Anh", "Minh", "Lan", "Hung"]


class ContributeRequest(BaseModel):
    member_name: str
    amount: float


# ── Core Existing Endpoints ──

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "TripMate Travel Planner API"}


@app.get("/budgets")
async def list_budgets():
    return {"budgets": BUDGETS}


@app.post("/itinerary")
async def create_itinerary(request: ItineraryRequest):
    try:
        result = generate_itinerary(
            destination=request.destination,
            days=request.days,
            budget=request.budget,
            interests=request.interests,
            travelers=request.travelers,
            model=request.model,
            temperature=request.temperature,
        )
        return {"itinerary": result, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/itinerary/multi")
async def create_multi_itinerary(request: MultiDestRequest):
    try:
        result = generate_multi_destination_itinerary(
            destinations=request.destinations,
            days_per_dest=request.days_per_dest,
            budget=request.budget,
            interests=request.interests,
            travelers=request.travelers,
            model=request.model,
            temperature=request.temperature,
        )
        return {"itinerary": result, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/place")
async def place_details(request: PlaceRequest):
    try:
        result = get_place_details(
            place=request.place,
            destination=request.destination,
            model=request.model,
            temperature=request.temperature,
        )
        return {"details": result, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/budget")
async def budget_breakdown(request: BudgetRequest):
    try:
        result = generate_budget_breakdown(
            itinerary=request.itinerary,
            budget=request.budget,
            travelers=request.travelers,
            model=request.model,
        )
        return {"breakdown": result, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/packing-list")
async def packing_list(request: PackingListRequest):
    try:
        result = generate_packing_list(
            destination=request.destination,
            days=request.days,
            interests=request.interests,
            model=request.model,
        )
        return {"packing_list": result, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Google Places API Endpoints (Phase 3) ──

@app.post("/places/search")
async def search_places(req: PlacesTextSearchRequest):
    """Google Places API (New) Text Search."""
    try:
        places = places_service.search_text(req.query, max_results=req.max_results)
        return {"places": places, "count": len(places), "status": "success"}
    except PlacesAPIError as e:
        raise HTTPException(status_code=e.status_code or 400, detail=e.message)


@app.post("/places/nearby")
async def nearby_places(req: PlacesNearbyRequest):
    """Google Places API (New) Nearby Search."""
    try:
        places = places_service.search_nearby(
            latitude=req.latitude,
            longitude=req.longitude,
            radius_meters=req.radius_meters,
            included_types=req.included_types,
            max_results=req.max_results,
        )
        return {"places": places, "count": len(places), "status": "success"}
    except PlacesAPIError as e:
        raise HTTPException(status_code=e.status_code or 400, detail=e.message)


@app.get("/places/{place_id}")
async def get_details(place_id: str):
    """Google Places API (New) Place Details by ID."""
    try:
        details = places_service.get_place_details(place_id)
        return {"details": details, "status": "success"}
    except PlacesAPIError as e:
        raise HTTPException(status_code=e.status_code or 400, detail=e.message)


from src.travel_planner.destinations import list_destinations, get_destination_info

# ── TripMate AI Planner & Intent (Phase 4 & 5) ──

@app.get("/tripmate/destinations")
async def get_supported_destinations():
    """Retrieve curated list of popular travel destinations with GPS coordinates and tags."""
    return {"destinations": list_destinations(), "status": "success"}


@app.post("/tripmate/parse-intent")
async def parse_intent(req: TripMatePlanRequest):
    """Extract structured travel intent without hallucinating places."""
    intent = extract_travel_intent(req.prompt)
    return {"intent": intent, "status": "success"}


@app.post("/tripmate/plan")
async def tripmate_plan(req: TripMatePlanRequest):
    """
    Complete TripMate end-to-end planning:
    Prompt -> AI Intent Extraction -> Google Places Query -> Verified Itinerary -> Deterministic Budget
    """
    try:
        plan = ai_planner.generate_full_trip_plan(req.prompt)
        return {"plan": plan, "status": "success"}
    except PlacesAPIError as e:
        raise HTTPException(status_code=e.status_code or 400, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Planning error: {str(e)}")


# ── TripMate Budget Foundation (Phase 6) ──

@app.post("/tripmate/budget/calculate")
async def calculate_trip_budget(req: DeterministicBudgetRequest):
    """
    Calculate deterministic budget breakdown based on authoritative backend arithmetic.
    """
    result = calculate_budget(
        travelers=req.travelers,
        days=req.days,
        transportation=req.transportation,
        tier=req.tier,
        selected_places=req.selected_places,
        activities=req.activities,
    )
    return {"budget": result, "status": "success"}


# ── TripMate Temporary Trip Fund (Phase 7) ──

@app.post("/tripmate/fund/create")
async def create_trip_fund(req: CreateFundRequest):
    """Create a temporary trip fund."""
    fund = fund_manager.create_fund(
        trip_id=req.trip_id,
        target_amount=req.target_amount,
        member_names=req.members,
        status=STATUS_FUNDRAISING,
    )
    return {"fund": fund, "status": "success"}


@app.get("/tripmate/fund/{trip_id}")
async def get_trip_fund(trip_id: str):
    """Get trip fund status and member contributions."""
    fund = fund_manager.get_fund(trip_id)
    if not fund:
        raise HTTPException(status_code=404, detail="Trip fund not found")
    return {"fund": fund, "status": "success"}


@app.post("/tripmate/fund/{trip_id}/contribute")
async def contribute_to_fund(trip_id: str, req: ContributeRequest):
    """
    Record member contribution.
    Auto-advances status to FULLY_FUNDED when total_contributed >= target_amount.
    """
    try:
        fund = fund_manager.contribute(
            trip_id=trip_id,
            member_name=req.member_name,
            amount=req.amount,
        )
        return {"fund": fund, "status": "success"}
    except TripFundError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/tripmate/fund/{trip_id}/confirm")
async def confirm_trip_fund(trip_id: str):
    """
    Confirm trip fund. Only allowed if status is FULLY_FUNDED.
    """
    try:
        fund = fund_manager.confirm_trip(trip_id=trip_id)
        return {"fund": fund, "status": "success", "message": "Trip successfully confirmed!"}
    except TripFundError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
