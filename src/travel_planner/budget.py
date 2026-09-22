"""Deterministic Budget Engine for TripMate Foundation.

Calculates:
- transportation
- accommodation
- food
- activities
- reserve
- total
- per_person

Strict arithmetic calculations - LLM is NOT the source of truth for financial numbers.
"""

from typing import Dict, Any, List, Optional
import math


# Default daily base rates in VND (standard benchmark for Vietnam / travel estimation)
BASE_RATES_VND = {
    "accommodation": {
        "budget": 350_000,     # Hostel / Budget hotel per room (2 people)
        "moderate": 800_000,    # 3-4 star hotel
        "luxury": 2_500_000,   # 5-star resort
    },
    "food": {
        "budget": 200_000,     # Street food / local eateries per person/day
        "moderate": 450_000,   # Sit-down restaurants, cafes per person/day
        "luxury": 1_200_000,   # Fine dining / seafood per person/day
    },
    "transportation": {
        "public": 100_000,     # Bus / bike rental per person/day
        "car": 350_000,        # Private car / taxi split per car/day (~4 pax)
        "motorcycle": 150_000, # Motorbike rental per bike/day (~2 pax)
        "flight": 1_800_000,   # Domestic return flight per person
    },
    "activity_default": 150_000, # Average ticket/admission fee
    "reserve_percentage": 0.10,  # 10% contingency buffer
}


def calculate_budget(
    travelers: int = 1,
    days: int = 3,
    transportation: str = "car",
    tier: str = "moderate",
    selected_places: Optional[List[Dict[str, Any]]] = None,
    activities: Optional[List[str]] = None,
    custom_accommodation: Optional[float] = None,
    custom_transportation: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate deterministic budget breakdown for a trip.

    Args:
        travelers: Number of participants (>= 1)
        days: Duration in days (>= 1)
        transportation: Mode of transit ('car', 'motorcycle', 'public', 'flight')
        tier: 'budget', 'moderate', or 'luxury'
        selected_places: List of places with priceLevel or admission details
        activities: List of custom activities planned
        custom_accommodation: Optional direct override for accommodation total
        custom_transportation: Optional direct override for transportation total

    Returns:
        {
            "transportation": float,
            "accommodation": float,
            "food": float,
            "activities": float,
            "reserve": float,
            "total": float,
            "per_person": float
        }
    """
    travelers = max(1, int(travelers))
    days = max(1, int(days))
    tier = tier.lower() if tier in ("budget", "moderate", "luxury") else "moderate"
    trans_mode = transportation.lower() if transportation.lower() in BASE_RATES_VND["transportation"] else "car"

    # 1. Accommodation Calculation
    if custom_accommodation is not None:
        accommodation_cost = float(custom_accommodation)
    else:
        # Assuming 2 travelers per room
        rooms_needed = math.ceil(travelers / 2.0)
        daily_room_rate = BASE_RATES_VND["accommodation"][tier]
        # Nights = days - 1 if days > 1, or minimum 1 night
        nights = max(1, days - 1)
        accommodation_cost = float(rooms_needed * daily_room_rate * nights)

    # 2. Food Calculation
    daily_food_rate = BASE_RATES_VND["food"][tier]
    food_cost = float(daily_food_rate * travelers * days)

    # 3. Transportation Calculation
    if custom_transportation is not None:
        transportation_cost = float(custom_transportation)
    else:
        if trans_mode == "car":
            cars_needed = math.ceil(travelers / 4.0)
            daily_car_rate = BASE_RATES_VND["transportation"]["car"]
            transportation_cost = float(cars_needed * daily_car_rate * days)
        elif trans_mode == "motorcycle":
            bikes_needed = math.ceil(travelers / 2.0)
            daily_bike_rate = BASE_RATES_VND["transportation"]["motorcycle"]
            transportation_cost = float(bikes_needed * daily_bike_rate * days)
        elif trans_mode == "flight":
            flight_cost = BASE_RATES_VND["transportation"]["flight"] * travelers
            local_transit = BASE_RATES_VND["transportation"]["car"] * math.ceil(travelers / 4.0) * days
            transportation_cost = float(flight_cost + local_transit)
        else: # public
            daily_public = BASE_RATES_VND["transportation"]["public"]
            transportation_cost = float(daily_public * travelers * days)

    # 4. Activities Calculation
    num_activities = len(activities or [])
    num_places = len(selected_places or [])
    effective_activities = max(num_activities, num_places, days * 2) # At least 2 attractions per day
    activity_unit_cost = BASE_RATES_VND["activity_default"]
    
    # Adjust activity unit cost slightly by tier
    if tier == "budget":
        activity_unit_cost = 80_000
    elif tier == "luxury":
        activity_unit_cost = 350_000

    activities_cost = float(effective_activities * activity_unit_cost * travelers)

    # 5. Subtotal & Reserve Buffer (10% contingency)
    subtotal = accommodation_cost + food_cost + transportation_cost + activities_cost
    reserve_cost = float(round(subtotal * BASE_RATES_VND["reserve_percentage"], -3))

    # 6. Total and Per Person
    total_cost = float(round(subtotal + reserve_cost, -3))
    per_person_cost = float(round(total_cost / travelers, -3))

    return {
        "transportation": float(round(transportation_cost, -3)),
        "accommodation": float(round(accommodation_cost, -3)),
        "food": float(round(food_cost, -3)),
        "activities": float(round(activities_cost, -3)),
        "reserve": reserve_cost,
        "total": total_cost,
        "per_person": per_person_cost,
    }
