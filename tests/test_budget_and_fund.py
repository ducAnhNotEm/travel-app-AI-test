"""Tests for deterministic budget calculation engine and temporary trip fund."""

import pytest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.travel_planner.budget import calculate_budget
from src.travel_planner.fund import (
    TripFundManager,
    TripFundError,
    STATUS_DRAFT,
    STATUS_FUNDRAISING,
    STATUS_FULLY_FUNDED,
    STATUS_CONFIRMED,
)


class TestBudgetEngine:
    def test_budget_basic_calculation(self):
        result = calculate_budget(travelers=4, days=3, transportation="car", tier="moderate")
        
        # Check required schema keys
        assert "transportation" in result
        assert "accommodation" in result
        assert "food" in result
        assert "activities" in result
        assert "reserve" in result
        assert "total" in result
        assert "per_person" in result

        # Check arithmetic truth
        subtotal = (
            result["transportation"]
            + result["accommodation"]
            + result["food"]
            + result["activities"]
        )
        assert result["total"] == subtotal + result["reserve"]
        assert result["per_person"] == round(result["total"] / 4, -3)

    def test_budget_increases_with_travelers_and_days(self):
        b1 = calculate_budget(travelers=2, days=3)
        b2 = calculate_budget(travelers=4, days=3)
        assert b2["food"] > b1["food"]
        assert b2["total"] > b1["total"]

    def test_budget_tiers(self):
        budget_tier = calculate_budget(travelers=2, days=3, tier="budget")
        luxury_tier = calculate_budget(travelers=2, days=3, tier="luxury")
        assert luxury_tier["accommodation"] > budget_tier["accommodation"]
        assert luxury_tier["total"] > budget_tier["total"]


class TestTripFundManager:
    def test_create_fund(self):
        mgr = TripFundManager()
        fund = mgr.create_fund(
            trip_id="test-fund-1",
            target_amount=6_600_000,
            member_names=["Anh", "Minh", "Lan", "Hung"]
        )
        assert fund["trip_id"] == "test-fund-1"
        assert fund["target_amount"] == 6_600_000
        assert len(fund["members"]) == 4
        assert fund["members"][0]["target"] == 1_650_000
        assert fund["status"] == STATUS_FUNDRAISING
        assert fund["total_contributed"] == 0

    def test_partial_contributions(self):
        mgr = TripFundManager()
        mgr.create_fund(
            trip_id="test-fund-2",
            target_amount=6_600_000,
            member_names=["Anh", "Minh", "Lan", "Hung"]
        )
        # Anh contributes full share
        f = mgr.contribute("test-fund-2", "Anh", 1_650_000)
        assert f["total_contributed"] == 1_650_000
        assert f["progress_percent"] == 25.0
        assert f["status"] == STATUS_FUNDRAISING
        anh = next(m for m in f["members"] if m["name"] == "Anh")
        assert anh["status"] == "paid"

    def test_fully_funded_state_transition(self):
        mgr = TripFundManager()
        mgr.create_fund(
            trip_id="test-fund-3",
            target_amount=6_600_000,
            member_names=["Anh", "Minh", "Lan", "Hung"]
        )
        mgr.contribute("test-fund-3", "Anh", 1_650_000)
        mgr.contribute("test-fund-3", "Minh", 1_650_000)
        mgr.contribute("test-fund-3", "Lan", 1_650_000)
        f = mgr.contribute("test-fund-3", "Hung", 1_650_000)

        # total_contributed == target_amount -> status must be FULLY_FUNDED
        assert f["total_contributed"] >= 6_600_000
        assert f["status"] == STATUS_FULLY_FUNDED
        assert f["progress_percent"] == 100.0

    def test_confirm_trip_rules(self):
        mgr = TripFundManager()
        mgr.create_fund(
            trip_id="test-fund-4",
            target_amount=6_600_000,
            member_names=["Anh", "Minh"]
        )
        # Attempt to confirm before fully funded must fail
        with pytest.raises(TripFundError) as exc:
            mgr.confirm_trip("test-fund-4")
        assert "not been reached" in str(exc.value)

        # Reach target and confirm
        mgr.contribute("test-fund-4", "Anh", 3_300_000)
        mgr.contribute("test-fund-4", "Minh", 3_300_000)
        f_confirmed = mgr.confirm_trip("test-fund-4")
        assert f_confirmed["status"] == STATUS_CONFIRMED

        # Cannot contribute after confirmed
        with pytest.raises(TripFundError) as exc2:
            mgr.contribute("test-fund-4", "Anh", 100_000)
        assert "already CONFIRMED" in str(exc2.value)
