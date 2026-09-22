"""Temporary Trip Fund Module for TripMate foundation.

Provides:
- Fund tracking with states: DRAFT, FUNDRAISING, FULLY_FUNDED, CONFIRMED
- Member contribution tracking
- Automatic state transition: when total_contributed >= target_amount -> FULLY_FUNDED
- Trip confirmation validation: only FULLY_FUNDED can transition to CONFIRMED
- Clean in-memory and JSON storage
"""

import json
import uuid
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Fund status constants
STATUS_DRAFT = "DRAFT"
STATUS_FUNDRAISING = "FUNDRAISING"
STATUS_FULLY_FUNDED = "FULLY_FUNDED"
STATUS_CONFIRMED = "CONFIRMED"

VALID_STATUSES = {STATUS_DRAFT, STATUS_FUNDRAISING, STATUS_FULLY_FUNDED, STATUS_CONFIRMED}


class TripFundError(Exception):
    """Exception for fund state violations."""
    pass


class TripFundManager:
    """Manages temporary trip funds."""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self._funds: Dict[str, Dict[str, Any]] = {}
        if self.storage_path and self.storage_path.exists():
            self._load()

    def _load(self):
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self._funds = {item["trip_id"]: item for item in data}
        except Exception as e:
            logger.warning("Failed to load funds from %s: %s", self.storage_path, e)

    def _save(self):
        if self.storage_path:
            try:
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)
                data = list(self._funds.values())
                self.storage_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to save funds to %s: %s", self.storage_path, e)

    def create_fund(
        self,
        trip_id: Optional[str] = None,
        target_amount: float = 0.0,
        member_names: Optional[List[str]] = None,
        status: str = STATUS_FUNDRAISING,
    ) -> Dict[str, Any]:
        """Create a new trip fund."""
        trip_id = trip_id or str(uuid.uuid4())[:8]
        member_names = member_names or ["Anh", "Minh", "Lan", "Hung"]
        count = max(1, len(member_names))
        per_target = round(target_amount / count, 0) if target_amount > 0 else 0.0

        members = []
        for name in member_names:
            members.append({
                "name": name,
                "target": per_target,
                "paid": 0.0,
                "status": "pending",
            })

        fund = {
            "trip_id": trip_id,
            "target_amount": float(target_amount),
            "total_contributed": 0.0,
            "progress_percent": 0.0,
            "status": status if status in VALID_STATUSES else STATUS_FUNDRAISING,
            "members": members,
        }
        self._funds[trip_id] = fund
        self._save()
        return fund

    def get_fund(self, trip_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve fund by trip_id."""
        return self._funds.get(trip_id)

    def contribute(self, trip_id: str, member_name: str, amount: float) -> Dict[str, Any]:
        """
        Record a contribution from a member.
        Automatically transitions status to FULLY_FUNDED if total_contributed >= target_amount.
        """
        fund = self._funds.get(trip_id)
        if not fund:
            raise TripFundError(f"Trip fund with id '{trip_id}' not found.")

        if fund["status"] == STATUS_CONFIRMED:
            raise TripFundError("Cannot contribute to a trip that is already CONFIRMED.")

        if amount <= 0:
            raise TripFundError("Contribution amount must be greater than zero.")

        # Find or add member
        member = next((m for m in fund["members"] if m["name"].lower() == member_name.lower()), None)
        if not member:
            member = {
                "name": member_name,
                "target": round(fund["target_amount"] / max(1, len(fund["members"]) + 1), 0),
                "paid": 0.0,
                "status": "pending",
            }
            fund["members"].append(member)

        member["paid"] += float(amount)
        if member["paid"] >= member["target"]:
            member["status"] = "paid"
        elif member["paid"] > 0:
            member["status"] = "partial"

        # Update totals
        total_contributed = sum(m["paid"] for m in fund["members"])
        fund["total_contributed"] = total_contributed

        if fund["target_amount"] > 0:
            pct = min(100.0, round((total_contributed / fund["target_amount"]) * 100, 1))
            fund["progress_percent"] = pct
        else:
            fund["progress_percent"] = 100.0

        # Business rule check:
        # If total_contributed >= target_amount, status = FULLY_FUNDED
        if fund["total_contributed"] >= fund["target_amount"] and fund["target_amount"] > 0:
            if fund["status"] != STATUS_CONFIRMED:
                fund["status"] = STATUS_FULLY_FUNDED
        elif fund["status"] == STATUS_FULLY_FUNDED and fund["total_contributed"] < fund["target_amount"]:
            fund["status"] = STATUS_FUNDRAISING

        self._save()
        return fund

    def confirm_trip(self, trip_id: str) -> Dict[str, Any]:
        """
        Confirm a trip.
        Business rule: Can only confirm if status is FULLY_FUNDED (or already confirmed).
        """
        fund = self._funds.get(trip_id)
        if not fund:
            raise TripFundError(f"Trip fund with id '{trip_id}' not found.")

        if fund["status"] != STATUS_FULLY_FUNDED and fund["total_contributed"] < fund["target_amount"]:
            raise TripFundError(
                f"Cannot confirm trip. Target amount ({fund['target_amount']:,} VND) has not been reached. "
                f"Currently collected: {fund['total_contributed']:,} VND."
            )

        fund["status"] = STATUS_CONFIRMED
        self._save()
        return fund


# Global default instance
fund_manager = TripFundManager(storage_path="saved_funds.json")
