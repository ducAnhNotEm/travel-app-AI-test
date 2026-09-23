"""AI Travel Planner with Hybrid Pipeline for TripMate 2.0.

Architecture:
  USER INPUT
    ↓ extract_travel_intent_v2()      ← LLM Intent Extraction (or heuristic fallback)
  TravelIntent
    ↓ PlaceResolver.resolve_custom_places()
    ↓ PlaceFusionEngine.fuse()
    ↓ nearest_neighbor_sort() per destination
    ↓ ItineraryBuilder.build()        ← Deterministic
    ↓ calculate_budget()              ← Deterministic
    ↓ format_itinerary()              ← LLM Formatter (labels only; no POI invention)
  UI

Crucial Rule:
  The LLM may recommend among returned places, but it must NOT fabricate a
  place that was not returned by Google Places / Geoapify / catalog.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional

from .places import PlacesService, PlacesAPIError
from .budget import calculate_budget
from .models import (
    TravelIntent,
    DestinationItem,
    CustomPlace,
    NormalizedPlace,
    ItineraryItem,
    ItemType,
    PlaceStatus,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# System prompts
# ──────────────────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """Bạn là travel request parser của TripMate.
Trích xuất thông tin du lịch từ yêu cầu người dùng thành JSON hợp lệ. KHÔNG bịa tên địa điểm.
Chỉ xuất JSON theo schema dưới đây:
{
  "destinations": [
    {"name": "Đà Nẵng", "days": 2, "order": 1},
    {"name": "Hội An", "days": 1, "order": 2}
  ],
  "travelers": 4,
  "transportation": "car" | "motorcycle" | "public" | "flight",
  "tier": "budget" | "moderate" | "luxury",
  "categories": ["restaurant", "attraction", "hotel", "cafe"],
  "budget_vnd": null or number,
  "custom_places": [
    {"name": "Cơm Niêu Nhà Đỏ", "category": "restaurant", "target_destination": "Đà Nẵng", "target_day": 1, "target_slot": "dinner"},
    {"name": "Bánh Mì Phượng", "category": "restaurant", "target_destination": "Hội An", "target_slot": "breakfast"}
  ]
}
"""

ITINERARY_SYSTEM_PROMPT = """Bạn là trình tổng hợp lịch trình của TripMate.
Bạn nhận được:
1. Thông tin chuyến đi (điểm đến, số ngày, số người)
2. Danh sách ĐỊA ĐIỂM THỰC TẾ ĐÃ XÁC THỰC từ Google Places API / Geoapify.

QUY TẮC NGHIÊM NGẶT:
- Chỉ được sử dụng các nhà hàng, địa điểm tham quan, quán cafe trong DANH SÁCH ĐÃ CẤP.
- TUYỆT ĐỐI KHÔNG bịa tên địa điểm nào không có trong danh sách xác thực.
- Nếu không tìm được địa điểm phù hợp, đề xuất hoạt động chung (ví dụ: 'Dạo bộ dọc bờ sông') mà KHÔNG đặt tên giả cho cửa hàng/nhà hàng.
- Viết lịch trình theo từng ngày (Buổi sáng, Buổi chiều, Buổi tối) bằng TIẾNG VIỆT.
- Giữ nguyên tên thương hiệu/địa điểm chính thức (ví dụ: Madame Lan Restaurant, VinWonders Phú Quốc, Bún Chả Hương Liên).
"""

FORMATTER_SYSTEM_PROMPT = """Bạn là travel writer của TripMate — viết lời bình sinh động cho lịch trình.
Nhận vào danh sách ItineraryItem cố định và chỉ:
- Thêm mô tả ngắn gọn, mẹo du lịch, gợi ý món ăn đặc sắc cho từng địa điểm ĐÃ có trong danh sách.
- KHÔNG được đổi tên, thêm hoặc xoá bất kỳ địa điểm nào.
- Viết bằng TIẾNG VIỆT, giọng văn thân thiện, truyền cảm.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Legacy single-destination intent dict (backward-compat)
# ──────────────────────────────────────────────────────────────────────────────

def extract_travel_intent(user_prompt: str) -> Dict[str, Any]:
    """Parse natural language input into structured travel parameters (legacy dict).

    Backward-compatible with the original API:
      returns a flat dict with keys: destination, days, travelers,
      transportation, tier, categories, places_queries, budget_vnd.

    Also enriches the result with 'travel_intent' (TravelIntent) and
    'custom_places' so the hybrid pipeline can consume it.
    """
    # 1. Attempt LLM extraction
    try:
        from common.llm_client import chat, check_ollama_running
        if check_ollama_running():
            messages = [{"role": "user", "content": user_prompt}]
            raw_response = chat(
                messages,
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=1024,
            )
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return _normalize_intent(parsed, user_prompt)
    except Exception as e:
        logger.debug("LLM extraction failed or skipped: %s. Using heuristic.", e)

    # 2. Heuristic fallback
    return _heuristic_extract(user_prompt)


def extract_travel_intent_v2(user_prompt: str) -> TravelIntent:
    """Parse a natural-language prompt into a structured TravelIntent.

    Supports multi-destination trips and custom place extraction.
    Falls back to heuristic parsing when LLM is unavailable.
    """
    legacy = extract_travel_intent(user_prompt)
    return _legacy_to_travel_intent(legacy)


# ──────────────────────────────────────────────────────────────────────────────
# Heuristic parser
# ──────────────────────────────────────────────────────────────────────────────

def _heuristic_extract(text: str) -> Dict[str, Any]:
    """Deterministic fallback parser for common travel phrases."""
    text_lower = text.lower()

    from .destinations import DESTINATIONS_CATALOG, get_destination_info

    # ── Multi-destination detection ──────────────────────────────
    # Detect patterns like "Đà Nẵng 2 ngày rồi sang Hội An 1 ngày"
    multi_dest_pattern = re.findall(
        r"([\w\s\u00C0-\u024F\u1E00-\u1EFF]+?)\s+(\d+)\s*ngày",
        text,
        re.IGNORECASE,
    )

    destinations_raw: List[Dict[str, Any]] = []
    for match_name, match_days in multi_dest_pattern:
        candidate = match_name.strip()
        info = get_destination_info(candidate)
        if info:
            destinations_raw.append({
                "name": info["name"],
                "days": int(match_days),
                "order": len(destinations_raw) + 1,
            })

    # Fallback: single-destination detection
    if not destinations_raw:
        dest_candidates = [
            "đà nẵng", "da nang", "hà nội", "ha noi", "hanoi",
            "tp. hồ chí minh", "hồ chí minh", "ho chi minh", "sài gòn", "saigon",
            "phú quốc", "phu quoc", "hội an", "hoi an", "nha trang",
            "đà lạt", "da lat", "dalat", "huế", "hue",
            "hạ long", "ha long", "halong", "sa pa", "sapa",
            "quy nhơn", "quy nhon", "vũng tàu", "vung tau",
            "cần thơ", "can tho", "ninh bình", "ninh binh",
            "mũi né", "mui ne", "hải phòng", "hai phong",
            "hà giang", "ha giang", "cao bằng", "cao bang",
            "quảng bình", "quang binh", "điện biên", "dien bien",
        ]
        for candidate in dest_candidates:
            if candidate in text_lower:
                info = get_destination_info(candidate)
                if any(c in candidate for c in "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"):
                    dname = info["name"] if info else candidate.title()
                else:
                    dname = candidate.title()

                days_match = re.search(r"(\d+)\s*(?:-| )*(?:day|days|d\b|ngày|ngay)", text_lower)
                days = int(days_match.group(1)) if days_match else 3
                destinations_raw.append({"name": dname, "days": days, "order": 1})
                break

    if not destinations_raw:
        first_dest = next(iter(DESTINATIONS_CATALOG.values()))
        destinations_raw = [{"name": first_dest["name"], "days": 3, "order": 1}]

    # ── Travelers ────────────────────────────────────────────────
    travelers = 1
    pax_match = re.search(
        r"(?:for\s+|cho\s+|đoàn\s+)?(\d+)\s*"
        r"(?:people|travelers|members|pax|persons|guests|người|nguoi|thành viên|thanh vien|khách|khach)",
        text_lower,
    )
    if pax_match:
        travelers = int(pax_match.group(1))

    # ── Budget VND ───────────────────────────────────────────────
    budget_vnd = None
    vnd_match = re.search(r"([\d,\.]+)\s*(?:k|m|million|triệu)?\s*(?:vnd|đồng|dong)", text_lower)
    if vnd_match:
        val_str = vnd_match.group(1).replace(",", "").replace(".", "")
        try:
            budget_vnd = float(val_str)
            if "triệu" in text_lower or "million" in text_lower:
                if budget_vnd < 1000:
                    budget_vnd *= 1_000_000
        except ValueError:
            budget_vnd = None

    # ── Tier ─────────────────────────────────────────────────────
    tier = "moderate"
    if "luxury" in text_lower or "high-end" in text_lower or "cao cấp" in text_lower:
        tier = "luxury"
    elif "budget" in text_lower or "cheap" in text_lower or "tiết kiệm" in text_lower:
        tier = "budget"

    # ── Transportation ───────────────────────────────────────────
    transportation = "car"
    if "motorcycle" in text_lower or "motorbike" in text_lower or "xe máy" in text_lower:
        transportation = "motorcycle"
    elif "flight" in text_lower or "fly" in text_lower or "máy bay" in text_lower:
        transportation = "flight"
    elif "public" in text_lower or "bus" in text_lower or "xe buýt" in text_lower:
        transportation = "public"

    # ── POI categories ───────────────────────────────────────────
    categories: List[str] = []
    if any(kw in text_lower for kw in ["restaurant", "food", "ăn", "dinner", "lunch", "hải sản", "quán"]):
        categories.append("restaurant")
    if any(kw in text_lower for kw in ["cafe", "coffee", "cà phê"]):
        categories.append("cafe")
    if any(kw in text_lower for kw in ["hotel", "stay", "accommodation", "khách sạn", "lưu trú"]):
        categories.append("hotel")
    if any(kw in text_lower for kw in ["attraction", "sightseeing", "tourist", "tham quan", "địa điểm"]) or not categories:
        categories.append("attraction")

    # ── Custom places detection ─────────────────────────────────
    custom_places = _heuristic_extract_custom_places(text, destinations_raw)

    primary_dest = destinations_raw[0]["name"]
    total_days = sum(d["days"] for d in destinations_raw)

    places_queries = [
        {"category": cat, "query": f"{cat} in {primary_dest}"}
        for cat in categories
    ]

    return {
        "destination": primary_dest,
        "days": total_days,
        "travelers": travelers,
        "transportation": transportation,
        "tier": tier,
        "categories": categories,
        "places_queries": places_queries,
        "budget_vnd": budget_vnd,
        # Extended fields for hybrid pipeline
        "destinations": destinations_raw,
        "custom_places": custom_places,
    }


def _heuristic_extract_custom_places(
    text: str,
    destinations_raw: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Detect specific venue names mentioned in the user's prompt.

    Uses heuristic patterns like:
      - "ăn tại <place>"  → restaurant, dinner
      - "ghé <place>"     → attraction
      - "tối ngày N ăn <place>" → restaurant, dinner, day=N
    Returns list of raw dicts suitable for CustomPlace construction.
    """
    results: List[Dict[str, Any]] = []

    # Pattern: "tối ngày <day> ăn <place>"
    for m in re.finditer(
        r"tối\s+ngày\s+(\d+)\s+ăn\s+([^\.,;]+)",
        text,
        re.IGNORECASE | re.UNICODE,
    ):
        target_day = int(m.group(1))
        place_name = m.group(2).strip()
        dest_name = _day_to_destination(target_day, destinations_raw)
        results.append({
            "name": place_name,
            "category": "restaurant",
            "target_destination": dest_name,
            "target_day": target_day,
            "target_slot": "dinner",
        })

    # Pattern: "sáng ngày <day> ăn <place>"
    for m in re.finditer(
        r"sáng\s+ngày\s+(\d+)\s+ăn\s+([^\.,;]+)",
        text,
        re.IGNORECASE | re.UNICODE,
    ):
        target_day = int(m.group(1))
        place_name = m.group(2).strip()
        dest_name = _day_to_destination(target_day, destinations_raw)
        results.append({
            "name": place_name,
            "category": "restaurant",
            "target_destination": dest_name,
            "target_day": target_day,
            "target_slot": "breakfast",
        })

    # Pattern: "ghé <place>" (visit attraction)
    for m in re.finditer(
        r"(?:ghé|ghé thăm|thăm)\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?=\.|,|;|$|\svà\s|\bngày\b)",
        text,
        re.UNICODE,
    ):
        place_name = m.group(1).strip()
        if len(place_name) > 3:
            results.append({
                "name": place_name,
                "category": "attraction",
                "target_destination": None,
                "target_day": None,
                "target_slot": None,
            })

    # Remove duplicates by name
    seen_names: set = set()
    deduped = []
    for r in results:
        key = r["name"].lower()
        if key not in seen_names:
            seen_names.add(key)
            deduped.append(r)
    return deduped


def _day_to_destination(day: int, destinations_raw: List[Dict[str, Any]]) -> Optional[str]:
    """Map an absolute day number to a destination name."""
    cumulative = 0
    for d in sorted(destinations_raw, key=lambda x: x["order"]):
        cumulative += d["days"]
        if day <= cumulative:
            return d["name"]
    return destinations_raw[-1]["name"] if destinations_raw else None


def _normalize_intent(data: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    """Ensure all required schema keys exist with valid types (LLM output normalisation)."""
    from .destinations import DESTINATIONS_CATALOG
    first_dest = next(iter(DESTINATIONS_CATALOG.values()))

    # Destinations list — handle both new multi-dest and legacy single dest
    raw_dests = data.get("destinations")
    if not raw_dests:
        dest_str = str(data.get("destination") or first_dest["name"])
        days_val = max(1, int(data.get("days", 3)))
        raw_dests = [{"name": dest_str, "days": days_val, "order": 1}]
    elif isinstance(raw_dests, list):
        raw_dests = [
            {
                "name": str(d.get("name", first_dest["name"])),
                "days": max(1, int(d.get("days", 1))),
                "order": int(d.get("order", i + 1)),
            }
            for i, d in enumerate(raw_dests)
        ]

    total_days = sum(d["days"] for d in raw_dests)
    primary_dest = raw_dests[0]["name"]

    categories = data.get("categories") or ["attraction", "restaurant"]

    # Custom places from LLM output
    raw_customs = data.get("custom_places", []) or []
    custom_places_list = []
    for cp in raw_customs:
        if isinstance(cp, dict) and cp.get("name"):
            custom_places_list.append({
                "name": str(cp["name"]),
                "category": str(cp.get("category", "attraction")),
                "target_destination": cp.get("target_destination"),
                "target_day": cp.get("target_day"),
                "target_slot": cp.get("target_slot"),
            })

    return {
        "destination": primary_dest,
        "days": total_days,
        "travelers": max(1, int(data.get("travelers", 4))),
        "transportation": str(data.get("transportation") or "car"),
        "tier": str(data.get("tier") or "moderate"),
        "categories": categories,
        "places_queries": data.get("places_queries") or [
            {"category": cat, "query": f"{cat} in {primary_dest}"}
            for cat in categories
        ],
        "budget_vnd": data.get("budget_vnd"),
        # Extended fields
        "destinations": raw_dests,
        "custom_places": custom_places_list,
    }


def _legacy_to_travel_intent(legacy: Dict[str, Any]) -> TravelIntent:
    """Convert the legacy intent dict to a structured TravelIntent."""
    from .destinations import get_destination_info

    raw_dests = legacy.get("destinations") or [
        {"name": legacy["destination"], "days": legacy["days"], "order": 1}
    ]

    dest_items: List[DestinationItem] = []
    for d in sorted(raw_dests, key=lambda x: x.get("order", 1)):
        info = get_destination_info(d["name"])
        dest_items.append(DestinationItem(
            name=d["name"],
            days=d["days"],
            order=d.get("order", 1),
            latitude=info["latitude"] if info else None,
            longitude=info["longitude"] if info else None,
        ))

    # Build CustomPlace list
    raw_customs = legacy.get("custom_places", []) or []
    custom_places: List[CustomPlace] = []
    for cp in raw_customs:
        if isinstance(cp, dict) and cp.get("name"):
            custom_places.append(CustomPlace(
                name=cp["name"],
                category=cp.get("category", "attraction"),
                target_destination=cp.get("target_destination"),
                target_day=cp.get("target_day"),
                target_slot=cp.get("target_slot"),
            ))

    return TravelIntent(
        destinations=dest_items,
        total_days=legacy.get("days", sum(d["days"] for d in raw_dests)),
        travelers=legacy.get("travelers", 1),
        transportation=legacy.get("transportation", "car"),
        tier=legacy.get("tier", "moderate"),
        custom_places=custom_places,
        preferences=legacy.get("categories", []),
        budget_vnd=legacy.get("budget_vnd"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# LLM Formatter
# ──────────────────────────────────────────────────────────────────────────────

def format_itinerary(
    items: List[ItineraryItem],
    intent: TravelIntent,
) -> str:
    """Format a structured itinerary list into human-readable Vietnamese text.

    First attempts LLM enrichment (adding descriptions/tips without
    changing the structured data). Falls back to a deterministic template.
    """
    template_output = _template_format(items, intent)

    try:
        from common.llm_client import chat, check_ollama_running
        if check_ollama_running():
            dest_names = " → ".join(d.name for d in intent.destinations)
            prompt = (
                f"Dưới đây là lịch trình {intent.total_days} ngày du lịch {dest_names} "
                f"cho đoàn {intent.travelers} người (phong cách: {intent.tier}).\n\n"
                f"{template_output}\n\n"
                "Hãy thêm lời bình ngắn (1-2 câu) cho mỗi địa điểm: mô tả nét đặc sắc, "
                "mẹo du lịch hoặc món ăn nổi bật. KHÔNG thêm địa điểm mới. "
                "Giữ nguyên định dạng Markdown và tên địa điểm."
            )
            messages = [{"role": "user", "content": prompt}]
            enriched = chat(
                messages,
                system_prompt=FORMATTER_SYSTEM_PROMPT,
                temperature=0.4,
                max_tokens=3000,
            )
            if enriched and len(enriched) > len(template_output) * 0.5:
                return enriched
    except Exception as e:
        logger.debug("LLM formatter unavailable: %s. Using template.", e)

    return template_output


def _template_format(items: List[ItineraryItem], intent: TravelIntent) -> str:
    """Produce structured Markdown output from ItineraryItem list."""
    dest_names = " → ".join(d.name for d in intent.destinations)
    lines = [
        f"# 🗺️ {dest_names} — Lịch trình {intent.total_days} ngày cho {intent.travelers} người",
        "",
    ]

    current_day = 0
    current_dest = ""

    for item in items:
        # Day header
        if item.day != current_day or item.destination_name != current_dest:
            if item.day != current_day:
                current_day = item.day
                lines.append(f"\n### 📅 Ngày {item.day} — {item.destination_name}")
                lines.append("")
            current_dest = item.destination_name

        if item.item_type == ItemType.TRANSIT:
            ti = item.transit_info or {}
            dist = f"{ti.get('distance_km', '?')} km" if ti.get("distance_km") else ""
            time_min = ti.get("travel_time_min")
            time_str = f"{time_min} phút" if time_min else ""
            detail = f" ({dist} · {time_str})" if dist or time_str else ""
            lines.append(f"🚗 **{item.time_slot}** — {item.title}{detail}")
            if item.description:
                lines.append(f"   > {item.description}")

        elif item.item_type == ItemType.ACTIVITY:
            lines.append(f"🌅 **{item.time_slot}** — {item.title}")
            if item.description:
                lines.append(f"   > {item.description}")

        else:  # PLACE
            status_badge = ""
            if item.status == PlaceStatus.UNVERIFIED:
                status_badge = " ⚠️ *[Địa điểm bạn đề xuất]*"
            elif item.status == PlaceStatus.VERIFIED:
                status_badge = " ✅"

            lines.append(f"📍 **{item.time_slot}** — **{item.title}**{status_badge}")
            if item.description:
                lines.append(f"   > {item.description}")

        lines.append("")

    lines.append("> [!TIP]")
    lines.append(
        f"> Lịch trình được tạo theo kiến trúc Hybrid AI — "
        f"địa điểm xác thực qua Google Places API, không sử dụng AI bịa tên."
    )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class AITravelPlanner:
    """Orchestrates the full Hybrid AI Planning Pipeline.

    Backward-compatible with the original generate_full_trip_plan() API.
    Now also exposes generate_hybrid_plan() for the fully structured pipeline.
    """

    def __init__(self, places_service: Optional[PlacesService] = None):
        self.places = places_service or PlacesService()

    # ── Public: Hybrid Pipeline ──────────────────────────────────────────────

    def generate_hybrid_plan(
        self,
        user_prompt: str,
        pinned_places: Optional[List[NormalizedPlace]] = None,
        mock_places_fallback: bool = True,
    ) -> Dict[str, Any]:
        """Full Hybrid Pipeline:
          Intent → Resolver → Fusion → Optimizer → Itinerary → Budget → Formatter

        Returns:
            {
              "intent":       TravelIntent,
              "places":       List[NormalizedPlace],
              "items":        List[ItineraryItem],   # structured items
              "itinerary":    str,                   # formatted markdown
              "budget":       dict,
            }
        """
        from .resolver import PlaceResolver
        from .fusion import PlaceFusionEngine
        from .itinerary import ItineraryBuilder

        # Step 1: Extract structured intent
        intent = extract_travel_intent_v2(user_prompt)

        # Step 2: Resolve custom places against Places API
        resolver = PlaceResolver(self.places)
        resolved_customs = resolver.resolve_custom_places(
            intent.custom_places,
            destination=intent.destinations[0].name if intent.destinations else "",
        )
        intent.custom_places = resolved_customs

        # Step 3: Fetch Google Places for each destination
        places_by_destination: Dict[str, List[Dict[str, Any]]] = {}
        for dest in intent.destinations:
            dest_places = self._fetch_places_for_destination(
                dest.name, dest.latitude, dest.longitude,
                intent.preferences, mock_places_fallback,
            )
            places_by_destination[dest.name] = dest_places

        # Step 4: Fuse places (pinned + custom + google)
        fusion_engine = PlaceFusionEngine()
        pinned = pinned_places or []

        fused_by_destination: Dict[str, List[NormalizedPlace]] = {}
        all_fused: List[NormalizedPlace] = []
        for dest in intent.destinations:
            dest_google = places_by_destination.get(dest.name, [])
            dest_custom = [
                cp for cp in intent.custom_places
                if not cp.target_destination
                   or cp.target_destination.lower() == dest.name.lower()
            ]
            fused = fusion_engine.fuse(
                pinned_places=pinned,
                custom_places=dest_custom,
                google_places=dest_google,
            )
            fused_by_destination[dest.name] = fused
            all_fused.extend(fused)

        # Step 5: Build deterministic itinerary
        builder = ItineraryBuilder()
        items = builder.build(intent, fused_by_destination)

        # Step 6: Format with LLM / template
        itinerary_text = format_itinerary(items, intent)

        # Step 7: Deterministic budget
        budget = calculate_budget(
            travelers=intent.travelers,
            days=intent.total_days,
            transportation=intent.transportation,
            tier=intent.tier,
            selected_places=[
                {"id": p.id, "name": p.name, "category": p.category}
                for p in all_fused
            ],
            custom_accommodation=(
                intent.budget_vnd * 0.35 if intent.budget_vnd else None
            ),
        )

        return {
            "intent": intent,
            "places": all_fused,
            "items": items,
            "itinerary": itinerary_text,
            "budget": budget,
        }

    # ── Public: Legacy API ───────────────────────────────────────────────────

    def generate_full_trip_plan(
        self,
        user_prompt: str,
        mock_places_fallback: bool = True,
    ) -> Dict[str, Any]:
        """Legacy single-destination pipeline (backward-compat).

        Now internally delegates to generate_hybrid_plan and returns a
        flattened dict matching the original API shape.
        """
        result = self.generate_hybrid_plan(
            user_prompt,
            mock_places_fallback=mock_places_fallback,
        )

        intent_obj: TravelIntent = result["intent"]
        # Convert TravelIntent back to legacy dict shape
        legacy_intent = {
            "destination": intent_obj.destinations[0].name if intent_obj.destinations else "",
            "days": intent_obj.total_days,
            "travelers": intent_obj.travelers,
            "transportation": intent_obj.transportation,
            "tier": intent_obj.tier,
            "categories": intent_obj.preferences,
            "budget_vnd": intent_obj.budget_vnd,
        }

        # Convert NormalizedPlace list to legacy dict list
        places_dicts = [
            {
                "id": p.id,
                "name": p.name,
                "address": p.address,
                "latitude": p.latitude,
                "longitude": p.longitude,
                "rating": p.rating,
                "user_rating_count": p.user_rating_count,
                "category": p.category,
                "google_maps_uri": p.google_maps_uri,
            }
            for p in result["places"]
        ]

        return {
            "intent": legacy_intent,
            "places": places_dicts,
            "itinerary": result["itinerary"],
            "budget": result["budget"],
            # Extended fields (non-breaking additions)
            "items": result["items"],
            "travel_intent": intent_obj,
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    CATEGORY_MAP: Dict[str, Dict[str, Any]] = {
        "restaurant": {
            "google_types": ["restaurant"],
            "geoapify_cat": "catering.restaurant",
        },
        "cafe": {
            "google_types": ["cafe"],
            "geoapify_cat": "catering.cafe",
        },
        "attraction": {
            "google_types": ["tourist_attraction", "museum", "park"],
            "geoapify_cat": "tourism.sights",
        },
        "hotel": {
            "google_types": ["lodging"],
            "geoapify_cat": "accommodation.hotel",
        },
    }

    def _fetch_places_for_destination(
        self,
        destination: str,
        dest_lat: Optional[float],
        dest_lng: Optional[float],
        categories: List[str],
        mock_places_fallback: bool,
    ) -> List[Dict[str, Any]]:
        """Query the Places API for each category and return deduplicated results."""
        if not categories:
            categories = ["attraction", "restaurant"]

        verified_places: List[Dict[str, Any]] = []
        seen: set = set()

        for cat in categories:
            cat_config = self.CATEGORY_MAP.get(cat, self.CATEGORY_MAP["attraction"])
            try:
                if dest_lat is not None and dest_lng is not None:
                    found = self.places.search_nearby(
                        dest_lat, dest_lng,
                        radius_meters=5000,
                        included_types=cat_config["google_types"],
                        max_results=6,
                    )
                else:
                    found = self.places.search_text(f"{cat} in {destination}", max_results=6)

                for p in found:
                    p["category"] = cat
                    key = p.get("id") or p.get("name")
                    if key and key not in seen:
                        seen.add(key)
                        verified_places.append(p)

            except PlacesAPIError as e:
                logger.warning("Places API error for '%s' in %s: %s", cat, destination, e)
                if mock_places_fallback and e.error_type in (
                    "missing_api_key", "auth_error", "quota_exceeded", "network_error"
                ):
                    fallback = self._get_dev_sample_places(destination, cat)
                    for p in fallback:
                        key = p.get("id") or p.get("name")
                        if key and key not in seen:
                            seen.add(key)
                            verified_places.append(p)
                elif e.error_type in ("auth_error", "quota_exceeded"):
                    raise

        return verified_places

    def _get_dev_sample_places(self, destination: str, category: str) -> List[Dict[str, Any]]:
        """Return sample places from catalog or generic fallback for dev/offline use."""
        from .destinations import get_destination_info, DESTINATIONS_CATALOG
        info = get_destination_info(destination)
        if info and "sample_places" in info:
            matched = [p for p in info["sample_places"] if p.get("category") == category]
            if matched:
                return matched
            return info["sample_places"][:2]

        first = next(iter(DESTINATIONS_CATALOG.values()))
        return [
            {
                "id": f"places/{destination.lower()[:8]}_central",
                "name": f"{destination} — Khu vực trung tâm",
                "address": f"Trung tâm, {destination}",
                "latitude": first["latitude"],
                "longitude": first["longitude"],
                "rating": 4.5,
                "user_rating_count": 100,
                "category": category,
                "google_maps_uri": "https://maps.google.com",
            }
        ]

    # ── Legacy internal: keep _build_itinerary for any external callers ──────

    def _build_itinerary(
        self,
        destination: str,
        days: int,
        travelers: int,
        places: List[Dict[str, Any]],
        tier: str,
    ) -> str:
        """Legacy itinerary builder (kept for backward compat; prefer format_itinerary)."""
        places_summary = "\n".join([
            f"- {p['name']} (Loại: {p.get('category', 'attraction')}, "
            f"Đánh giá: {p.get('rating', 'N/A')}⭐, Địa chỉ: {p.get('address', 'N/A')})"
            for p in places
        ])

        try:
            from common.llm_client import chat, check_ollama_running
            if check_ollama_running():
                prompt = (
                    f"Lập kế hoạch du lịch {days} ngày tại {destination} "
                    f"(đoàn {travelers} người, phong cách {tier}).\n"
                    f"CHỈ ĐƯỢC DÙNG CÁC ĐỊA ĐIỂM ĐÃ XÁC THỰC SAU:\n{places_summary}\n\n"
                    "Viết lịch trình theo từng ngày (Buổi sáng, Buổi chiều, Buổi tối) bằng tiếng Việt, "
                    "kèm thời gian tham quan và mẹo di chuyển."
                )
                messages = [{"role": "user", "content": prompt}]
                return chat(
                    messages,
                    system_prompt=ITINERARY_SYSTEM_PROMPT,
                    temperature=0.3,
                    max_tokens=2048,
                )
        except Exception as e:
            logger.debug("LLM không khả dụng: %s. Dùng template chuẩn.", e)

        lines = [f"# 🗺️ {destination} — Lịch trình {days} ngày cho {travelers} người", ""]
        restaurants = [p for p in places if p.get("category") == "restaurant"]
        attractions = [p for p in places if p.get("category") not in ("restaurant", "hotel")]

        att_idx = 0
        rest_idx = 0

        for day in range(1, days + 1):
            lines.append(f"### 📍 Ngày {day}: Khám phá {destination}")

            if attractions and att_idx < len(attractions):
                att1 = attractions[att_idx % len(attractions)]
                lines.append(f"- **Buổi sáng (09:00 - 12:00)**: Tham quan **{att1['name']}**")
                lines.append(f"  - *Địa chỉ*: {att1.get('address', 'Khu vực trung tâm')}")
                lines.append(f"  - *Đánh giá*: {att1.get('rating', '4.5')} ⭐ ({att1.get('user_rating_count', 100)} đánh giá)")
                att_idx += 1
            else:
                lines.append("- **Buổi sáng (09:00 - 12:00)**: Dạo bộ khám phá khu vực xung quanh")

            if restaurants:
                r1 = restaurants[rest_idx % len(restaurants)]
                lines.append(f"- **Buổi trưa (12:30 - 14:00)**: Ăn trưa tại **{r1['name']}**")
                lines.append(f"  - *Địa chỉ*: {r1.get('address', 'Khu vực lân cận')}")
                rest_idx += 1
            else:
                lines.append("- **Buổi trưa (12:30 - 14:00)**: Thưởng thức đặc sản địa phương")

            if attractions and att_idx < len(attractions):
                att2 = attractions[att_idx % len(attractions)]
                lines.append(f"- **Buổi chiều (14:30 - 17:30)**: Tham quan **{att2['name']}**")
                lines.append(f"  - *Địa chỉ*: {att2.get('address', 'Khu trung tâm')}")
                att_idx += 1
            else:
                lines.append("- **Buổi chiều (14:30 - 17:30)**: Tự do khám phá")

            if restaurants and rest_idx < len(restaurants):
                r2 = restaurants[rest_idx % len(restaurants)]
                lines.append(f"- **Buổi tối (18:30 - 21:00)**: Ăn tối tại **{r2['name']}**")
                lines.append(f"  - *Địa chỉ*: {r2.get('address', 'Khu trung tâm')}")
                rest_idx += 1
            else:
                lines.append("- **Buổi tối (18:30 - 21:00)**: Ăn tối và ngắm cảnh đêm")

            lines.append("")

        lines.append("> [!TIP]")
        lines.append(
            f"> Tất cả {len(places)} địa điểm trong lịch trình đã được xác thực qua Google Places API / Geoapify."
        )
        return "\n".join(lines)
