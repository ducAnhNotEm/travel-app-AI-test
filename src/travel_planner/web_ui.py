"""Streamlit web interface for TripMate 2.0 — Hybrid AI Travel Planner."""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.travel_planner.config import load_config, setup_logging
from src.travel_planner.core import (
    check_ollama_running,
    generate_itinerary,
    generate_multi_destination_itinerary,
    get_place_details,
    generate_budget_breakdown,
    generate_packing_list,
    BUDGETS,
)
from src.travel_planner.utils import parse_destinations, parse_budget_items, save_itinerary, load_saved_itineraries
from src.travel_planner.places import PlacesService, PlacesAPIError
from src.travel_planner.ai_planner import AITravelPlanner, extract_travel_intent
from src.travel_planner.budget import calculate_budget
from src.travel_planner.fund import fund_manager, STATUS_FULLY_FUNDED, STATUS_CONFIRMED
from src.travel_planner.models import NormalizedPlace, ItemType, PlaceStatus

# Page configuration
st.set_page_config(
    page_title="TripMate — Group Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SaaS styling adhering to requested color palette:
# Primary red: #C62828 | Warm yellow: #F4C542 | Teal: #168F8B | Dark navy: #172033 | Background: #FAF9F6
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #FAF9F6;
        color: #172033;
    }
    
    /* Header Brand */
    .brand-title {
        color: #C62828;
        font-weight: 700;
        font-size: 2.2rem;
        letter-spacing: -0.5px;
        margin-bottom: 0px;
    }
    .brand-sub {
        color: #555e6d;
        font-size: 1rem;
        margin-top: -5px;
        margin-bottom: 25px;
    }
    
    /* Clean Cards */
    .saas-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(23, 32, 51, 0.05);
        margin-bottom: 20px;
    }
    
    /* Metric / Stat badge */
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-teal { background-color: #E0F2F1; color: #168F8B; }
    .badge-yellow { background-color: #FEF9E7; color: #B78103; }
    .badge-red { background-color: #FFEBEE; color: #C62828; }
    
    /* Verified / Unverified badges */
    .badge-verified {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        background-color: #E8F5E9;
        color: #2E7D32;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-unverified {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        background-color: #FFF8E1;
        color: #F57F17;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-transit {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        background-color: #E3F2FD;
        color: #1565C0;
        font-size: 0.78rem;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #C62828;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.6rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #B71C1C;
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(198, 40, 40, 0.25);
    }
    
    /* Custom tab headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        color: #555e6d;
    }
    .stTabs [aria-selected="true"] {
        color: #C62828 !important;
        border-bottom-color: #C62828 !important;
    }
    
    /* Progress bar coloring */
    .stProgress > div > div > div > div {
        background-color: #168F8B;
    }
</style>
""", unsafe_allow_html=True)


def init_state():
    from src.travel_planner.destinations import DESTINATIONS_CATALOG
    first_dest_key = next(iter(DESTINATIONS_CATALOG.keys()))
    defaults = {
        "config": load_config(),
        "plan_result": None,
        "hybrid_result": None,
        "active_trip_id": None,
        "trip_name": "",
        "destination": first_dest_key,
        "travelers": 4,
        "days": 3,
        "fund": None,
        "places_service": PlacesService(),
        # Multi-destination
        "multi_destinations": [],
        # Pinned places from UI explorer
        "pinned_places": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    init_state()
    cfg = st.session_state.config
    setup_logging(cfg)

    # Top Brand Bar
    st.markdown('<div class="brand-title">TripMate 2.0</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">Hybrid AI Travel Planner — Google Places · Deterministic Scheduling · Anti-Hallucination</div>',
        unsafe_allow_html=True,
    )

    # Sidebar: Trip Settings & Keys
    with st.sidebar:
        st.markdown("### ⚙️ TripMate Settings")
        st.caption("Google Places & AI Configuration")

        import os
        has_google_key = bool(os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("GOOGLE_PLACES_API_KEY"))
        has_geoapify_key = bool(os.environ.get("GEOAPIFY_API_KEY"))

        if has_google_key:
            st.success("✓ Google Places API key active")
        elif has_geoapify_key:
            st.info("✓ Geoapify API key active (Fallback)")
        else:
            st.warning("⚠️ No Places API key set. Using verified cached places fallback.")

        key_input = st.text_input("Enter/Override GOOGLE_MAPS_API_KEY", type="password")
        if key_input:
            os.environ["GOOGLE_MAPS_API_KEY"] = key_input
            st.session_state.places_service = PlacesService(api_key=key_input)
            st.rerun()

        geo_input = st.text_input("Enter/Override GEOAPIFY_API_KEY", type="password")
        if geo_input:
            os.environ["GEOAPIFY_API_KEY"] = geo_input
            st.session_state.places_service = PlacesService(geoapify_key=geo_input)
            st.rerun()

        st.markdown("---")
        st.markdown("### 👥 Thông tin Chuyến đi")
        st.text(f"Trip ID: {st.session_state.active_trip_id or 'Chưa tạo chuyến đi'}")
        st.text(f"Tên chuyến: {st.session_state.trip_name or 'Chưa đặt tên'}")

        # Pinned places count
        pinned = st.session_state.pinned_places
        if pinned:
            st.markdown(f"📌 **{len(pinned)} địa điểm đã ghim**")
            if st.button("🗑️ Xoá tất cả địa điểm ghim"):
                st.session_state.pinned_places = []
                st.rerun()

    # Main tabs
    tab_planner, tab_places, tab_fund = st.tabs([
        "🚀 AI Trip Planner",
        "📍 Google Places Explorer",
        "💳 Group Trip Fund Manager",
    ])

    from src.travel_planner.destinations import DESTINATIONS_CATALOG, get_destination_info

    dest_options = list(DESTINATIONS_CATALOG.keys()) + ["Other (Custom Destination)..."]

    # ─────────────────────────────────────────────────────────────
    # TAB 1: AI TRIP PLANNER
    # ─────────────────────────────────────────────────────────────
    with tab_planner:
        col_input, col_meta = st.columns([5, 3])

        with col_input:
            st.markdown("#### 1. Chọn điểm đến & Cấu hình chuyến đi")

            # ── Multi-destination builder ──────────────────────
            st.markdown("**Thêm nhiều điểm đến (Multi-destination):**")
            md_col1, md_col2, md_col3 = st.columns([3, 1, 1])
            new_dest_key = md_col1.selectbox(
                "Thêm điểm đến:",
                dest_options,
                index=0,
                key="new_dest_select",
            )
            new_dest_days = md_col2.number_input("Số ngày:", min_value=1, max_value=14, value=2, key="new_dest_days")

            if md_col3.button("➕ Thêm"):
                if new_dest_key == "Other (Custom Destination)...":
                    st.warning("Vui lòng chọn điểm đến từ danh sách.")
                else:
                    existing = [d["name"] for d in st.session_state.multi_destinations]
                    dest_info = DESTINATIONS_CATALOG[new_dest_key]
                    if dest_info["name"] not in existing:
                        st.session_state.multi_destinations.append({
                            "name": dest_info["name"],
                            "days": int(new_dest_days),
                            "order": len(st.session_state.multi_destinations) + 1,
                        })
                        st.rerun()

            # Show multi-destination list
            if st.session_state.multi_destinations:
                st.markdown("**Hành trình đã chọn:**")
                for idx, d in enumerate(st.session_state.multi_destinations):
                    dc1, dc2, dc3 = st.columns([4, 1, 1])
                    dc1.write(f"📍 **{d['name']}** — {d['days']} ngày")
                    if dc3.button("❌", key=f"rm_dest_{idx}"):
                        st.session_state.multi_destinations.pop(idx)
                        # Re-number orders
                        for i, item in enumerate(st.session_state.multi_destinations):
                            item["order"] = i + 1
                        st.rerun()
            else:
                st.info("Chưa có điểm đến nào. Thêm ít nhất 1 điểm đến để tiếp tục.")

            st.markdown("---")

            # ── Traveler / style settings ──────────────────────
            col_d1, col_d2 = st.columns(2)
            travelers_count = col_d1.slider("Số lượng thành viên:", min_value=1, max_value=20, value=4)

            # Computed total days
            total_days_computed = sum(d["days"] for d in st.session_state.multi_destinations) or 3

            col_s1, col_s2 = st.columns(2)
            tier_choice = col_s1.radio(
                "Phân khúc ngân sách:",
                ["budget", "moderate", "luxury"],
                index=1,
                format_func=lambda x: {
                    "budget": "Tiết kiệm 💸",
                    "moderate": "Tiêu chuẩn ⚖️",
                    "luxury": "Cao cấp 💎",
                }[x],
                horizontal=True,
            )
            trans_choice = col_s2.radio(
                "Phương tiện di chuyển:",
                ["car", "motorcycle", "public", "flight"],
                index=0,
                format_func=lambda x: {
                    "car": "🚗 Ô tô",
                    "motorcycle": "🛵 Xe máy",
                    "public": "🚌 Xe buýt",
                    "flight": "✈️ Máy bay",
                }[x],
                horizontal=True,
            )

            st.markdown("#### 2. Yêu cầu chi tiết gửi AI:")
            if st.session_state.multi_destinations:
                dest_strs = " rồi sang ".join(
                    f"{d['name']} {d['days']} ngày" for d in st.session_state.multi_destinations
                )
                generated_prompt = (
                    f"Tôi muốn đi {dest_strs} cho {travelers_count} người, "
                    f"phong cách {tier_choice}, di chuyển bằng {trans_choice}. "
                    "Tìm nhà hàng ngon và địa điểm tham quan thực tế kèm dự toán chi phí."
                )
            else:
                generated_prompt = (
                    f"Lập kế hoạch du lịch 3 ngày cho {travelers_count} người, "
                    f"phong cách {tier_choice}."
                )

            user_prompt = st.text_area(
                "Prompt mô tả kế hoạch du lịch:",
                value=generated_prompt,
                height=110,
                help="TripMate phân tích prompt, truy vấn Google Places API và tạo lịch trình xác định.",
            )

            # Pinned places summary
            if st.session_state.pinned_places:
                st.markdown(f"📌 **{len(st.session_state.pinned_places)} địa điểm đã ghim** sẽ được ưu tiên trong lịch trình.")

            if st.button("🚀 Khởi tạo Lịch trình & Ngân sách", type="primary"):
                if not st.session_state.multi_destinations:
                    st.error("Vui lòng thêm ít nhất 1 điểm đến trước khi tạo lịch trình.")
                else:
                    planner = AITravelPlanner(places_service=st.session_state.places_service)
                    with st.spinner("AI đang phân tích yêu cầu & truy vấn Places API..."):
                        try:
                            pinned = st.session_state.pinned_places
                            plan_res = planner.generate_hybrid_plan(
                                user_prompt,
                                pinned_places=pinned,
                            )
                            st.session_state.hybrid_result = plan_res
                            # Also set legacy plan_result for compatibility
                            intent_obj = plan_res["intent"]
                            st.session_state.plan_result = {
                                "intent": {
                                    "destination": intent_obj.destinations[0].name if intent_obj.destinations else "",
                                    "days": intent_obj.total_days,
                                    "travelers": intent_obj.travelers,
                                    "transportation": intent_obj.transportation,
                                    "tier": intent_obj.tier,
                                    "budget_vnd": intent_obj.budget_vnd,
                                },
                                "places": [
                                    {
                                        "id": p.id, "name": p.name, "address": p.address,
                                        "latitude": p.latitude, "longitude": p.longitude,
                                        "rating": p.rating, "user_rating_count": p.user_rating_count,
                                        "category": p.category, "google_maps_uri": p.google_maps_uri,
                                    }
                                    for p in plan_res["places"]
                                ],
                                "itinerary": plan_res["itinerary"],
                                "budget": plan_res["budget"],
                            }
                            st.session_state.travelers = travelers_count

                            import re as _re
                            primary_name = intent_obj.destinations[0].name if intent_obj.destinations else "trip"
                            safe_dest = _re.sub(r"[^a-z0-9]", "-", primary_name.lower())[:16].strip("-")
                            trip_id = f"{safe_dest}-2026"
                            st.session_state.active_trip_id = trip_id
                            st.session_state.trip_name = f"Chuyến du lịch {primary_name}"

                            total_budget = plan_res["budget"]["total"]
                            fund_manager.create_fund(
                                trip_id=trip_id,
                                target_amount=total_budget,
                                member_names=(
                                    [f"Thành viên {i+1}" for i in range(travelers_count)]
                                    or ["Thành viên 1"]
                                ),
                            )
                            st.session_state.fund = fund_manager.get_fund(trip_id)
                            st.success("✓ Đã tạo thành công lịch trình với địa điểm thật & dự toán ngân sách!")
                        except Exception as e:
                            st.error(f"Lỗi khởi tạo lịch trình: {e}")

        with col_meta:
            st.markdown("#### ⚡ Kiến trúc Hybrid AI Pipeline")
            st.markdown("""
            ```
            User Prompt
              ↓ Intent Extraction (LLM / Heuristic)
            TravelIntent
              ↓ PlaceResolver → Google Places
            CustomPlace (Verified/Unverified)
              ↓ PlaceFusionEngine
            Pinned + Custom + Google
              ↓ Nearest-Neighbor Optimizer
            Optimised POI Pool
              ↓ ConstraintEngine → Slot Assignment
            ItineraryItem list (PLACE/ACTIVITY/TRANSIT)
              ↓ Budget Engine + LLM Formatter
            UI
            ```
            """)
            st.markdown("""
            > [!TIP]
            > **Anti-Hallucination**: LLM chỉ được thêm lời bình — không được đổi tên hay bịa địa điểm.
            """)

        # ── Display Result ─────────────────────────────────────────────
        if st.session_state.hybrid_result:
            result = st.session_state.hybrid_result
            intent_obj = result["intent"]
            items = result.get("items", [])

            st.markdown("---")

            # Overview Metrics
            dest_names_str = " → ".join(d.name for d in intent_obj.destinations)
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Hành trình", dest_names_str[:30])
            mcol2.metric("Thời gian / Pax", f"{intent_obj.total_days} ngày / {intent_obj.travelers} người")
            mcol3.metric("Địa điểm xác thực", f"{len(result['places'])}")
            mcol4.metric("Dự toán tổng", f"{int(result['budget']['total']):,} VND")

            col_itin, col_bud = st.columns([3, 2])

            with col_itin:
                # ── Structured itinerary rendering with item-type badges ──
                st.markdown("### 🗺️ Lịch trình Chi tiết")

                if items:
                    _render_structured_itinerary(items)
                else:
                    st.markdown(result["itinerary"])

                # ── Verified places accordion ─────────────────────────────
                st.markdown("### 📍 Danh sách Địa điểm Xác thực")
                for p in result["places"]:
                    source = p.source if isinstance(p, NormalizedPlace) else "google_places"
                    icon = "📌" if source == "pinned" else "✅"
                    with st.expander(f"{icon} {p.name if isinstance(p, NormalizedPlace) else p['name']}"):
                        if isinstance(p, NormalizedPlace):
                            st.write(f"**Địa chỉ**: {p.address or 'N/A'}")
                            st.write(f"**Đánh giá**: {p.rating or 'N/A'} ⭐ ({p.user_rating_count} đánh giá)")
                            st.write(f"**Loại**: {p.category}")
                            st.write(f"**Nguồn**: {p.source}")
                            if p.google_maps_uri:
                                st.markdown(f"[Mở Google Maps]({p.google_maps_uri})")
                        else:
                            st.write(f"**Địa chỉ**: {p.get('address', 'N/A')}")
                            st.write(f"**Đánh giá**: {p.get('rating', 'N/A')} ⭐")

            with col_bud:
                st.markdown("### 💰 Dự toán Ngân sách Nhóm")
                b = result["budget"]
                budget_data = {
                    "Hạng mục": [
                        "Chỗ ở (Khách sạn/Homestay)",
                        "Ăn uống (Ẩm thực & Nhà hàng)",
                        "Phương tiện di chuyển",
                        "Tham quan & Trải nghiệm",
                        "Dự phòng phát sinh (10%)",
                    ],
                    "Số tiền (VND)": [
                        b["accommodation"], b["food"], b["transportation"],
                        b["activities"], b["reserve"],
                    ],
                }
                df_b = pd.DataFrame(budget_data)
                df_b["Số tiền (VND)"] = df_b["Số tiền (VND)"].apply(lambda x: f"{int(x):,} VND")
                st.table(df_b)

                st.metric("Tổng chi phí chuyến đi", f"{int(b['total']):,} VND")
                st.metric("Chia bình quân mỗi người", f"{int(b['per_person']):,} VND")
                st.caption("🔒 Tính toán số học chuẩn xác, không phụ thuộc LLM.")

                # Download markdown itinerary
                st.download_button(
                    label="📥 Tải lịch trình (Markdown)",
                    data=result["itinerary"],
                    file_name="tripmate_itinerary.md",
                    mime="text/markdown",
                )

    # ─────────────────────────────────────────────────────────────
    # TAB 2: GOOGLE PLACES EXPLORER + PIN
    # ─────────────────────────────────────────────────────────────
    with tab_places:
        st.markdown("### 📍 Khám phá Địa điểm với Places API (New)")
        st.caption("Tìm kiếm địa điểm và nhấn 📌 Ghim để ưu tiên chúng trong lịch trình AI.")

        p_tab1, p_tab2 = st.tabs(["🔍 Tìm theo từ khoá (Text Search)", "📡 Tìm kiếm lân cận (Nearby Search)"])

        with p_tab1:
            q_col1, q_col2 = st.columns([3, 1])
            search_query = q_col1.text_input("Từ khoá tìm kiếm:", value="nhà hàng ngon Đà Nẵng")
            max_p = q_col2.number_input("Số lượng kết quả:", 1, 10, 5)

            if st.button("Tìm kiếm địa điểm", key="btn_text_search"):
                with st.spinner("Đang truy vấn Places API..."):
                    try:
                        svc = st.session_state.places_service
                        results = svc.search_text(search_query, max_results=max_p)
                        if not results:
                            st.info("Không tìm thấy địa điểm. Thử từ khóa khác.")
                        else:
                            st.success(f"Tìm thấy {len(results)} địa điểm:")
                            _render_search_results_with_pin(results, "text")
                    except PlacesAPIError as e:
                        st.error(f"Lỗi Places API: {e.message}")

        with p_tab2:
            st.markdown("#### Tự động điền toạ độ theo thành phố:")
            preset_city = st.selectbox("Chọn nhanh thành phố:", list(DESTINATIONS_CATALOG.keys()), index=0)
            city_meta = DESTINATIONS_CATALOG[preset_city]

            c1, c2, c3 = st.columns(3)
            lat = c1.number_input("Vĩ độ (Latitude):", value=float(city_meta["latitude"]), format="%.4f")
            lng = c2.number_input("Kinh độ (Longitude):", value=float(city_meta["longitude"]), format="%.4f")
            radius = c3.number_input("Bán kính tìm kiếm (mét):", value=3000, step=500)

            if st.button("Tìm kiếm địa điểm xung quanh", key="btn_nearby_search"):
                with st.spinner("Đang quét địa điểm trong bán kính..."):
                    try:
                        svc = st.session_state.places_service
                        results = svc.search_nearby(lat, lng, radius_meters=radius)
                        if not results:
                            st.info("Không tìm thấy địa điểm nào trong bán kính này.")
                        else:
                            st.success(f"Tìm thấy {len(results)} địa điểm lân cận:")
                            _render_search_results_with_pin(results, "nearby")
                    except PlacesAPIError as e:
                        st.error(f"Lỗi Places API: {e.message}")

    # ─────────────────────────────────────────────────────────────
    # TAB 3: GROUP TRIP FUND MANAGER
    # ─────────────────────────────────────────────────────────────
    with tab_fund:
        st.markdown("### 💳 Quản lý Quỹ Nhóm Tạm Thời (Temporary Trip Fund)")
        st.caption("Theo dõi mục tiêu quỹ, phần đóng góp của từng thành viên và xác nhận chuyến đi.")

        trip_id = st.session_state.get("active_trip_id") or "trip-default"
        fund = fund_manager.get_fund(trip_id)
        if not fund:
            fund = fund_manager.create_fund(
                trip_id=trip_id,
                target_amount=6_600_000,
                member_names=["Thành viên 1", "Thành viên 2", "Thành viên 3", "Thành viên 4"],
            )
            st.session_state.fund = fund
            st.session_state.active_trip_id = trip_id

        fcol1, fcol2, fcol3 = st.columns(3)
        fcol1.metric("Mục tiêu quỹ", f"{int(fund['target_amount']):,} VND")
        fcol2.metric("Đã đóng góp", f"{int(fund['total_contributed']):,} VND")
        fcol3.metric("Trạng thái quỹ", fund["status"])

        progress = min(1.0, fund["total_contributed"] / fund["target_amount"]) if fund["target_amount"] > 0 else 0.0
        st.progress(progress)
        st.write(f"**Tiến độ thu quỹ: {round(progress * 100, 1)}%**")

        if fund["status"] in (STATUS_FULLY_FUNDED, STATUS_CONFIRMED):
            st.success("🎉 Quỹ đã được gom đủ 100% mục tiêu!")
            if fund["status"] == STATUS_FULLY_FUNDED:
                if st.button("✈️ Xác nhận chuyến đi (Confirm Trip)", type="primary"):
                    fund_manager.confirm_trip(st.session_state.active_trip_id)
                    st.rerun()
            elif fund["status"] == STATUS_CONFIRMED:
                st.info("✅ Chuyến đi đã được chốt và xác nhận chính thức!")

        st.markdown("#### Bảng phân bổ đóng góp thành viên")
        member_rows = []
        for m in fund["members"]:
            member_rows.append({
                "Thành viên": m["name"],
                "Hạn mức cần đóng (VND)": f"{int(m['target']):,} VND",
                "Đã đóng (VND)": f"{int(m['paid']):,} VND",
                "Trạng thái": (
                    "✅ Đã đóng đủ" if m["status"] == "paid"
                    else ("🟡 Đóng một phần" if m["status"] == "partial" else "⏳ Chưa đóng")
                ),
            })
        st.table(pd.DataFrame(member_rows))

        if fund["status"] != STATUS_CONFIRMED:
            st.markdown("#### Ghi nhận đóng góp:")
            cf1, cf2, cf3 = st.columns([2, 2, 1])
            contrib_member = cf1.selectbox("Chọn thành viên:", [m["name"] for m in fund["members"]])
            default_share = fund["members"][0]["target"] if fund["members"] else 1_650_000
            contrib_amount = cf2.number_input("Số tiền đóng góp (VND):", value=float(default_share), step=100_000.0)

            if cf3.button("Xác nhận đóng"):
                try:
                    fund_manager.contribute(trip_id, contrib_member, contrib_amount)
                    st.success(f"Đã ghi nhận {int(contrib_amount):,} VND cho {contrib_member}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi ghi nhận đóng góp: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# UI Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _render_structured_itinerary(items: list) -> None:
    """Render ItineraryItem list with type-specific badges and colours."""
    current_day = 0

    for item in items:
        if item.day != current_day:
            current_day = item.day
            st.markdown(f"#### 📅 Ngày {item.day} — {item.destination_name}")

        if item.item_type == ItemType.TRANSIT:
            ti = item.transit_info or {}
            dist = ti.get("distance_km")
            mins = ti.get("travel_time_min")
            detail = ""
            if dist or mins:
                d_str = f"{dist} km" if dist else ""
                m_str = f"{mins} phút" if mins else ""
                detail = f" ({d_str}{' · ' if d_str and m_str else ''}{m_str})"
            st.markdown(
                f'<span class="badge-transit">TRANSIT</span> '
                f'**{item.time_slot}** — {item.title}{detail}',
                unsafe_allow_html=True,
            )
            if item.description:
                st.caption(item.description)

        elif item.item_type == ItemType.ACTIVITY:
            st.markdown(f"🌅 **{item.time_slot}** — {item.title}")
            if item.description:
                st.caption(item.description)

        else:  # PLACE
            if item.status == PlaceStatus.UNVERIFIED:
                badge = '<span class="badge-unverified">⚠️ Bạn đề xuất</span>'
            else:
                badge = '<span class="badge-verified">✅ Đã xác thực</span>'

            st.markdown(
                f'{badge} **{item.time_slot}** — **{item.title}**',
                unsafe_allow_html=True,
            )
            if item.description:
                st.caption(item.description)

        st.divider() if item.item_type == ItemType.TRANSIT else None


def _render_search_results_with_pin(results: list, key_prefix: str) -> None:
    """Display search results with a 📌 Pin button beside each."""
    for idx, r in enumerate(results):
        with st.container():
            rc1, rc2 = st.columns([5, 1])
            with rc1:
                st.markdown(f"#### {r['name']}")
                st.write(f"**Địa chỉ**: {r.get('address', 'N/A')}")
                st.write(f"**Đánh giá**: {r.get('rating', 'N/A')} ⭐ ({r.get('user_rating_count', 0)} đánh giá)")
                if r.get("google_maps_uri"):
                    st.markdown(f"[Xem trên Google Maps]({r['google_maps_uri']})")
            with rc2:
                pin_key = f"pin_{key_prefix}_{idx}_{r.get('id', idx)}"
                already_pinned = any(
                    p.name == r["name"] for p in st.session_state.pinned_places
                )
                if already_pinned:
                    st.markdown("📌 *Đã ghim*")
                elif st.button("📌 Ghim", key=pin_key):
                    np = NormalizedPlace(
                        id=r.get("id", f"pinned_{idx}"),
                        name=r["name"],
                        category=r.get("category", "attraction"),
                        address=r.get("address", ""),
                        latitude=r.get("latitude") or 0.0,
                        longitude=r.get("longitude") or 0.0,
                        rating=r.get("rating") or 0.0,
                        user_rating_count=r.get("user_rating_count", 0),
                        google_maps_uri=r.get("google_maps_uri"),
                        source="pinned",
                    )
                    st.session_state.pinned_places.append(np)
                    st.rerun()
            st.divider()


if __name__ == "__main__":
    main()
