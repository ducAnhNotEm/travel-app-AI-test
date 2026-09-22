"""Streamlit web interface for TripMate — Collaborative Group Travel Planner."""

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
    defaults = {
        "config": load_config(),
        "plan_result": None,
        "active_trip_id": "danang-2026",
        "trip_name": "Da Nang Group Vacation",
        "destination": "Da Nang",
        "travelers": 4,
        "days": 3,
        "fund": None,
        "places_service": PlacesService(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.fund is None:
        st.session_state.fund = fund_manager.get_fund("danang-2026") or fund_manager.create_fund(
            trip_id="danang-2026",
            target_amount=6_600_000,
            member_names=["Anh", "Minh", "Lan", "Hung"]
        )


def main():
    init_state()
    cfg = st.session_state.config
    setup_logging(cfg)

    # Top Brand Bar
    st.markdown('<div class="brand-title">TripMate</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Collaborative Group Travel Planner with AI & Google Places API</div>', unsafe_allow_html=True)

    # Sidebar: Trip Settings & Keys
    with st.sidebar:
        st.markdown("### ⚙️ TripMate Settings")
        st.caption("Google Places & AI Configuration")
        
        # Check API key status
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
        st.markdown("### 👥 Active Trip Info")
        st.text(f"Trip ID: {st.session_state.active_trip_id}")
        st.text(f"Name: {st.session_state.trip_name}")

    # Tabs for TripMate Flow: 3 streamlined, highly interactive SaaS tabs
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
            col_t1, col_t2 = st.columns([1, 1])
            selected_dest_key = col_t1.selectbox(
                "Điểm đến (Thành phố):",
                dest_options,
                index=0,
                help="Chọn nhanh từ danh sách điểm đến hàng đầu hoặc tự nhập"
            )
            
            if selected_dest_key == "Other (Custom Destination)...":
                active_dest = col_t2.text_input("Nhập tên điểm đến:", value="Hạ Long")
                dest_tags = ["🍜 Ẩm thực", "🏖️ Ngắm cảnh", "📸 Check-in", "☕ Cafe"]
            else:
                dest_info = DESTINATIONS_CATALOG[selected_dest_key]
                active_dest = dest_info["name"]
                dest_tags = dest_info.get("tags", [])
                col_t2.text_input("Quốc gia / Khu vực:", value=dest_info.get("country", "Vietnam"), disabled=True)

            col_d1, col_d2 = st.columns(2)
            travelers_count = col_d1.slider("Số lượng thành viên (Travelers):", min_value=1, max_value=20, value=4)
            days_count = col_d2.slider("Thời gian chuyến đi (Số ngày):", min_value=1, max_value=14, value=3)

            col_s1, col_s2 = st.columns(2)
            tier_choice = col_s1.radio(
                "Phân khúc ngân sách:",
                ["budget", "moderate", "luxury"],
                index=1,
                format_func=lambda x: {"budget": "Tiết kiệm 💸 (Budget)", "moderate": "Tiêu chuẩn ⚖️ (Moderate)", "luxury": "Cao cấp 💎 (Luxury)"}[x],
                horizontal=True
            )
            trans_choice = col_s2.radio(
                "Phương tiện di chuyển:",
                ["car", "motorcycle", "public", "flight"],
                index=0,
                format_func=lambda x: {"car": "🚗 Ô tô / Taxi", "motorcycle": "🛵 Xe máy", "public": "🚌 Xe buýt / Công cộng", "flight": "✈️ Máy bay"}[x],
                horizontal=True
            )

            # Interactive Tag Chips for User Preferences
            st.markdown("#### 2. Sở thích & Điểm nhấn mong muốn:")
            chosen_tags = st.multiselect(
                "Chọn các chủ đề bạn quan tâm:",
                options=dest_tags,
                default=dest_tags[:2] if len(dest_tags) >= 2 else dest_tags,
                help="AI sẽ dựa vào các thẻ này để lọc và tìm địa điểm phù hợp trên Google Places API"
            )

            tag_summary = ", ".join(chosen_tags) if chosen_tags else "thưởng thức ẩm thực và tham quan"
            generated_prompt = f"Lập kế hoạch du lịch {days_count} ngày tại {active_dest} cho đoàn {travelers_count} người, phong cách {tier_choice}, di chuyển bằng {trans_choice}. Tập trung vào: {tag_summary}. Tìm kiếm địa điểm ăn uống và tham quan thực tế kèm dự toán chi phí chi tiết."

            st.markdown("#### 3. Yêu cầu chi tiết gửi AI:")
            user_prompt = st.text_area(
                "Prompt mô tả kế hoạch du lịch:",
                value=generated_prompt,
                height=95,
                help="TripMate phân tích prompt, truy vấn Google Places API lấy địa điểm thực tế và tự động tính ngân sách chi tiết."
            )

            if st.button("🚀 Khởi tạo Lịch trình & Ngân sách", type="primary"):
                planner = AITravelPlanner(places_service=st.session_state.places_service)
                with st.spinner(f"AI đang phân tích yêu cầu & truy vấn Places API cho {active_dest}..."):
                    try:
                        plan_res = planner.generate_full_trip_plan(user_prompt)
                        st.session_state.plan_result = plan_res
                        st.session_state.destination = active_dest
                        st.session_state.travelers = travelers_count
                        st.session_state.days = days_count

                        # Sync with Trip Fund target
                        total_budget = plan_res["budget"]["total"]
                        fund_manager.create_fund(
                            trip_id=st.session_state.active_trip_id,
                            target_amount=total_budget,
                            member_names=["Thành viên 1", "Thành viên 2", "Thành viên 3", "Thành viên 4"][:travelers_count] or ["Thành viên 1"]
                        )
                        st.session_state.fund = fund_manager.get_fund(st.session_state.active_trip_id)
                        st.success(f"✓ Đã tạo thành công lịch trình {active_dest} với địa điểm thật & dự toán ngân sách!")
                    except Exception as e:
                        st.error(f"Lỗi khởi tạo lịch trình: {e}")

        with col_meta:
            st.markdown("#### ⚡ Cơ chế Vận hành TripMate")
            st.markdown("""
            ```mermaid
            flowchart TD
            A[Tùy chọn tương tác] --> B[AI Intent Parser]
            B --> C[Google Places API New]
            C --> D[Xác thực địa điểm thật]
            D --> E[Lập lịch trình chi tiết]
            D --> F[Deterministic Budget Engine]
            F --> G[Quỹ nhóm Trip Fund]
            ```
            """)
            st.markdown("""
            > [!TIP]
            > **100% Chống Ảo giác (Anti-Hallucination)**  
            > Mọi địa điểm, nhà hàng, danh lam thắng cảnh được xác thực trực tiếp qua **Google Places API / Geoapify**.
            """)
            if selected_dest_key != "Other (Custom Destination)...":
                info = DESTINATIONS_CATALOG[selected_dest_key]
                st.caption(f"📍 Toạ độ trung tâm: `{info['latitude']}, {info['longitude']}`")

        # Display Result
        if st.session_state.plan_result:
            plan = st.session_state.plan_result
            st.markdown("---")
            
            # Overview Metrics
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Điểm đến", plan["intent"]["destination"])
            mcol2.metric("Thời gian / Số người", f"{plan['intent']['days']} Ngày / {plan['intent']['travelers']} Pax")
            mcol3.metric("Địa điểm xác thực", f"{len(plan['places'])} Địa điểm")
            mcol4.metric("Dự toán tổng", f"{int(plan['budget']['total']):,} VND")

            # Two columns: Verified Places + Itinerary
            col_itin, col_bud = st.columns([3, 2])

            with col_itin:
                st.markdown("### 🗺️ Lịch trình Chi tiết Từng Ngày")
                st.markdown(plan["itinerary"])

                st.markdown("### 📍 Danh sách Địa điểm thật đã xác thực")
                for p in plan["places"]:
                    with st.expander(f"📍 {p['name']} ({p.get('category', 'attraction').capitalize()})"):
                        st.write(f"**Địa chỉ**: {p.get('address', 'Khu vực trung tâm')}")
                        st.write(f"**Đánh giá**: {p.get('rating', '4.5')} ⭐ ({p.get('user_rating_count', 0)} đánh giá)")
                        if p.get("google_maps_uri"):
                            st.markdown(f"[Mở trên Google Maps]({p['google_maps_uri']})")

            with col_bud:
                st.markdown("### 💰 Dự toán Ngân sách Nhóm")
                b = plan["budget"]
                
                budget_data = {
                    "Hạng mục": ["Chỗ ở (Khách sạn/Homestay)", "Ăn uống (Ẩm thực & Nhà hàng)", "Phương tiện di chuyển", "Tham quan & Trải nghiệm", "Dự phòng phát sinh (10%)"],
                    "Số tiền (VND)": [b["accommodation"], b["food"], b["transportation"], b["activities"], b["reserve"]]
                }
                df_b = pd.DataFrame(budget_data)
                df_b["Số tiền (VND)"] = df_b["Số tiền (VND)"].apply(lambda x: f"{int(x):,} VND")
                st.table(df_b)

                st.metric("Tổng chi phí chuyến đi", f"{int(b['total']):,} VND")
                st.metric("Chia bình quân mỗi người", f"{int(b['per_person']):,} VND")
                st.caption("🔒 Tính toán số học chuẩn xác qua Budget Engine, không phụ thuộc ảo giác LLM.")

    # ─────────────────────────────────────────────────────────────
    # TAB 2: GOOGLE PLACES EXPLORER
    # ─────────────────────────────────────────────────────────────
    with tab_places:
        st.markdown("### 📍 Khám phá Địa điểm với Places API (New)")
        st.caption("Tìm kiếm địa điểm du lịch, quán ăn, quán cafe và tìm kiếm theo toạ độ bán kính.")

        p_tab1, p_tab2 = st.tabs(["🔍 Tìm theo từ khoá (Text Search)", "📡 Tìm kiếm lân cận (Nearby Search)"])

        with p_tab1:
            q_col1, q_col2 = st.columns([3, 1])
            search_query = q_col1.text_input("Từ khoá tìm kiếm:", value="nhà hàng ngon Đà Nẵng")
            max_p = q_col2.number_input("Số lượng kết quả tối đa:", 1, 10, 5)

            if st.button("Tìm kiếm địa điểm"):
                with st.spinner("Đang truy vấn Places API..."):
                    try:
                        svc = st.session_state.places_service
                        results = svc.search_text(search_query, max_results=max_p)
                        if not results:
                            st.info("Không tìm thấy địa điểm nào phù hợp. Hãy thử từ khóa khác.")
                        else:
                            st.success(f"Tìm thấy {len(results)} địa điểm:")
                            for r in results:
                                with st.container():
                                    st.markdown(f"#### {r['name']}")
                                    st.write(f"**Địa chỉ**: {r.get('address')}")
                                    st.write(f"**Đánh giá**: {r.get('rating')} ⭐ ({r.get('user_rating_count')} đánh giá)")
                                    if r.get('google_maps_uri'):
                                        st.markdown(f"[Xem bản đồ Google Maps]({r['google_maps_uri']})")
                                    st.divider()
                    except PlacesAPIError as e:
                        st.error(f"Lỗi Places API: {e.message}")

        with p_tab2:
            st.markdown("#### Tự động điền toạ độ theo thành phố:")
            preset_city = st.selectbox(
                "Chọn nhanh thành phố:",
                list(DESTINATIONS_CATALOG.keys()),
                index=0
            )
            city_meta = DESTINATIONS_CATALOG[preset_city]

            c1, c2, c3 = st.columns(3)
            lat = c1.number_input("Vĩ độ (Latitude):", value=float(city_meta["latitude"]), format="%.4f")
            lng = c2.number_input("Kinh độ (Longitude):", value=float(city_meta["longitude"]), format="%.4f")
            radius = c3.number_input("Bán kính tìm kiếm (mét):", value=3000, step=500)

            if st.button("Tìm kiếm địa điểm xung quanh"):
                with st.spinner("Đang quét địa điểm trong bán kính..."):
                    try:
                        svc = st.session_state.places_service
                        results = svc.search_nearby(lat, lng, radius_meters=radius)
                        if not results:
                            st.info("Không tìm thấy địa điểm nào trong bán kính này.")
                        else:
                            st.success(f"Tìm thấy {len(results)} địa điểm lân cận:")
                            for r in results:
                                st.write(f"**{r['name']}** — {r.get('address', 'N/A')}")
                    except PlacesAPIError as e:
                        st.error(f"Lỗi Places API: {e.message}")

    # ─────────────────────────────────────────────────────────────
    # TAB 3: GROUP TRIP FUND MANAGER
    # ─────────────────────────────────────────────────────────────
    with tab_fund:
        st.markdown("### 💳 Quản lý Quỹ Nhóm Tạm Thời (Temporary Trip Fund)")
        st.caption("Theo dõi mục tiêu quỹ, phần đóng góp của từng thành viên và tự động xác nhận chuyến đi khi gom đủ tiền.")

        fund = fund_manager.get_fund(st.session_state.active_trip_id)
        if not fund:
            fund = fund_manager.create_fund(
                trip_id=st.session_state.active_trip_id,
                target_amount=6_600_000,
                member_names=["Thành viên 1", "Thành viên 2", "Thành viên 3", "Thành viên 4"]
            )
            st.session_state.fund = fund

        # Status Banner
        fcol1, fcol2, fcol3 = st.columns(3)
        fcol1.metric("Mục tiêu quỹ", f"{int(fund['target_amount']):,} VND")
        fcol2.metric("Đã đóng góp", f"{int(fund['total_contributed']):,} VND")
        fcol3.metric("Trạng thái quỹ", fund["status"])

        # Progress bar
        progress = min(1.0, fund["total_contributed"] / fund["target_amount"]) if fund["target_amount"] > 0 else 0.0
        st.progress(progress)
        st.write(f"**Tiến độ thu quỹ: {round(progress * 100, 1)}%**")

        # Fully Funded Notification & Confirmation Trigger
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
                "Trạng thái": "✅ Đã đóng đủ" if m["status"] == "paid" else ("🟡 Đóng một phần" if m["status"] == "partial" else "⏳ Chưa đóng")
            })
        st.table(pd.DataFrame(member_rows))

        # Contribution Form
        if fund["status"] != STATUS_CONFIRMED:
            st.markdown("#### Ghi nhận đóng góp:")
            cf1, cf2, cf3 = st.columns([2, 2, 1])
            contrib_member = cf1.selectbox("Chọn thành viên:", [m["name"] for m in fund["members"]])
            default_share = fund["members"][0]["target"] if fund["members"] else 1_650_000
            contrib_amount = cf2.number_input("Số tiền đóng góp (VND):", value=float(default_share), step=100_000.0)

            if cf3.button("Xác nhận đóng"):
                try:
                    fund_manager.contribute(st.session_state.active_trip_id, contrib_member, contrib_amount)
                    st.success(f"Đã ghi nhận {int(contrib_amount):,} VND cho {contrib_member}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi ghi nhận đóng góp: {e}")


if __name__ == "__main__":
    main()
