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

    # Tabs for TripMate Flow & Original bot features
    tab_planner, tab_places, tab_fund, tab_original = st.tabs([
        "🚀 AI Trip Planner",
        "📍 Google Places Search",
        "💳 Temporary Trip Fund",
        "🛠️ Original Tools"
    ])

    # ─────────────────────────────────────────────────────────────
    # TAB 1: AI TRIP PLANNER (Phase 4, 5, 6, 8)
    # ─────────────────────────────────────────────────────────────
    with tab_planner:
        col_input, col_meta = st.columns([2, 1])

        with col_input:
            st.markdown("#### 1. Trip Setup")
            col_t1, col_t2 = st.columns(2)
            trip_name = col_t1.text_input("Trip Name", value=st.session_state.trip_name)
            dest_input = col_t2.text_input("Destination", value=st.session_state.destination)

            col_d1, col_d2, col_d3 = st.columns(3)
            travelers_count = col_d1.number_input("Number of Travelers", min_value=1, max_value=20, value=st.session_state.travelers)
            days_count = col_d2.number_input("Duration (Days)", min_value=1, max_value=14, value=st.session_state.days)
            tier_choice = col_d3.selectbox("Budget Tier", ["budget", "moderate", "luxury"], index=1)

            st.markdown("#### 2. AI Prompt & Requirements")
            sample_prompt = f"Plan a {days_count}-day trip to {dest_input} for {travelers_count} people. Find dinner near Dragon Bridge and top attractions, estimate the budget."
            user_prompt = st.text_area(
                "Describe your group travel request:",
                value=sample_prompt,
                height=90,
                help="AI extracts parameters, verifies real places on Google Places API, and computes authoritative budget."
            )

            if st.button("✈️ Generate Trip Plan", type="primary"):
                planner = AITravelPlanner(places_service=st.session_state.places_service)
                with st.spinner("AI parsing request & querying Google Places API..."):
                    try:
                        plan_res = planner.generate_full_trip_plan(user_prompt)
                        st.session_state.plan_result = plan_res
                        
                        # Sync with Trip Fund target
                        total_budget = plan_res["budget"]["total"]
                        fund_manager.create_fund(
                            trip_id=st.session_state.active_trip_id,
                            target_amount=total_budget,
                            member_names=["Anh", "Minh", "Lan", "Hung"][:travelers_count] or ["Member 1"]
                        )
                        st.session_state.fund = fund_manager.get_fund(st.session_state.active_trip_id)
                        st.success("Plan generated successfully with verified Google Places!")
                    except Exception as e:
                        st.error(f"Error generating plan: {e}")

        with col_meta:
            st.markdown("#### ⚙️ How AI + Places Works")
            st.markdown("""
            ```mermaid
            flowchart TD
            A[User Request] --> B[AI Intent Extraction]
            B --> C[Google Places API New]
            C --> D[Real Places Verified]
            D --> E[Synthesized Itinerary]
            D --> F[Deterministic Budget Engine]
            ```
            """)
            st.info("🔒 **Anti-Hallucination**: Venues are strictly matched with Google Places API. AI is prohibited from inventing restaurants or attractions.")

        # Display Result
        if st.session_state.plan_result:
            plan = st.session_state.plan_result
            st.markdown("---")
            
            # Overview Metrics
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Destination", plan["intent"]["destination"])
            mcol2.metric("Duration / Travelers", f"{plan['intent']['days']} Days / {plan['intent']['travelers']} Pax")
            mcol3.metric("Verified Places", f"{len(plan['places'])} Places")
            mcol4.metric("Estimated Total", f"{int(plan['budget']['total']):,} VND")

            # Two columns: Verified Places + Itinerary
            col_itin, col_bud = st.columns([3, 2])

            with col_itin:
                st.markdown("### 🗺️ Day-by-Day Itinerary")
                st.markdown(plan["itinerary"])

                st.markdown("### 📍 Verified Places in this Itinerary")
                for p in plan["places"]:
                    with st.expander(f"📍 {p['name']} ({p.get('category', 'attraction').capitalize()})"):
                        st.write(f"**Address**: {p.get('address', 'N/A')}")
                        st.write(f"**Rating**: {p.get('rating', 'N/A')} ⭐ ({p.get('user_rating_count', 0)} reviews)")
                        if p.get("google_maps_uri"):
                            st.markdown(f"[View on Google Maps]({p['google_maps_uri']})")

            with col_bud:
                st.markdown("### 💰 Estimated Trip Budget")
                b = plan["budget"]
                
                budget_data = {
                    "Category": ["Accommodation", "Food & Dining", "Transportation", "Activities & Sights", "Reserve Buffer (10%)"],
                    "Amount (VND)": [b["accommodation"], b["food"], b["transportation"], b["activities"], b["reserve"]]
                }
                df_b = pd.DataFrame(budget_data)
                df_b["Amount (VND)"] = df_b["Amount (VND)"].apply(lambda x: f"{int(x):,} VND")
                st.table(df_b)

                st.metric("Total Group Budget", f"{int(b['total']):,} VND")
                st.metric("Per Person", f"{int(b['per_person']):,} VND")
                
                st.caption("ℹ️ Computed deterministically by TripMate Budget Engine. Not hallucinated by LLM.")

    # ─────────────────────────────────────────────────────────────
    # TAB 2: GOOGLE PLACES SEARCH (Phase 3)
    # ─────────────────────────────────────────────────────────────
    with tab_places:
        st.markdown("### 🔍 Google Places API (New) Explorer")
        st.caption("Directly test Text Search, Nearby Search, and Place Details.")

        p_tab1, p_tab2 = st.tabs(["Text Search", "Nearby Search"])

        with p_tab1:
            q_col1, q_col2 = st.columns([3, 1])
            search_query = q_col1.text_input("Search query", value="restaurants near Dragon Bridge Da Nang")
            max_p = q_col2.number_input("Max results", 1, 10, 5)

            if st.button("Search Places"):
                with st.spinner("Calling Google Places API..."):
                    try:
                        svc = st.session_state.places_service
                        results = svc.search_text(search_query, max_results=max_p)
                        if not results:
                            st.info("No places were found for this search. Try a larger area or a different keyword.")
                        else:
                            st.success(f"Found {len(results)} places:")
                            for r in results:
                                with st.container():
                                    st.markdown(f"#### {r['name']}")
                                    st.write(f"**Address**: {r.get('address')}")
                                    st.write(f"**Rating**: {r.get('rating')} ⭐ ({r.get('user_rating_count')} reviews)")
                                    if r.get('google_maps_uri'):
                                        st.markdown(f"[Google Maps Link]({r['google_maps_uri']})")
                                    st.divider()
                    except PlacesAPIError as e:
                        st.error(f"Places API Error: {e.message}")

        with p_tab2:
            st.markdown("Search around coordinates (e.g., Da Nang center: 16.0611, 108.2272)")
            c1, c2, c3 = st.columns(3)
            lat = c1.number_input("Latitude", value=16.0611, format="%.4f")
            lng = c2.number_input("Longitude", value=108.2272, format="%.4f")
            radius = c3.number_input("Radius (meters)", value=3000, step=500)

            if st.button("Search Nearby"):
                with st.spinner("Searching nearby..."):
                    try:
                        svc = st.session_state.places_service
                        results = svc.search_nearby(lat, lng, radius_meters=radius)
                        if not results:
                            st.info("No places found within this radius.")
                        else:
                            st.success(f"Found {len(results)} places nearby:")
                            for r in results:
                                st.write(f"**{r['name']}** — {r.get('address', 'N/A')}")
                    except PlacesAPIError as e:
                        st.error(f"Places API Error: {e.message}")

    # ─────────────────────────────────────────────────────────────
    # TAB 3: TEMPORARY TRIP FUND (Phase 7 & 8)
    # ─────────────────────────────────────────────────────────────
    with tab_fund:
        st.markdown("### 💳 Temporary Trip Fund")
        st.caption("Internal project fund tracker for TripMate group coordination (no actual banking required).")

        fund = fund_manager.get_fund(st.session_state.active_trip_id)
        if not fund:
            fund = fund_manager.create_fund(
                trip_id=st.session_state.active_trip_id,
                target_amount=6_600_000,
                member_names=["Anh", "Minh", "Lan", "Hung"]
            )
            st.session_state.fund = fund

        # Status Banner
        status_color = {
            "DRAFT": "badge-yellow",
            "FUNDRAISING": "badge-yellow",
            "FULLY_FUNDED": "badge-teal",
            "CONFIRMED": "badge-teal",
        }.get(fund["status"], "badge-yellow")

        fcol1, fcol2, fcol3 = st.columns(3)
        fcol1.metric("Target Amount", f"{int(fund['target_amount']):,} VND")
        fcol2.metric("Collected Amount", f"{int(fund['total_contributed']):,} VND")
        fcol3.metric("Status", fund["status"])

        # Progress bar
        progress = min(1.0, fund["total_contributed"] / fund["target_amount"]) if fund["target_amount"] > 0 else 0.0
        st.progress(progress)
        st.write(f"**Progress: {round(progress * 100, 1)}%**")

        # Fully Funded Notification
        if fund["status"] in (STATUS_FULLY_FUNDED, STATUS_CONFIRMED):
            st.success("✓ Fund fully collected!")
            if fund["status"] == STATUS_FULLY_FUNDED:
                if st.button("🎉 Confirm Trip", type="primary"):
                    fund_manager.confirm_trip(st.session_state.active_trip_id)
                    st.rerun()
            elif fund["status"] == STATUS_CONFIRMED:
                st.info("✈️ Trip has been confirmed!")

        st.markdown("#### Member Contributions")
        member_rows = []
        for m in fund["members"]:
            member_rows.append({
                "Member": m["name"],
                "Target (VND)": f"{int(m['target']):,} VND",
                "Paid (VND)": f"{int(m['paid']):,} VND",
                "Status": "✅ Paid" if m["status"] == "paid" else ("🟡 Partial" if m["status"] == "partial" else "⏳ Pending")
            })
        st.table(pd.DataFrame(member_rows))

        # Contribution Form
        if fund["status"] != STATUS_CONFIRMED:
            st.markdown("#### Record Contribution")
            cf1, cf2, cf3 = st.columns([2, 2, 1])
            contrib_member = cf1.selectbox("Select Member", [m["name"] for m in fund["members"]])
            default_share = fund["members"][0]["target"] if fund["members"] else 1_650_000
            contrib_amount = cf2.number_input("Amount (VND)", value=float(default_share), step=100_000.0)

            if cf3.button("Pay Contribution"):
                try:
                    fund_manager.contribute(st.session_state.active_trip_id, contrib_member, contrib_amount)
                    st.success(f"Recorded {int(contrib_amount):,} VND for {contrib_member}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # ─────────────────────────────────────────────────────────────
    # TAB 4: ORIGINAL BOT TOOLS
    # ─────────────────────────────────────────────────────────────
    with tab_original:
        st.markdown("### 🛠️ Original Travel Itinerary Bot Tools")
        st.caption("Multi-destination, free-form LLM packing list, and original text generation.")
        
        orig1, orig2 = st.columns(2)
        with orig1:
            st.markdown("#### Packing List Generator")
            pack_dest = st.text_input("Packing Destination", value="Da Nang")
            pack_days = st.number_input("Packing Days", value=3, min_value=1)
            if st.button("🎒 Generate Packing List"):
                with st.spinner("Generating..."):
                    pl = generate_packing_list(pack_dest, pack_days, "beach, walking, dining", model=cfg["model"]["name"])
                    st.markdown(pl)

        with orig2:
            st.markdown("#### Multi-Destination Itinerary")
            multi_dests = st.text_input("Destinations (comma separated)", value="Da Nang, Hoi An, Hue")
            if st.button("✈️ Multi-Destination Plan"):
                if not check_ollama_running():
                    st.warning("Local Ollama server is not running. Please start Ollama or use the TripMate AI Planner tab.")
                else:
                    d_list = parse_destinations(multi_dests)
                    res = generate_multi_destination_itinerary(d_list, 2, "moderate", model=cfg["model"]["name"])
                    st.markdown(res)


if __name__ == "__main__":
    main()
