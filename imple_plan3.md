# 🧭 Kế Hoạch Triển Khai: Hệ Thống Hybrid AI Travel Planning (TripMate 2.0)

Bản kế hoạch này nâng cấp TripMate theo đúng chuẩn kiến trúc phần mềm đồ án chuyên sâu: **Tách biệt hoàn toàn tầng LLM (NLP), tầng dữ liệu thực (Google Places), và tầng thuật toán quyết định (Deterministic Planning Pipeline)**.

> [!NOTE]
> **Nguyên tắc cốt lõi về Anti-Hallucination**:
> *"Giảm thiểu hallucination bằng cách giới hạn LLM chỉ diễn đạt dữ liệu đã được xác định bởi hệ thống."*
> Toàn bộ quyết định về địa điểm nào được đưa vào, thứ tự ra sao, thời gian nào và khi nào di chuyển liên tỉnh đều do **Code & Thuật toán** đảm nhận 100%.

---

## 🏗️ Kiến Trúc Hệ Thống Chuẩn (Target Architecture)

```
User
 ↓
Streamlit UI
 ↓
TravelIntent (LLM Intent Extractor / Heuristic)
 ↓
┌─────────────────────────────┐
│ Place Resolution            │  --> resolver.py
│ - Google Places             │
│ - Custom Places             │
│ - Pinned Places             │
└─────────────┬───────────────┘
              ↓
       Place Fusion              --> fusion.py
              ↓
      Constraint Engine          --> optimizer.py
              ↓
      Route Optimizer            --> optimizer.py
              ↓
     Itinerary Builder           --> itinerary.py
              ↓
   ┌──────────┴──────────┐
   ↓                     ↓
Budget Engine       LLM Formatter (ai_planner.py)
   ↓                     ↓
   └──────────┬──────────┘
              ↓
        Streamlit UI
```

---

## 📦 Cấu Trúc Module Chuẩn (Modular Code Organization)

Thay vì dồn logic vào một file, hệ thống phân chia rõ ràng trách nhiệm đơn lẻ (Single Responsibility Principle):

```
src/travel_planner/
├── models.py        # Data classes & enums: TravelIntent, DestinationItem, CustomPlace, ItineraryItem
├── places.py        # Google Places / Geoapify API client (Search, Nearby, Details)
├── resolver.py      # Custom & prompt place resolution against Places API (verified vs unverified)
├── fusion.py        # Place Fusion Engine (Hợp nhất Pinned, Custom, Google Recommendations)
├── optimizer.py     # Constraint Engine (Time slots, Meal slots) & Route Optimizer (Haversine / Nearest Neighbor)
├── itinerary.py     # Deterministic Itinerary Builder (Sinh danh sách ItineraryItem: PLACE, ACTIVITY, TRANSIT)
├── ai_planner.py    # LLM Intent Extraction (đầu vào) & LLM Formatter (đầu ra)
├── budget.py        # Deterministic Budget Engine (tính toán chi phí)
├── fund.py          # Quỹ nhóm tạm thời Trip Fund
├── destinations.py  # Destination metadata & fallback catalog
└── web_ui.py        # Streamlit SaaS Dashboard
```

---

## 📐 Chi Tiết Thiết Kế Data Models (`src/travel_planner/models.py`)

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

class ItemType(str, Enum):
    PLACE = "PLACE"        # POI cụ thể: nhà hàng, điểm tham quan, cafe
    ACTIVITY = "ACTIVITY"  # Hoạt động trải nghiệm: ngắm hoàng hôn, tắm biển, dạo phố
    TRANSIT = "TRANSIT"    # Di chuyển giữa các tỉnh/thành phố hoặc chặng xa

class PlaceStatus(str, Enum):
    VERIFIED = "verified"      # Đã xác thực trên Google Places (có place_id, lat, lng, rating)
    UNVERIFIED = "unverified"  # Người dùng đề xuất, không tìm thấy trên bản đồ

@dataclass
class DestinationItem:
    name: str
    days: int
    order: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None

@dataclass
class CustomPlace:
    name: str
    category: str  # restaurant, attraction, cafe, hotel
    source: str = "user_prompt"
    status: PlaceStatus = PlaceStatus.UNVERIFIED
    place_id: Optional[str] = None
    address: Optional[str] = None
    rating: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    target_destination: Optional[str] = None
    target_day: Optional[int] = None
    target_slot: Optional[str] = None  # breakfast, lunch, dinner, morning, afternoon, evening

@dataclass
class NormalizedPlace:
    id: str
    name: str
    category: str
    address: str
    latitude: float
    longitude: float
    rating: float
    user_rating_count: int
    google_maps_uri: Optional[str] = None
    source: str = "google_places"  # google_places, pinned, custom_verified

@dataclass
class ItineraryItem:
    item_type: ItemType
    title: str
    time_slot: str  # Ví dụ: "07:30 - 08:30", "09:00 - 11:30", "12:00 - 13:30"
    day: int
    destination_name: str
    description: str = ""
    place_details: Optional[Dict[str, Any]] = None  # Full normalized place info nếu là PLACE
    status: Optional[PlaceStatus] = None            # verified / unverified nếu là PLACE
    transit_info: Optional[Dict[str, Any]] = None   # From, To, Distance/Time nếu là TRANSIT

@dataclass
class TravelIntent:
    destinations: List[DestinationItem]
    total_days: int
    travelers: int
    transportation: str
    tier: str
    custom_places: List[CustomPlace] = field(default_factory=list)
    preferences: List[str] = field(default_factory=list)
    budget_vnd: Optional[float] = None
```

---

## 🛠️ Quy Trình Thuật Toán & Xử Lý Logic (Algorithmic Logic)

### 1. Phân Bổ Ngày & Điểm Chuyển Tỉnh (Deterministic Transit Scheduling)
Thay vì để AI tự đoán thời điểm di chuyển, hệ thống tính toán chính xác bằng code:
- Mảng `destinations = [ {"name": "Đà Nẵng", "days": 2, "order": 1}, {"name": "Hội An", "days": 1, "order": 2} ]`
- **Quy tắc lịch trình**:
  - `Day 1`: Thuộc **Đà Nẵng** (100% thời gian tại Đà Nẵng).
  - `Day 2`: Thuộc **Đà Nẵng** (100% thời gian tại Đà Nẵng).
  - `Day 3`: Bắt đầu chặng **Hội An**.
    - **Đầu ngày 3 (08:00 - 09:00)**: Chèn tự động item `TRANSIT`: *"Di chuyển từ Đà Nẵng → Hội An (~30km - 45 phút ô tô)"*.
    - Các khung giờ còn lại của Day 3: Tham quan, ăn uống tại Hội An.

### 2. Place Resolver (`resolver.py`)
- Quét từng `CustomPlace` do người dùng nhập trong Prompt:
  - Gọi `PlacesService.search_text(f"{place.name} {destination}")`.
  - Nếu có kết quả tin cậy: Cập nhật `status = VERIFIED`, gán `place_id`, `address`, `rating`, `lat/lng`.
  - Nếu không tìm thấy: Giữ `status = UNVERIFIED`. Vẫn đưa vào lịch trình với nhãn `[Địa điểm do bạn đề xuất]`.

### 3. Place Fusion (`fusion.py`)
- Hợp nhất theo phân cấp ưu tiên:
  1. **Pinned Places** (Do người dùng ghim trực tiếp từ UI Explorer) — Ưu tiên cao nhất (#1).
  2. **Custom Places** (Trích xuất từ prompt, cả verified & unverified) — Ưu tiên #2.
  3. **Verified AI Places** (Truy vấn Google Places API theo danh mục và điểm đến) — Lấp đầy các vị trí còn trống.
- Khử trùng lặp thông minh theo `place_id` hoặc tên địa điểm.

### 4. Constraint Engine & Route Optimizer (`optimizer.py`)
- **Constraint Engine**:
  - Phân bổ slot theo chu kỳ ngày:
    - Bữa sáng (07:30 - 08:30, category: `restaurant` / `cafe`)
    - Buổi sáng (09:00 - 11:30, category: `attraction`)
    - Bữa trưa (12:00 - 13:30, category: `restaurant`)
    - Buổi chiều (14:30 - 17:00, category: `attraction`)
    - Chiều tà (17:30 - 18:30, type: `ACTIVITY` - dạo biển, ngắm hoàng hôn, nghỉ ngơi)
    - Bữa tối (19:00 - 20:30, category: `restaurant`)
    - Buổi tối (21:00 - 22:30, category: `cafe` / `nightlife` hoặc `ACTIVITY`)
  - Khóa (Lock) các địa điểm chỉ định vào đúng slot mục tiêu:
    - Nếu prompt yêu cầu *"tối ngày 1 ăn Cơm Niêu Nhà Đỏ"*: Khóa quán vào Slot Bữa tối Ngày 1.
- **Route Optimizer (Nearest Neighbor via Haversine Distance)**:
  - Tính ma trận khoảng cách giữa các POI trong cùng 1 buổi/ngày.
  - Sắp xếp thứ tự ghé thăm theo POI gần nhất để tối ưu hành trình, tránh chạy vòng vèo.

### 5. Itinerary Builder (`itinerary.py`)
- Lắp ráp danh sách các `ItineraryItem` (`PLACE`, `ACTIVITY`, `TRANSIT`).
- Đảm bảo tính nhất quán và hoàn chỉnh của từng ngày.

### 6. LLM Formatter (`ai_planner.py`)
- Nhận đầu vào là danh sách `ItineraryItem` đã được chốt cố định.
- Nhiệm vụ duy nhất của LLM: Tạo lời bình sinh động, mẹo du lịch, lưu ý trang phục, gợi ý món ăn đặc sắc tại các địa điểm đã có sẵn trong danh sách.
- **Quy tắc nghiêm ngặt**: LLM không được phép đổi tên hoặc bịa thêm bất kỳ POI nào khác.

---

## 🚀 Kế Hoạch Triển Khai Từng Bước (Implementation Phases)

### Phase 1 — Data Models (`src/travel_planner/models.py`) [NEW]
- Xây dựng các dataclass và enum: `ItemType`, `PlaceStatus`, `DestinationItem`, `CustomPlace`, `NormalizedPlace`, `ItineraryItem`, `TravelIntent`.

### Phase 2 — Place Resolver (`src/travel_planner/resolver.py`) [NEW]
- Viết logic phân giải địa điểm tùy chỉnh: `resolve_custom_places(custom_places, destination)`.

### Phase 3 — Place Fusion Engine (`src/travel_planner/fusion.py`) [NEW]
- Viết logic hợp nhất Pinned Places, Custom Places và Google Places theo độ ưu tiên và khử trùng lặp.

### Phase 4 — Constraint Engine & Route Optimizer (`src/travel_planner/optimizer.py`) [NEW]
- Triển khai tính khoảng cách Haversine.
- Xây dựng thuật toán Nearest Neighbor cho các POI trong ngày.
- Triển khai Constraint Engine phân bổ slot và khóa địa điểm cố định.

### Phase 5 — Deterministic Itinerary Builder (`src/travel_planner/itinerary.py`) [NEW]
- Xây dựng bộ tạo lịch trình xác định (Deterministic Itinerary Builder) sinh ra danh sách `ItineraryItem`.
- Xử lý chuyển tỉnh `TRANSIT` theo số ngày phân bổ của từng `DestinationItem`.

### Phase 6 — Cập nhật AI Planner (`src/travel_planner/ai_planner.py`) [MODIFY]
- Tinh gọn `ai_planner.py`:
  - `extract_travel_intent()`: Nâng cấp regex và LLM prompt để bóc tách multi-destination (`destinations`) và `custom_places`.
  - Kết nối pipeline hoàn chỉnh: `TravelIntent` -> `resolver` -> `fusion` -> `optimizer` -> `itinerary`.
  - `format_itinerary()`: Dùng LLM hoặc template engine định dạng kết quả từ `ItineraryItem`.

### Phase 7 — UI Streamlit Tích Hợp (`src/travel_planner/web_ui.py`) [MODIFY]
- Hỗ trợ thêm/bớt nhiều điểm đến (Multi-destination).
- Tích hợp bộ tìm kiếm và nút **📌 Ghim vào Lịch trình** (lưu `NormalizedPlace` vào session).
- Render lịch trình theo từng loại item (`PLACE`, `ACTIVITY`, `TRANSIT`) kèm badge xác thực rõ ràng.

### Phase 8 — Kiểm Thử & Đánh Giá Toàn Diện (Testing & Validation)
- Viết bộ test `tests/test_hybrid_planner.py` kiểm tra:
  - Intent extraction cho nhiều điểm đến và custom places.
  - Place resolution (`verified` vs `unverified`).
  - Scheduling logic (chuyển tỉnh đúng ngày, không lệch transit).
  - Route optimization (thứ tự khoảng cách hợp lý).
  - Kháng hallucination (không có POI lạ xuất hiện).

---

## 🧪 Kế Hoạch Xác Minh (Verification Plan)

### Automated Tests
```bash
pytest tests/ -v
```
Đảm bảo tất cả 37 test hiện tại tiếp tục pass, cùng với bộ unit test mới cho pipeline hybrid planning.

### Manual Verification Test Case (Scenario Chuyển Tỉnh & Địa Điểm Chỉ Định)
- **Input**: *"Tôi muốn đi Đà Nẵng 2 ngày rồi sang Hội An 1 ngày cho 4 người. Ngày 1 ăn tối Cơm Niêu Nhà Đỏ Đà Nẵng, sang Hội An ghé Bánh Mì Phượng."*
- **Kiểm tra**:
  - `Day 1` & `Day 2` 100% tại Đà Nẵng. Bữa tối Day 1 là Cơm Niêu Nhà Đỏ.
  - `Day 3` bắt đầu bằng `TRANSIT` Đà Nẵng → Hội An.
  - Lịch trình Day 3 có Bánh Mì Phượng tại Hội An.
  - Các slot còn lại là địa điểm thật được tối ưu tuyến đường và các hoạt động `ACTIVITY` hợp lý.
