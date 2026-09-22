"""Destinations Catalog and Metadata for TripMate.

Provides rich metadata, GPS coordinates, popular travel categories,
curated attraction queries, and verified sample places for offline/dev fallback.
"""

from typing import Dict, Any, List, Optional

DESTINATIONS_CATALOG: Dict[str, Dict[str, Any]] = {
    "Đà Nẵng": {
        "id": "danang",
        "name": "Đà Nẵng",
        "country": "Vietnam",
        "latitude": 16.0544,
        "longitude": 108.2022,
        "tags": ["🍜 Ẩm thực miền Trung", "🏖️ Biển Mỹ Khê", "🌉 Cầu Rồng", "⛰️ Ngũ Hành Sơn", "☕ Cafe ven sông"],
        "default_queries": [
            {"category": "restaurant", "query": "nhà hàng hải sản đặc sản Đà Nẵng"},
            {"category": "attraction", "query": "địa điểm tham quan nổi tiếng Đà Nẵng"}
        ],
        "sample_places": [
            {
                "id": "places/danang_madame_lan",
                "name": "Madame Lan Restaurant",
                "address": "04 Bạch Đằng, Thạch Thang, Hải Châu, Đà Nẵng",
                "latitude": 16.0827,
                "longitude": 108.2238,
                "rating": 4.5,
                "user_rating_count": 4200,
                "category": "restaurant"
            },
            {
                "id": "places/danang_bep_cuon",
                "name": "Bếp Cuốn Đà Nẵng",
                "address": "54 Nguyễn Văn Thoại, Ngũ Hành Sơn, Đà Nẵng",
                "latitude": 16.0592,
                "longitude": 108.2415,
                "rating": 4.8,
                "user_rating_count": 2500,
                "category": "restaurant"
            },
            {
                "id": "places/danang_dragon_bridge",
                "name": "Cầu Rồng (Dragon Bridge)",
                "address": "Đường Nguyễn Văn Linh, Phước Ninh, Hải Châu, Đà Nẵng",
                "latitude": 16.0611,
                "longitude": 108.2272,
                "rating": 4.7,
                "user_rating_count": 15600,
                "category": "attraction"
            },
            {
                "id": "places/danang_marble_mountains",
                "name": "Ngũ Hành Sơn (Marble Mountains)",
                "address": "81 Huyền Trân Công Chúa, Hoà Hải, Ngũ Hành Sơn, Đà Nẵng",
                "latitude": 16.0041,
                "longitude": 108.2635,
                "rating": 4.6,
                "user_rating_count": 13800,
                "category": "attraction"
            }
        ]
    },
    "Hà Nội": {
        "id": "hanoi",
        "name": "Hà Nội",
        "country": "Vietnam",
        "latitude": 21.0285,
        "longitude": 105.8542,
        "tags": ["🍜 Phở & Bún chả", "🏰 Phố cổ 36 phố phường", "🏛️ Hồ Gươm & Lăng Bác", "☕ Cà phê trứng"],
        "default_queries": [
            {"category": "restaurant", "query": "nhà hàng đặc sản phố cổ Hà Nội"},
            {"category": "attraction", "query": "di tích và điểm tham quan Hà Nội"}
        ],
        "sample_places": [
            {
                "id": "places/hanoi_bun_cha_huong_lien",
                "name": "Bún Chả Hương Liên (Bún Chả Obama)",
                "address": "24 Lê Văn Hưu, Phan Chu Trinh, Hai Bà Trưng, Hà Nội",
                "latitude": 21.0189,
                "longitude": 105.8532,
                "rating": 4.3,
                "user_rating_count": 6800,
                "category": "restaurant"
            },
            {
                "id": "places/hanoi_giang_cafe",
                "name": "Cafe Giảng (Cà Phê Trứng)",
                "address": "39 Nguyễn Hữu Huân, Hàng Bạc, Hoàn Kiếm, Hà Nội",
                "latitude": 21.0347,
                "longitude": 105.8539,
                "rating": 4.5,
                "user_rating_count": 7900,
                "category": "restaurant"
            },
            {
                "id": "places/hanoi_hoan_kiem_lake",
                "name": "Hồ Hoàn Kiếm & Đền Ngọc Sơn",
                "address": "Đinh Tiên Hoàng, Hàng Trống, Hoàn Kiếm, Hà Nội",
                "latitude": 21.0287,
                "longitude": 105.8524,
                "rating": 4.7,
                "user_rating_count": 21000,
                "category": "attraction"
            },
            {
                "id": "places/hanoi_temple_of_literature",
                "name": "Văn Miếu - Quốc Tử Giám",
                "address": "58 Quốc Tử Giám, Văn Miếu, Đống Đa, Hà Nội",
                "latitude": 21.0278,
                "longitude": 105.8355,
                "rating": 4.6,
                "user_rating_count": 16400,
                "category": "attraction"
            }
        ]
    },
    "TP. Hồ Chí Minh": {
        "id": "hcm",
        "name": "TP. Hồ Chí Minh",
        "country": "Vietnam",
        "latitude": 10.8231,
        "longitude": 106.6297,
        "tags": ["🌆 Sài Gòn Nightlife", "🍢 Ẩm thực đường phố", "🏛️ Dinh Độc Lập", "☕ Cà phê bệt Nhà thờ Đức Bà"],
        "default_queries": [
            {"category": "restaurant", "query": "nhà hàng ẩm thực Sài Gòn quận 1"},
            {"category": "attraction", "query": "điểm du lịch nổi tiếng Sài Gòn"}
        ],
        "sample_places": [
            {
                "id": "places/hcm_cuc_gach_quan",
                "name": "Cục Gạch Quán",
                "address": "10 Đặng Tất, Tân Định, Quận 1, TP. Hồ Chí Minh",
                "latitude": 10.7928,
                "longitude": 106.6896,
                "rating": 4.4,
                "user_rating_count": 3600,
                "category": "restaurant"
            },
            {
                "id": "places/hcm_independence_palace",
                "name": "Dinh Độc Lập (Independence Palace)",
                "address": "135 Nam Kỳ Khởi Nghĩa, Bến Thành, Quận 1, TP. Hồ Chí Minh",
                "latitude": 10.7769,
                "longitude": 106.6954,
                "rating": 4.6,
                "user_rating_count": 28000,
                "category": "attraction"
            },
            {
                "id": "places/hcm_ben_thanh",
                "name": "Chợ Bến Thành",
                "address": "Lê Lợi, Phường Bến Thành, Quận 1, TP. Hồ Chí Minh",
                "latitude": 10.7726,
                "longitude": 106.6980,
                "rating": 4.3,
                "user_rating_count": 32000,
                "category": "attraction"
            }
        ]
    },
    "Phú Quốc": {
        "id": "phuquoc",
        "name": "Phú Quốc",
        "country": "Vietnam",
        "latitude": 10.2899,
        "longitude": 103.9840,
        "tags": ["🏖️ Bãi Sao & Sunset Sanato", "🦀 Hải sản Hàm Ninh", "🤿 Lặn ngắm san hô", "🌅 Hoàng hôn Bãi Trường"],
        "default_queries": [
            {"category": "restaurant", "query": "hải sản tươi sống Phú Quốc"},
            {"category": "attraction", "query": "bãi biển và đảo ngọc Phú Quốc"}
        ],
        "sample_places": [
            {
                "id": "places/pq_xin_chao",
                "name": "Nhà Hàng Xin Chào Phú Quốc",
                "address": "66 Trần Hưng Đạo, Dương Đông, Phú Quốc",
                "latitude": 10.2134,
                "longitude": 103.9572,
                "rating": 4.4,
                "user_rating_count": 3400,
                "category": "restaurant"
            },
            {
                "id": "places/pq_bai_sao",
                "name": "Bãi Sao (Starfish Beach)",
                "address": "Ấp Bãi Sao, An Thới, Phú Quốc",
                "latitude": 10.0543,
                "longitude": 104.0321,
                "rating": 4.5,
                "user_rating_count": 11500,
                "category": "attraction"
            },
            {
                "id": "places/pq_hon_thom",
                "name": "Cáp treo Hòn Thơm Sun World",
                "address": "Bãi Đất Đỏ, An Thới, Phú Quốc",
                "latitude": 10.0264,
                "longitude": 104.0125,
                "rating": 4.7,
                "user_rating_count": 14200,
                "category": "attraction"
            }
        ]
    },
    "Đà Lạt": {
        "id": "dalat",
        "name": "Đà Lạt",
        "country": "Vietnam",
        "latitude": 11.9404,
        "longitude": 108.4583,
        "tags": ["🍓 Vườn dâu & Săn mây", "☕ Cafe đồi thông", "🥖 Bánh mì xíu mại & Lẩu gà lá é", "🌸 Hồ Xuân Hương"],
        "default_queries": [
            {"category": "restaurant", "query": "quán ăn ngon đặc sản Đà Lạt"},
            {"category": "attraction", "query": "địa điểm săn mây ngắm cảnh Đà Lạt"}
        ],
        "sample_places": [
            {
                "id": "places/dalat_lau_ga_tao_ngo",
                "name": "Lẩu Gà Lá É Tao Ngộ",
                "address": "Số 5 Đường 3 Tháng 4, Phường 3, Đà Lạt",
                "latitude": 11.9312,
                "longitude": 108.4451,
                "rating": 4.3,
                "user_rating_count": 4800,
                "category": "restaurant"
            },
            {
                "id": "places/dalat_xuan_huong_lake",
                "name": "Hồ Xuân Hương",
                "address": "Trung tâm Phường 1, Đà Lạt",
                "latitude": 11.9424,
                "longitude": 108.4459,
                "rating": 4.6,
                "user_rating_count": 18500,
                "category": "attraction"
            },
            {
                "id": "places/dalat_langbiang",
                "name": "Đỉnh Langbiang",
                "address": "Thị trấn Lạc Dương, Lạc Dương, Lâm Đồng",
                "latitude": 12.0464,
                "longitude": 108.4357,
                "rating": 4.5,
                "user_rating_count": 13200,
                "category": "attraction"
            }
        ]
    },
    "Hội An": {
        "id": "hoian",
        "name": "Hội An",
        "country": "Vietnam",
        "latitude": 15.8801,
        "longitude": 108.3380,
        "tags": ["🏮 Đèn lồng Phố cổ", "🥖 Bánh mì Phượng", "🍜 Cao lầu & Cơm gà", "🛶 Thả hoa đăng sông Hoài"],
        "default_queries": [
            {"category": "restaurant", "query": "đặc sản cao lầu cơm gà Hội An"},
            {"category": "attraction", "query": "phố cổ Hội An và làng nghề"}
        ],
        "sample_places": [
            {
                "id": "places/hoian_banh_mi_phuong",
                "name": "Bánh Mì Phượng",
                "address": "2B Phan Chu Trinh, Cẩm Châu, Hội An",
                "latitude": 15.8778,
                "longitude": 108.3315,
                "rating": 4.4,
                "user_rating_count": 9200,
                "category": "restaurant"
            },
            {
                "id": "places/hoian_chua_cau",
                "name": "Chùa Cầu (Japanese Covered Bridge)",
                "address": "Nguyễn Thị Minh Khai, Phường Minh An, Hội An",
                "latitude": 15.8771,
                "longitude": 108.3258,
                "rating": 4.5,
                "user_rating_count": 12700,
                "category": "attraction"
            }
        ]
    },
    "Tokyo": {
        "id": "tokyo",
        "name": "Tokyo",
        "country": "Japan",
        "latitude": 35.6762,
        "longitude": 139.6503,
        "tags": ["🍣 Sushi & Ramen", "🗼 Tokyo Tower & Skytree", "🏮 Asakusa Sensoji", "🛍️ Shibuya Crossing"],
        "default_queries": [
            {"category": "restaurant", "query": "best ramen and sushi spots Tokyo"},
            {"category": "attraction", "query": "top landmarks and temples Tokyo"}
        ],
        "sample_places": [
            {
                "id": "places/tokyo_sensoji",
                "name": "Sensō-ji Temple (浅草寺)",
                "address": "2-3-1 Asakusa, Taito City, Tokyo",
                "latitude": 35.7148,
                "longitude": 139.7967,
                "rating": 4.6,
                "user_rating_count": 65000,
                "category": "attraction"
            },
            {
                "id": "places/tokyo_shibuya",
                "name": "Shibuya Scramble Crossing",
                "address": "Shibuya, Tokyo",
                "latitude": 35.6595,
                "longitude": 139.7005,
                "rating": 4.5,
                "user_rating_count": 48000,
                "category": "attraction"
            }
        ]
    },
    "Bangkok": {
        "id": "bangkok",
        "name": "Bangkok",
        "country": "Thailand",
        "latitude": 13.7563,
        "longitude": 100.5018,
        "tags": ["🍜 Pad Thai & Tom Yum", "🛕 Wat Arun & Grand Palace", "🛍️ Chatuchak Market", "⛴️ Chao Phraya Cruise"],
        "default_queries": [
            {"category": "restaurant", "query": "top street food and restaurants Bangkok"},
            {"category": "attraction", "query": "famous temples and palaces Bangkok"}
        ],
        "sample_places": [
            {
                "id": "places/bkk_wat_arun",
                "name": "Wat Arun (Temple of Dawn)",
                "address": "Bangkok Yai, Bangkok",
                "latitude": 13.7437,
                "longitude": 100.4889,
                "rating": 4.7,
                "user_rating_count": 42000,
                "category": "attraction"
            }
        ]
    }
}


def _strip_accents(text: str) -> str:
    """Remove Vietnamese diacritics for flexible fuzzy matching."""
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).replace('đ', 'd').replace('Đ', 'D').lower()


def get_destination_info(name: str) -> Optional[Dict[str, Any]]:
    """Lookup destination info case-insensitively and accent-insensitively."""
    target = name.strip().lower()
    target_clean = _strip_accents(target)

    for key, data in DESTINATIONS_CATALOG.items():
        key_clean = _strip_accents(key)
        name_clean = _strip_accents(data["name"])
        id_val = data["id"].lower()

        if target == key.lower() or target == id_val or target == data["name"].lower():
            return data
        if target_clean == key_clean or target_clean == id_val or target_clean == name_clean:
            return data
        if target_clean in key_clean or key_clean in target_clean:
            return data
    return None


def list_destinations() -> List[Dict[str, Any]]:
    """Return list of supported destination summaries."""
    return [
        {
            "id": v["id"],
            "name": v["name"],
            "country": v["country"],
            "latitude": v["latitude"],
            "longitude": v["longitude"],
            "tags": v["tags"],
        }
        for v in DESTINATIONS_CATALOG.values()
    ]
