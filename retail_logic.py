import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PRODUCTS_CSV = DATA_DIR / "products.csv"
INVENTORY_CSV = DATA_DIR / "inventory.csv"
IMAGE_PROFILES_JSON = DATA_DIR / "image_profiles.json"
INBOUND_SHIPMENTS_CSV = DATA_DIR / "inbound_shipments.csv"
PREFERRED_SIZE = "6"
NEARBY_STORES = ["San Francisco Union Square", "Palo Alto", "San Jose Santana Row"]
FORMAL_OCCASIONS = {"gala", "formal", "black tie", "cocktail"}


@lru_cache(maxsize=1)
def load_products() -> pd.DataFrame:
    return pd.read_csv(PRODUCTS_CSV)


@lru_cache(maxsize=1)
def load_inventory() -> pd.DataFrame:
    return pd.read_csv(INVENTORY_CSV)


@lru_cache(maxsize=1)
def load_image_profiles() -> Dict[str, Any]:
    return json.loads(IMAGE_PROFILES_JSON.read_text())


@lru_cache(maxsize=1)
def load_inbound_shipments() -> pd.DataFrame:
    return pd.read_csv(INBOUND_SHIPMENTS_CSV)


def analyze_uploaded_photo(file_name: str) -> Dict[str, Any]:
    profile = load_image_profiles().get(file_name)
    if not profile:
        return {
            "status": "not_found",
            "message": f"No style profile found for {file_name}.",
        }
    return {
        "status": "success",
        "file_name": file_name,
        **profile,
    }


def _score_product(
    row: pd.Series, category: str, occasion: str, color: str, style_tags: List[str]
) -> int:
    score = 0
    row_tags = [t.strip().lower() for t in str(row["style_tags"]).split(",")]

    if str(row["category"]).lower() == category.lower():
        score += 40
    if str(row["occasion"]).lower() == occasion.lower():
        score += 30
    elif str(row["occasion"]).lower() in FORMAL_OCCASIONS and occasion.lower() in FORMAL_OCCASIONS:
        score += 15
    if str(row["color"]).lower() == color.lower():
        score += 20

    overlap = len(set(t.lower() for t in style_tags).intersection(set(row_tags)))
    score += overlap * 5
    return score


def search_products(category: str, occasion: str, color: str, style_tags: List[str]) -> Dict[str, Any]:
    results = []
    for _, row in load_products().iterrows():
        score = _score_product(row, category, occasion, color, style_tags)
        if score > 0:
            results.append(
                {
                    "product_id": row["product_id"],
                    "product_name": row["product_name"],
                    "category": row["category"],
                    "occasion": row["occasion"],
                    "color": row["color"],
                    "material": row["material"],
                    "price": int(row["price"]),
                    "brand": row["brand"],
                    "image_path": row["image_path"],
                    "size_range": row["size_range"],
                    "match_score": score,
                }
            )
    results = sorted(results, key=lambda x: x["match_score"], reverse=True)
    return {"status": "success", "results": results[:5]}


def get_product_by_id(product_id: str) -> Dict[str, Any]:
    products = load_products()
    product = products[products["product_id"] == product_id]
    if product.empty:
        raise ValueError(f"Unknown product_id: {product_id}")

    row = product.iloc[0]
    return {
        "product_id": row["product_id"],
        "product_name": row["product_name"],
        "category": row["category"],
        "occasion": row["occasion"],
        "color": row["color"],
        "material": row["material"],
        "price": int(row["price"]),
        "brand": row["brand"],
        "image_path": row["image_path"],
        "size_range": row["size_range"],
    }


def live_inventory_available(product_id: str, preferred_size: str, nearby_stores: List[str]) -> bool:
    inv = load_inventory()
    inv = inv[inv["product_id"] == product_id].copy()
    inv = inv[inv["store_name"].isin(nearby_stores)]
    if inv.empty:
        return False

    for _, row in inv.iterrows():
        size_match = str(row["size"]) == str(preferred_size) or str(row["size"]).upper() == "OS"
        if size_match and int(row["quantity"]) > 0:
            return True
    return False


def get_back_office_color_candidates(
    anchor_product: Dict[str, Any], occasion: str, preferred_size: str, nearby_stores: List[str]
) -> List[Dict[str, Any]]:
    candidates = []
    for _, row in load_products().iterrows():
        if row["product_id"] == anchor_product["product_id"]:
            continue
        if str(row["category"]).lower() != str(anchor_product["category"]).lower():
            continue
        if str(row["color"]).lower() == str(anchor_product["color"]).lower():
            continue
        if str(row["occasion"]).lower() != occasion.lower() and not (
            str(row["occasion"]).lower() in FORMAL_OCCASIONS and occasion.lower() in FORMAL_OCCASIONS
        ):
            continue

        product_id = row["product_id"]
        inbound = load_inbound_shipments()
        inbound = inbound[inbound["product_id"] == product_id].copy()
        inbound = inbound[inbound["store_name"].isin(nearby_stores)]
        if inbound.empty or live_inventory_available(product_id, preferred_size, nearby_stores):
            continue

        candidates.append(
            {
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "category": row["category"],
                "occasion": row["occasion"],
                "color": row["color"],
                "material": row["material"],
                "price": int(row["price"]),
                "brand": row["brand"],
                "image_path": row["image_path"],
                "size_range": row["size_range"],
            }
        )

    return sorted(candidates, key=lambda x: abs(x["price"] - anchor_product["price"]))


def check_inventory(product_id: str, preferred_size: str, nearby_stores: List[str]) -> Dict[str, Any]:
    inv = load_inventory()
    inv = inv[inv["product_id"] == product_id].copy()
    inv = inv[inv["store_name"].isin(nearby_stores)]

    def store_rank(name: str) -> int:
        try:
            return nearby_stores.index(name)
        except ValueError:
            return 999

    records = []
    for _, row in inv.iterrows():
        size_match = str(row["size"]) == str(preferred_size) or str(row["size"]).upper() == "OS"
        records.append(
            {
                "store_name": row["store_name"],
                "size": str(row["size"]),
                "quantity": int(row["quantity"]),
                "pickup_available": row["pickup_available"],
                "reserve_eligible": row["reserve_eligible"],
                "preferred_size_match": size_match,
            }
        )

    records = sorted(
        records,
        key=lambda x: (
            0 if x["preferred_size_match"] else 1,
            0 if x["quantity"] > 0 else 1,
            store_rank(x["store_name"]),
        ),
    )

    return {
        "status": "success",
        "product_id": product_id,
        "preferred_size": preferred_size,
        "inventory": records,
    }


def check_back_office_shipments(product_id: str, preferred_size: str, nearby_stores: List[str]) -> Dict[str, Any]:
    inbound = load_inbound_shipments()
    inbound = inbound[inbound["product_id"] == product_id].copy()
    inbound = inbound[inbound["store_name"].isin(nearby_stores)]

    def store_rank(name: str) -> int:
        try:
            return nearby_stores.index(name)
        except ValueError:
            return 999

    def shipment_rank(status: str) -> int:
        if status == "Received in back office":
            return 0
        if status == "In transit from stockroom":
            return 1
        return 2

    records = []
    for _, row in inbound.iterrows():
        size_match = str(row["size"]) == str(preferred_size) or str(row["size"]).upper() == "OS"
        records.append(
            {
                "invoice_id": row["invoice_id"],
                "store_name": row["store_name"],
                "size": str(row["size"]),
                "quantity": int(row["quantity"]),
                "invoice_status": row["invoice_status"],
                "shipment_status": row["shipment_status"],
                "shipped_date": row["shipped_date"],
                "eta_window": row["eta_window"],
                "associate_note": row["associate_note"],
                "preferred_size_match": size_match,
            }
        )

    records = sorted(
        records,
        key=lambda x: (
            0 if x["preferred_size_match"] else 1,
            shipment_rank(x["shipment_status"]),
            store_rank(x["store_name"]),
        ),
    )

    return {
        "status": "success",
        "product_id": product_id,
        "preferred_size": preferred_size,
        "shipments": records,
    }


def reserve_item(product_id: str, store_name: str, size: str, customer_name: str) -> Dict[str, Any]:
    product = load_products()
    product = product[product["product_id"] == product_id].iloc[0]
    return {
        "status": "confirmed",
        "reservation_id": "RES-2048",
        "customer_name": customer_name,
        "product_id": product_id,
        "reserved_item": product["product_name"],
        "store_name": store_name,
        "size": size,
        "pickup_window": "Today before 7:00 PM",
        "associate_note": (
            f"Please hold {product['product_name']} in size {size} for {customer_name}. "
            "Premium event shopper, requested quick pickup."
        ),
    }


def verify_back_office_hold(
    product_id: str, store_name: str, size: str, customer_name: str, invoice_id: str, shipped_date: str
) -> Dict[str, Any]:
    product = load_products()
    product = product[product["product_id"] == product_id].iloc[0]
    return {
        "status": "confirmed",
        "invoice_id": invoice_id,
        "customer_name": customer_name,
        "product_id": product_id,
        "verified_item": product["product_name"],
        "store_name": store_name,
        "size": size,
        "shipped_date": shipped_date,
        "associate_note": (
            f"Verified {product['product_name']} in size {size} in the {store_name} back office "
            f"from invoice {invoice_id}. Prepare it for {customer_name}."
        ),
    }


def build_outfit(anchor_product_id: str, occasion: str) -> Dict[str, Any]:
    products = load_products()
    anchor = products[products["product_id"] == anchor_product_id].iloc[0]
    recommendations = []
    for _, row in products.iterrows():
        if row["product_id"] == anchor_product_id:
            continue
        if row["category"] in ["Shoes", "Bag", "Jewelry", "Accessory"]:
            if str(row["occasion"]).lower() in [occasion.lower(), "black tie", "gala", "formal"]:
                recommendations.append(
                    {
                        "product_id": row["product_id"],
                        "product_name": row["product_name"],
                        "category": row["category"],
                        "price": int(row["price"]),
                        "brand": row["brand"],
                        "image_path": row["image_path"],
                    }
                )
    return {
        "status": "success",
        "anchor_product_id": anchor_product_id,
        "anchor_product_name": anchor["product_name"],
        "recommendations": recommendations[:4],
    }
