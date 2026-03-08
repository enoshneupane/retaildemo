from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from retail_logic import (
    BASE_DIR,
    NEARBY_STORES,
    PREFERRED_SIZE,
    analyze_uploaded_photo,
    build_outfit,
    check_back_office_shipments,
    check_inventory,
    get_back_office_color_candidates,
    reserve_item,
    search_products,
    verify_back_office_hold,
)

st.set_page_config(page_title="RetailNext Luxury Associate Assistant", layout="wide")

WORKFLOW_STATE_KEYS = [
    "workflow_upload_key",
    "analysis_result",
    "search_result",
    "inventory_result",
    "outfit_result",
    "reservation_result",
    "reservation_error",
    "back_office_choice",
    "back_office_product",
    "back_office_result",
    "back_office_verification",
    "back_office_error",
]

def reset_workflow_state() -> None:
    for key in WORKFLOW_STATE_KEYS:
        st.session_state.pop(key, None)

def render_product_card(product: Dict[str, Any], show_score: bool = False) -> None:
    col1, col2 = st.columns([1, 2])
    with col1:
        img_path = BASE_DIR / product["image_path"]
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.warning(f"Missing image: {product['image_path']}")
    with col2:
        st.markdown(f"### {product['product_name']}")
        if "brand" in product:
            st.write(f"**Brand:** {product['brand']}")
        if "category" in product:
            st.write(f"**Category:** {product['category']}")
        if "occasion" in product:
            st.write(f"**Occasion:** {product['occasion']}")
        if "color" in product:
            st.write(f"**Color:** {product['color']}")
        if "material" in product:
            st.write(f"**Material:** {product['material']}")
        st.write(f"**Price:** ${product['price']}")
        if show_score and "match_score" in product:
            st.write(f"**Match score:** {product['match_score']}")

def render_inventory_table(inventory_records: List[Dict[str, Any]]) -> None:
    if not inventory_records:
        st.warning("No inventory records found.")
        return
    inv_df = pd.DataFrame(inventory_records)
    st.dataframe(inv_df, use_container_width=True)

def render_back_office_table(shipment_records: List[Dict[str, Any]]) -> None:
    if not shipment_records:
        st.warning("No invoiced back-office arrivals found.")
        return
    shipment_df = pd.DataFrame(shipment_records)
    st.dataframe(shipment_df, use_container_width=True)

st.title("RetailNext Luxury Associate Assistant")
st.caption("Luxury retail associate workspace")

st.markdown("""
This assistant helps a store associate:
- interpret a customer’s inspiration photo
- find aligned luxury eveningwear
- check nearby store inventory
- reserve the right item for pickup
- complete the outfit with coordinated add-ons
""")

st.markdown("## 1. Upload customer inspiration photo")
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    current_upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("workflow_upload_key") != current_upload_key:
        reset_workflow_state()
        st.session_state["workflow_upload_key"] = current_upload_key

    st.image(uploaded_file, caption="Customer inspiration image", width=320)

    upload_dir = BASE_DIR / "images" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_upload_path = upload_dir / uploaded_file.name
    saved_upload_path.write_bytes(uploaded_file.getbuffer())

    st.success(f"Uploaded: {uploaded_file.name}")

    st.markdown("## 2. Style analysis")
    if st.button("Analyze style", key="analyze_style"):
        st.session_state["analysis_result"] = analyze_uploaded_photo(uploaded_file.name)
        st.session_state.pop("search_result", None)
        st.session_state.pop("inventory_result", None)
        st.session_state.pop("outfit_result", None)
        st.session_state.pop("reservation_result", None)
        st.session_state.pop("reservation_error", None)
        st.session_state.pop("back_office_product", None)
        st.session_state.pop("back_office_result", None)
        st.session_state.pop("back_office_verification", None)
        st.session_state.pop("back_office_error", None)
        st.session_state.pop("back_office_choice", None)

    analysis_result = st.session_state.get("analysis_result")
    if analysis_result and analysis_result.get("status") == "success":
        st.markdown("### Style interpretation")
        st.write(analysis_result["narrative"])
        st.caption(
            f"Detected: {analysis_result['occasion']} | {analysis_result['category']} | "
            f"{analysis_result['color']} | {analysis_result['material']}"
        )

        st.markdown("## 3. Recommended matches")
        if st.button("Find matching products", key="find_matches"):
            st.session_state["search_result"] = search_products(
                category=analysis_result["category"],
                occasion=analysis_result["occasion"],
                color=analysis_result["color"],
                style_tags=analysis_result["style_tags"]
            )
            st.session_state.pop("inventory_result", None)
            st.session_state.pop("outfit_result", None)
            st.session_state.pop("reservation_result", None)
            st.session_state.pop("reservation_error", None)
            st.session_state.pop("back_office_product", None)
            st.session_state.pop("back_office_result", None)
            st.session_state.pop("back_office_verification", None)
            st.session_state.pop("back_office_error", None)
            st.session_state.pop("back_office_choice", None)

        search_result = st.session_state.get("search_result")
        if search_result:
            st.caption(f"Found {len(search_result['results'])} matching products.")

            st.markdown("### Recommended luxury matches")
            for product in search_result["results"][:3]:
                render_product_card(product, show_score=True)
                st.divider()

            if not search_result["results"]:
                st.error("No matching products found for this inspiration photo.")
            else:
                anchor_product = search_result["results"][0]
                back_office_candidates = get_back_office_color_candidates(
                    anchor_product=anchor_product,
                    occasion=analysis_result["occasion"],
                    preferred_size=PREFERRED_SIZE,
                    nearby_stores=NEARBY_STORES
                )

                if back_office_candidates:
                    st.markdown("## 4. Different color request")
                    st.caption(
                        "If the client wants another color that is not yet on the floor, "
                        "check today's invoiced arrivals before saying no."
                    )

                    selected_back_office_color = None
                    if len(back_office_candidates) == 1:
                        selected_back_office_color = back_office_candidates[0]["color"]
                        st.write(
                            f"Customer asks: Do you have this in {selected_back_office_color.lower()}?"
                        )
                    else:
                        selected_back_office_color = st.selectbox(
                            "Customer requested color",
                            options=[product["color"] for product in back_office_candidates],
                            key="back_office_choice",
                            index=None,
                            placeholder="Select a color to check"
                        )

                    if st.button("Check back office and today's invoice", key="check_back_office"):
                        selected_product = next(
                            (
                                product for product in back_office_candidates
                                if product["color"] == selected_back_office_color
                            ),
                            None
                        )

                        if selected_product:
                            st.session_state["back_office_product"] = selected_product
                            st.session_state["back_office_result"] = check_back_office_shipments(
                                product_id=selected_product["product_id"],
                                preferred_size=PREFERRED_SIZE,
                                nearby_stores=NEARBY_STORES
                            )
                            st.session_state.pop("back_office_verification", None)
                            st.session_state.pop("back_office_error", None)
                        else:
                            st.session_state.pop("back_office_product", None)
                            st.session_state.pop("back_office_result", None)
                            st.session_state.pop("back_office_verification", None)
                            st.session_state["back_office_error"] = "Select a color to check."

                    back_office_product = st.session_state.get("back_office_product")
                    back_office_result = st.session_state.get("back_office_result")
                    if back_office_product and back_office_result:
                        st.markdown("### Alternate color located")
                        render_product_card(back_office_product)
                        st.info(
                            "This color is not yet reflected in live floor inventory. "
                            "Today's invoice shows an inbound unit that the associate can verify in the back office."
                        )
                        st.markdown("### Back office / invoice check")
                        render_back_office_table(back_office_result["shipments"])

                        ready_now = [
                            shipment for shipment in back_office_result["shipments"]
                            if shipment["quantity"] > 0
                            and shipment["preferred_size_match"]
                            and shipment["shipment_status"] == "Received in back office"
                        ]

                        if ready_now and st.button("Verify back office availability", key="verify_back_office"):
                            chosen_shipment = ready_now[0]
                            st.session_state["back_office_verification"] = verify_back_office_hold(
                                product_id=back_office_product["product_id"],
                                store_name=chosen_shipment["store_name"],
                                size=chosen_shipment["size"],
                                customer_name="Natasha",
                                invoice_id=chosen_shipment["invoice_id"],
                                shipped_date=chosen_shipment["shipped_date"]
                            )

                        back_office_verification = st.session_state.get("back_office_verification")
                        if back_office_verification:
                            st.success("Back office availability verified")
                            st.write(
                                f"{back_office_verification['verified_item']} in size "
                                f"{back_office_verification['size']} is verified in the "
                                f"{back_office_verification['store_name']} back office from invoice "
                                f"{back_office_verification['invoice_id']} dated {back_office_verification['shipped_date']}."
                            )
                            st.markdown("### Customer update")
                            st.markdown(
                                f"""
We located the **{back_office_verification['verified_item']}** in **size {back_office_verification['size']}**
from today’s inbound shipment and can have it prepared from the
**{back_office_verification['store_name']}** back office.
"""
                            )
                    elif st.session_state.get("back_office_error"):
                        st.error(st.session_state["back_office_error"])

                st.markdown("### Next actions")
                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    if st.button("Check nearby inventory", key="check_inventory"):
                        st.session_state["inventory_result"] = check_inventory(
                            product_id=anchor_product["product_id"],
                            preferred_size=PREFERRED_SIZE,
                            nearby_stores=NEARBY_STORES
                        )
                        st.session_state.pop("reservation_result", None)
                        st.session_state.pop("reservation_error", None)
                with action_col2:
                    if st.button("See styling add-ons", key="build_outfit"):
                        st.session_state["outfit_result"] = build_outfit(
                            anchor_product_id=anchor_product["product_id"],
                            occasion=analysis_result["occasion"]
                        )

                inventory_result = st.session_state.get("inventory_result")
                if inventory_result:
                    st.markdown("## 5. Nearby inventory")
                    st.caption(
                        f"Checking size {inventory_result['preferred_size']} availability for "
                        f"{anchor_product['product_name']}."
                    )
                    render_inventory_table(inventory_result["inventory"])

                outfit_result = st.session_state.get("outfit_result")
                if outfit_result:
                    st.markdown("## 6. Complete the outfit")
                    st.caption(f"Styled around {outfit_result['anchor_product_name']}.")
                    st.markdown("### Suggested add-ons")
                    for accessory in outfit_result["recommendations"]:
                        render_product_card(accessory)
                        st.divider()

                if inventory_result:
                    st.markdown("## 7. Reserve for pickup")
                    reserve_label = "Reserve item"
                    if outfit_result:
                        reserve_label = "Reserve item and prepare outfit"

                    if st.button(reserve_label, key="reserve_item"):
                        valid_options = [
                            r for r in inventory_result["inventory"]
                            if r["quantity"] > 0 and r["reserve_eligible"] == "Yes"
                        ]

                        if valid_options:
                            chosen_store = valid_options[0]

                            st.session_state["reservation_result"] = reserve_item(
                                product_id=anchor_product["product_id"],
                                store_name=chosen_store["store_name"],
                                size=PREFERRED_SIZE,
                                customer_name="Natasha"
                            )
                            st.session_state.pop("reservation_error", None)
                        else:
                            st.session_state.pop("reservation_result", None)
                            st.session_state["reservation_error"] = (
                                "No reservable inventory available for this item."
                            )

                    reservation_result = st.session_state.get("reservation_result")
                    if reservation_result:
                        st.success("Reservation confirmed")
                        st.write(
                            f"Reservation {reservation_result['reservation_id']} is ready for "
                            f"{reservation_result['customer_name']} at {reservation_result['store_name']}."
                        )

                        add_on_line = (
                            "We’ve also prepared matching accessories for you to review in store, including heels and an evening clutch."
                            if outfit_result
                            else "We can also review complementary accessories with you in store if you'd like."
                        )

                        st.markdown("### Customer follow-up draft")
                        st.markdown(
                            f"""
Hi Natasha,

We’ve reserved the **{reservation_result['reserved_item']}** in **size {reservation_result['size']}**
at our **{reservation_result['store_name']}** location for pickup **{reservation_result['pickup_window']}**.

{add_on_line}

Best,  
RetailNext Styling Team
"""
                        )

                        st.markdown("### Associate summary")
                        if outfit_result:
                            st.write(
                                "The associate matched the look, secured nearby inventory, and added styling recommendations to support a higher-value appointment."
                            )
                        else:
                            st.write(
                                "The associate matched the look and secured nearby inventory quickly, keeping the experience focused and efficient."
                            )
                    elif st.session_state.get("reservation_error"):
                        st.error(st.session_state["reservation_error"])
    elif analysis_result:
        st.error(analysis_result["message"])
else:
    reset_workflow_state()
    st.info("Upload `gala_inspiration.jpg` to begin styling.")
