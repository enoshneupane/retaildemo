import os
from typing import Any

from fastmcp import FastMCP
import uvicorn
from starlette.responses import JSONResponse

from retail_logic import (
    NEARBY_STORES,
    PREFERRED_SIZE,
    analyze_uploaded_photo,
    build_outfit,
    check_back_office_shipments,
    check_inventory,
    get_back_office_color_candidates,
    get_product_by_id,
    match_products_from_style_brief,
    reserve_item,
    search_products,
    verify_back_office_hold,
)

MCP_PATH = os.getenv("MCP_PATH", "/mcp")
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("PORT", os.getenv("MCP_PORT", "8001")))

READ_ONLY_TOOL = {
    "readOnlyHint": True,
    "openWorldHint": False,
}

WRITE_TOOL = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": False,
}

mcp = FastMCP(
    name="RetailNext Luxury Assistant",
    instructions=(
        "Use these tools to help a luxury retail associate interpret a customer's inspiration image, "
        "find matching eveningwear, check nearby inventory, verify alternate colors from today's inbound "
        "back-office shipments, and place a store pickup hold. If the user uploads an image directly in "
        "ChatGPT, inspect the image yourself and call find_matching_products_from_chat_image with the "
        "dress attributes you observe. Use the filename-based image tools only for backend-known demo files "
        "such as gala_inspiration.jpg. Guide the associate one step at a time: first the dress match, then "
        "inventory or alternate color, then styling add-ons, then hold preparation if requested."
    ),
)


def _stores(nearby_stores: list[str] | None) -> list[str]:
    return nearby_stores or list(NEARBY_STORES)


@mcp.tool(
    description="Analyze the uploaded inspiration image by filename.",
    annotations=READ_ONLY_TOOL,
)
def analyze_inspiration_photo(file_name: str = "gala_inspiration.jpg") -> dict[str, Any]:
    return analyze_uploaded_photo(file_name)


@mcp.tool(
    description=(
        "Analyze a known inspiration image filename and return the best matching products from the catalog."
    ),
    annotations=READ_ONLY_TOOL,
)
def find_matching_products_from_photo(file_name: str = "gala_inspiration.jpg") -> dict[str, Any]:
    analysis = analyze_uploaded_photo(file_name)
    if analysis.get("status") != "success":
        return analysis

    matches = search_products(
        category=analysis["category"],
        occasion=analysis["occasion"],
        color=analysis["color"],
        style_tags=analysis["style_tags"],
    )
    return {
        "status": "success",
        "analysis": analysis,
        "matches": matches["results"],
    }


@mcp.tool(
    description=(
        "Use this when the user uploaded a dress image directly in ChatGPT. Inspect the image yourself, "
        "extract the dress attributes, and pass them here because the MCP server cannot read chat "
        "attachments directly."
    ),
    annotations=READ_ONLY_TOOL,
)
def find_matching_products_from_chat_image(
    category: str,
    occasion: str,
    color: str,
    style_tags: list[str],
    visual_summary: str | None = None,
) -> dict[str, Any]:
    return match_products_from_style_brief(
        category=category,
        occasion=occasion,
        color=color,
        style_tags=style_tags,
        visual_summary=visual_summary,
    )


@mcp.tool(
    description="Search the product catalog using structured styling attributes.",
    annotations=READ_ONLY_TOOL,
)
def search_products_by_attributes(
    category: str, occasion: str, color: str, style_tags: list[str]
) -> dict[str, Any]:
    return search_products(
        category=category,
        occasion=occasion,
        color=color,
        style_tags=style_tags,
    )


@mcp.tool(
    description="Check nearby store inventory for a product and preferred size.",
    annotations=READ_ONLY_TOOL,
)
def check_store_inventory(
    product_id: str,
    preferred_size: str = PREFERRED_SIZE,
    nearby_stores: list[str] | None = None,
) -> dict[str, Any]:
    return check_inventory(
        product_id=product_id,
        preferred_size=preferred_size,
        nearby_stores=_stores(nearby_stores),
    )


@mcp.tool(
    description=(
        "Find alternate color options for the matched product that are not yet on the floor but were invoiced "
        "and shipped today, with back-office shipment details."
    ),
    annotations=READ_ONLY_TOOL,
)
def check_alternate_color_back_office(
    anchor_product_id: str,
    occasion: str,
    requested_color: str | None = None,
    preferred_size: str = PREFERRED_SIZE,
    nearby_stores: list[str] | None = None,
) -> dict[str, Any]:
    stores = _stores(nearby_stores)
    anchor_product = get_product_by_id(anchor_product_id)
    candidates = get_back_office_color_candidates(
        anchor_product=anchor_product,
        occasion=occasion,
        preferred_size=preferred_size,
        nearby_stores=stores,
    )

    if requested_color:
        candidates = [
            candidate
            for candidate in candidates
            if candidate["color"].lower() == requested_color.lower()
        ]

    if not candidates:
        return {
            "status": "not_found",
            "anchor_product": anchor_product,
            "requested_color": requested_color,
            "message": "No invoiced alternate-color back-office units found for this request.",
        }

    return {
        "status": "success",
        "anchor_product": anchor_product,
        "requested_color": requested_color,
        "alternates": [
            {
                **candidate,
                "back_office_shipments": check_back_office_shipments(
                    product_id=candidate["product_id"],
                    preferred_size=preferred_size,
                    nearby_stores=stores,
                )["shipments"],
            }
            for candidate in candidates
        ],
    }


@mcp.tool(
    description="Build complementary accessories around a selected anchor product.",
    annotations=READ_ONLY_TOOL,
)
def build_outfit_recommendations(anchor_product_id: str, occasion: str) -> dict[str, Any]:
    return build_outfit(anchor_product_id=anchor_product_id, occasion=occasion)


@mcp.tool(
    description="Create a store pickup reservation for an in-stock item.",
    annotations=WRITE_TOOL,
)
def reserve_store_pickup(
    product_id: str,
    store_name: str,
    size: str,
    customer_name: str,
) -> dict[str, Any]:
    return reserve_item(
        product_id=product_id,
        store_name=store_name,
        size=size,
        customer_name=customer_name,
    )


@mcp.tool(
    description=(
        "Verify that an alternate-color item from today's invoice is physically available in the back office "
        "and prepare it for the customer."
    ),
    annotations=WRITE_TOOL,
)
def verify_back_office_availability(
    product_id: str,
    store_name: str,
    size: str,
    customer_name: str,
    invoice_id: str,
    shipped_date: str,
) -> dict[str, Any]:
    return verify_back_office_hold(
        product_id=product_id,
        store_name=store_name,
        size=size,
        customer_name=customer_name,
        invoice_id=invoice_id,
        shipped_date=shipped_date,
    )


mcp_app = mcp.http_app(path=MCP_PATH, transport="streamable-http")


async def root(_: Any) -> JSONResponse:
    return JSONResponse(
        {
            "service": "RetailNext Luxury Assistant MCP",
            "status": "ok",
            "mcp_path": MCP_PATH,
            "health_path": "/health",
        }
    )


async def health(_: Any) -> JSONResponse:
    return JSONResponse({"status": "ok"})


app = mcp_app
app.add_route("/", root, methods=["GET"])
app.add_route("/health", health, methods=["GET"])


if __name__ == "__main__":
    uvicorn.run(app, host=MCP_HOST, port=MCP_PORT)
