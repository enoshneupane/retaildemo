import os
from functools import lru_cache
from io import BytesIO
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.apps import AppConfig, ResourceCSP
from fastmcp.tools.tool import ToolResult
from fastmcp.utilities.types import Image
from mcp.types import TextContent
from PIL import Image as PILImage
import uvicorn
from starlette.staticfiles import StaticFiles
from starlette.responses import JSONResponse

from retail_logic import (
    BASE_DIR,
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
PUBLIC_BASE_URL = (
    os.getenv("PUBLIC_BASE_URL")
    or os.getenv("RENDER_EXTERNAL_URL")
    or f"http://127.0.0.1:{MCP_PORT}"
).rstrip("/")
MATCH_CAROUSEL_URI = "ui://retailnext/match-carousel.html"
OPENAI_TOOL_UI_META = {
    "openai/outputTemplate": MATCH_CAROUSEL_URI,
    "openai/toolInvocation/invoking": "Finding top dress matches",
    "openai/toolInvocation/invoked": "Top dress matches ready",
}
OPENAI_RESOURCE_UI_META = {
    "openai/widgetDescription": "Shows the top 3 dress matches as swipeable cards with images and key details.",
    "openai/widgetPrefersBorder": True,
    "openai/widgetCSP": {
        "resource_domains": [PUBLIC_BASE_URL, "https://cdn.jsdelivr.net"],
    },
}

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


def _image_url(image_path: str | None) -> str | None:
    if not image_path:
        return None
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return image_path
    return f"{PUBLIC_BASE_URL}/{image_path.lstrip('/')}"


def _product_with_image_url(product: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(product)
    enriched["image_url"] = _image_url(product.get("image_path"))
    return enriched


def _local_image_path(image_path: str | None) -> str | None:
    if not image_path:
        return None
    candidate = BASE_DIR / image_path
    if candidate.exists():
        return str(candidate)
    return None


@lru_cache(maxsize=16)
def _thumbnail_bytes(image_path: str) -> bytes:
    with PILImage.open(image_path) as img:
        working = img.convert("RGB")
        working.thumbnail((720, 900))
        buffer = BytesIO()
        working.save(buffer, format="JPEG", quality=74, optimize=True)
        return buffer.getvalue()


def _image_block(match: dict[str, Any]) -> Any | None:
    local_path = _local_image_path(match.get("image_path"))
    if not local_path:
        return None
    return Image(
        data=_thumbnail_bytes(local_path),
        format="jpeg",
    ).to_image_content()


def _match_reason(analysis: dict[str, Any], product: dict[str, Any]) -> str:
    reasons = []

    if str(product.get("color", "")).lower() == str(analysis.get("color", "")).lower():
        reasons.append(f"matches the {analysis['color'].lower()} palette")
    if str(product.get("occasion", "")).lower() == str(analysis.get("occasion", "")).lower():
        reasons.append(f"fits the {analysis['occasion'].lower()} dress code")
    if str(product.get("category", "")).lower() == str(analysis.get("category", "")).lower():
        reasons.append("keeps the same dress silhouette")

    style_tags = [str(tag).lower() for tag in analysis.get("style_tags", [])]
    product_tags = {
        tag.strip().lower()
        for tag in str(product.get("style_tags", "")).split(",")
        if tag.strip()
    }
    shared_tags = [tag for tag in style_tags if tag in product_tags]
    if shared_tags:
        reasons.append(f"shares the {', '.join(shared_tags[:2])} styling cues")

    if not reasons:
        reasons.append("stays close to the inspiration's formal evening direction")

    return ", ".join(reasons[:3]).capitalize() + "."


def _carousel_matches(analysis: dict[str, Any], matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards = []
    for rank, match in enumerate(matches[:3], start=1):
        enriched = _product_with_image_url(match)
        cards.append(
            {
                **enriched,
                "rank": rank,
                "match_reason": _match_reason(analysis, match),
            }
        )
    return cards


def _match_summary(analysis: dict[str, Any], matches: list[dict[str, Any]]) -> str:
    if not matches:
        return "No close dress matches found."

    top = matches[0]
    return (
        f"Top dress match: {top['product_name']} by {top['brand']} at ${top['price']}. "
        f"It aligns with the {analysis['color'].lower()} {analysis['occasion'].lower()} direction."
    )


def _match_tool_result(analysis: dict[str, Any], matches: list[dict[str, Any]]) -> ToolResult:
    cards = _carousel_matches(analysis, matches)
    content: list[Any] = [TextContent(type="text", text=_match_summary(analysis, cards))]

    for match in cards:
        content.append(
            TextContent(
                type="text",
                text=(
                    f"Top {match['rank']}: {match['product_name']} by {match['brand']} | "
                    f"${match['price']} | {match['color']} | {match['occasion']}. "
                    f"{match['match_reason']}"
                ),
            )
        )
        image_block = _image_block(match)
        if image_block is not None:
            content.append(image_block)

    return ToolResult(
        content=content,
        structured_content={
            "analysis": analysis,
            "matches": cards,
        },
    )


@mcp.resource(
    MATCH_CAROUSEL_URI,
    mime_type="text/html",
    app=AppConfig(
        csp=ResourceCSP(
            resource_domains=[PUBLIC_BASE_URL, "https://cdn.jsdelivr.net"],
        ),
        prefers_border=True,
    ),
    meta=OPENAI_RESOURCE_UI_META,
)
def match_carousel_resource() -> str:
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>RetailNext Match Carousel</title>
    <style>
      :root {
        color-scheme: light dark;
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      body {
        margin: 0;
        background: transparent;
        color: var(--color-text-primary, #18181b);
      }
      .shell {
        border: 1px solid var(--color-border, rgba(24, 24, 27, 0.12));
        border-radius: 24px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(248, 246, 243, 0.98));
        box-shadow: 0 24px 60px rgba(17, 24, 39, 0.08);
        overflow: hidden;
      }
      .header {
        padding: 18px 20px 12px;
      }
      .eyebrow {
        font-size: 11px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--color-text-secondary, rgba(24, 24, 27, 0.6));
      }
      .title {
        margin-top: 6px;
        font-size: 18px;
        font-weight: 600;
      }
      .summary {
        margin-top: 6px;
        font-size: 13px;
        line-height: 1.45;
        color: var(--color-text-secondary, rgba(24, 24, 27, 0.72));
      }
      .carousel {
        padding: 0 16px 16px;
      }
      .frame {
        position: relative;
        overflow: hidden;
        border-radius: 20px;
      }
      .track {
        display: flex;
        transition: transform 240ms ease;
      }
      .card {
        min-width: 100%;
        box-sizing: border-box;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(24, 24, 27, 0.08);
        border-radius: 20px;
        overflow: hidden;
      }
      .image-wrap {
        aspect-ratio: 4 / 5;
        background: #ece7df;
      }
      .image-wrap img {
        display: block;
        width: 100%;
        height: 100%;
        object-fit: cover;
      }
      .info {
        padding: 16px;
      }
      .rank {
        display: inline-flex;
        align-items: center;
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(17, 24, 39, 0.06);
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .name {
        margin-top: 10px;
        font-size: 18px;
        font-weight: 600;
      }
      .brand {
        margin-top: 4px;
        font-size: 13px;
        color: var(--color-text-secondary, rgba(24, 24, 27, 0.7));
      }
      .meta {
        margin-top: 12px;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
      }
      .meta-block {
        padding: 10px 12px;
        border-radius: 14px;
        background: rgba(17, 24, 39, 0.04);
      }
      .meta-label {
        display: block;
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--color-text-secondary, rgba(24, 24, 27, 0.58));
      }
      .meta-value {
        display: block;
        margin-top: 4px;
        font-size: 13px;
        font-weight: 600;
      }
      .reason {
        margin-top: 12px;
        font-size: 13px;
        line-height: 1.5;
        color: var(--color-text-secondary, rgba(24, 24, 27, 0.78));
      }
      .controls {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-top: 14px;
      }
      .buttons {
        display: flex;
        gap: 8px;
      }
      button {
        appearance: none;
        border: none;
        border-radius: 999px;
        padding: 10px 14px;
        background: rgba(17, 24, 39, 0.08);
        color: inherit;
        font: inherit;
      }
      button:disabled {
        opacity: 0.35;
      }
      .dots {
        display: flex;
        gap: 8px;
      }
      .dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: rgba(17, 24, 39, 0.18);
      }
      .dot.active {
        background: rgba(17, 24, 39, 0.72);
      }
      .empty {
        padding: 20px;
        font-size: 14px;
        color: var(--color-text-secondary, rgba(24, 24, 27, 0.72));
      }
      @media (max-width: 640px) {
        .meta {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <div class="header">
        <div class="eyebrow">RetailNext Match Edit</div>
        <div class="title" id="title">Waiting for dress matches</div>
        <div class="summary" id="summary">The top dress matches will appear here with image cards.</div>
      </div>
      <div class="carousel">
        <div class="frame">
          <div class="track" id="track"></div>
        </div>
        <div class="empty" id="empty">No dress cards yet.</div>
        <div class="controls" id="controls" hidden>
          <div class="dots" id="dots"></div>
          <div class="buttons">
            <button id="prev" type="button">Previous</button>
            <button id="next" type="button">Next</button>
          </div>
        </div>
      </div>
    </div>
    <script type="module">
      import { App } from "https://cdn.jsdelivr.net/npm/@modelcontextprotocol/ext-apps@1.1.2/+esm";

      const state = {
        activeIndex: 0,
        matches: [],
        analysis: null,
      };
      const track = document.getElementById("track");
      const dots = document.getElementById("dots");
      const controls = document.getElementById("controls");
      const empty = document.getElementById("empty");
      const title = document.getElementById("title");
      const summary = document.getElementById("summary");
      const prev = document.getElementById("prev");
      const next = document.getElementById("next");
      const app = new App({
        name: "RetailNext Match Carousel",
        version: "1.0.0",
      });

      function render() {
        const total = state.matches.length;
        if (!total) {
          track.style.transform = "translateX(0)";
          track.innerHTML = "";
          dots.innerHTML = "";
          controls.hidden = true;
          empty.hidden = false;
          title.textContent = "Waiting for dress matches";
          summary.textContent = "The top dress matches will appear here with image cards.";
          return;
        }

        controls.hidden = false;
        empty.hidden = true;
        title.textContent = "Recommended gala dress matches";
        const analysis = state.analysis || {};
        summary.textContent = [
          analysis.color ? analysis.color + " palette" : null,
          analysis.occasion ? analysis.occasion + " dress code" : null,
          Array.isArray(analysis.style_tags) && analysis.style_tags.length
            ? analysis.style_tags.slice(0, 3).join(", ")
            : null,
        ].filter(Boolean).join(" • ");

        track.innerHTML = state.matches.map((match) => `
          <article class="card">
            <div class="image-wrap">
              <img src="${match.image_url || ""}" alt="${match.product_name}" loading="eager">
            </div>
            <div class="info">
              <span class="rank">Top Match ${match.rank}</span>
              <div class="name">${match.product_name}</div>
              <div class="brand">${match.brand}</div>
              <div class="meta">
                <div class="meta-block">
                  <span class="meta-label">Price</span>
                  <span class="meta-value">$${match.price}</span>
                </div>
                <div class="meta-block">
                  <span class="meta-label">Color</span>
                  <span class="meta-value">${match.color}</span>
                </div>
                <div class="meta-block">
                  <span class="meta-label">Occasion</span>
                  <span class="meta-value">${match.occasion}</span>
                </div>
                <div class="meta-block">
                  <span class="meta-label">Material</span>
                  <span class="meta-value">${match.material}</span>
                </div>
              </div>
              <div class="reason">${match.match_reason || ""}</div>
            </div>
          </article>
        `).join("");

        dots.innerHTML = state.matches.map((_, index) => `
          <span class="dot${index === state.activeIndex ? " active" : ""}"></span>
        `).join("");

        prev.disabled = state.activeIndex === 0;
        next.disabled = state.activeIndex === total - 1;
        track.style.transform = `translateX(-${state.activeIndex * 100}%)`;
      }

      function updateResult(params) {
        const structured = params && params.structuredContent ? params.structuredContent : {};
        state.analysis = structured.analysis || null;
        state.matches = Array.isArray(structured.matches) ? structured.matches : [];
        state.activeIndex = 0;
        render();
      }

      prev.addEventListener("click", () => {
        if (state.activeIndex > 0) {
          state.activeIndex -= 1;
          render();
        }
      });

      next.addEventListener("click", () => {
        if (state.activeIndex < state.matches.length - 1) {
          state.activeIndex += 1;
          render();
        }
      });

      app.ontoolresult = (params) => {
        updateResult(params || {});
      };

      async function init() {
        await app.connect();
      }

      init().catch((error) => {
        title.textContent = "Unable to initialize match carousel";
        summary.textContent = error && error.message ? error.message : "The embedded app could not start.";
      });
    </script>
  </body>
</html>
"""


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
    app=AppConfig(resource_uri=MATCH_CAROUSEL_URI, prefers_border=True),
    meta=OPENAI_TOOL_UI_META,
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
    return _match_tool_result(analysis, matches["results"])


@mcp.tool(
    description=(
        "Use this when the user uploaded a dress image directly in ChatGPT. Inspect the image yourself, "
        "extract the dress attributes, and pass them here because the MCP server cannot read chat "
        "attachments directly."
    ),
    annotations=READ_ONLY_TOOL,
    app=AppConfig(resource_uri=MATCH_CAROUSEL_URI, prefers_border=True),
    meta=OPENAI_TOOL_UI_META,
)
def find_matching_products_from_chat_image(
    category: str,
    occasion: str,
    color: str,
    style_tags: list[str],
    visual_summary: str | None = None,
) -> dict[str, Any]:
    result = match_products_from_style_brief(
        category=category,
        occasion=occasion,
        color=color,
        style_tags=style_tags,
        visual_summary=visual_summary,
    )
    return _match_tool_result(result["analysis"], result["matches"])


@mcp.tool(
    description="Search the product catalog using structured styling attributes.",
    annotations=READ_ONLY_TOOL,
    app=AppConfig(resource_uri=MATCH_CAROUSEL_URI, prefers_border=True),
    meta=OPENAI_TOOL_UI_META,
)
def search_products_by_attributes(
    category: str, occasion: str, color: str, style_tags: list[str]
) -> dict[str, Any]:
    result = search_products(
        category=category,
        occasion=occasion,
        color=color,
        style_tags=style_tags,
    )
    analysis = {
        "category": category,
        "occasion": occasion,
        "color": color,
        "style_tags": style_tags,
    }
    return _match_tool_result(analysis, result["results"])


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
app.mount("/images", StaticFiles(directory=str(BASE_DIR / "images")), name="images")
app.add_route("/", root, methods=["GET"])
app.add_route("/health", health, methods=["GET"])


if __name__ == "__main__":
    uvicorn.run(app, host=MCP_HOST, port=MCP_PORT)
