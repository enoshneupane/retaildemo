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
from starlette.requests import Request
from starlette.staticfiles import StaticFiles
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse

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


def _products_with_image_urls(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_product_with_image_url(product) for product in products]


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


def _match_response(analysis: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "success",
        "analysis": analysis,
        "matches": _carousel_matches(analysis, matches),
    }


def _inventory_response(result: dict[str, Any]) -> dict[str, Any]:
    return {
        **result,
        "product": _product_with_image_url(get_product_by_id(result["product_id"])),
    }


def _alternate_color_response(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "success":
        return result

    alternates = []
    for alternate in result["alternates"]:
        enriched = _product_with_image_url(alternate)
        alternates.append(enriched)

    return {
        **result,
        "anchor_product": _product_with_image_url(result["anchor_product"]),
        "alternates": alternates,
    }


def _outfit_response(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "success":
        return result

    return {
        **result,
        "anchor_product": _product_with_image_url(get_product_by_id(result["anchor_product_id"])),
        "recommendations": _products_with_image_urls(result["recommendations"]),
    }


async def _json_payload(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _bad_request(message: str) -> JSONResponse:
    return JSONResponse({"status": "error", "message": message}, status_code=400)


def _openapi_spec(base_url: str) -> dict[str, Any]:
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "RetailNext Luxury Assistant Actions API",
            "version": "1.0.0",
            "description": (
                "API for a luxury retail associate workflow: match a customer inspiration look to the top 3 dresses, "
                "check nearby inventory, verify alternate colors from inbound shipments, suggest styling add-ons, "
                "and prepare pickup holds."
            ),
        },
        "servers": [{"url": base_url}],
        "paths": {
            "/api/match-from-style-brief": {
                "post": {
                    "operationId": "matchFromStyleBrief",
                    "summary": "Match top 3 dresses from a style brief",
                    "description": (
                        "Use after visually inspecting the customer's uploaded dress image. Always return the top 3 dress matches."
                    ),
                    "x-openai-isConsequential": False,
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/MatchFromStyleBriefRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Top 3 dress matches",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/MatchResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/api/match-from-demo-photo": {
                "post": {
                    "operationId": "matchFromDemoPhoto",
                    "summary": "Run the built-in gala demo photo flow",
                    "description": "Use the known demo filename such as gala_inspiration.jpg and return the top 3 dress matches.",
                    "x-openai-isConsequential": False,
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/MatchFromDemoPhotoRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Top 3 dress matches from the demo photo",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/MatchResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/api/check-inventory": {
                "post": {
                    "operationId": "checkInventory",
                    "summary": "Check nearby store inventory",
                    "description": "Check nearby live inventory for a selected dress and preferred size.",
                    "x-openai-isConsequential": False,
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CheckInventoryRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Inventory results",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/InventoryResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/api/check-alternate-color": {
                "post": {
                    "operationId": "checkAlternateColor",
                    "summary": "Check back-office alternate colors",
                    "description": (
                        "Check whether another dress color was invoiced and shipped today even if it is not yet on the sales floor."
                    ),
                    "x-openai-isConsequential": False,
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CheckAlternateColorRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Alternate color and back-office shipment details",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AlternateColorResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/api/build-outfit": {
                "post": {
                    "operationId": "buildOutfit",
                    "summary": "Suggest complementary add-ons",
                    "description": "Recommend accessories that complement the selected dress.",
                    "x-openai-isConsequential": False,
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/BuildOutfitRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Accessory recommendations",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/BuildOutfitResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/api/reserve-pickup": {
                "post": {
                    "operationId": "reservePickup",
                    "summary": "Reserve an in-stock item for pickup",
                    "description": "Create a store pickup hold for a confirmed in-stock item.",
                    "x-openai-isConsequential": True,
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ReservePickupRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Reservation confirmation",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ConfirmationResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/api/verify-back-office": {
                "post": {
                    "operationId": "verifyBackOffice",
                    "summary": "Verify an inbound back-office item",
                    "description": "Confirm a just-shipped alternate color is in the back office and prepare it for the customer.",
                    "x-openai-isConsequential": True,
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/VerifyBackOfficeRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Verification confirmation",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ConfirmationResponse"}
                                }
                            },
                        }
                    },
                }
            },
        },
        "components": {
            "schemas": {
                "MatchFromStyleBriefRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["category", "occasion", "color", "style_tags"],
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Observed dress category, for example gown, evening dress, or cocktail dress.",
                        },
                        "occasion": {
                            "type": "string",
                            "description": "Observed event context, for example gala, formal, black tie, or cocktail.",
                        },
                        "color": {
                            "type": "string",
                            "description": "Primary observed color such as black, emerald, silver, or champagne.",
                        },
                        "style_tags": {
                            "type": "array",
                            "description": "Short style cues observed from the dress image.",
                            "items": {"type": "string"},
                        },
                        "visual_summary": {
                            "type": "string",
                            "description": "One concise summary sentence describing the uploaded dress image.",
                        },
                    },
                },
                "MatchFromDemoPhotoRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "Known demo photo filename. Defaults to gala_inspiration.jpg.",
                            "default": "gala_inspiration.jpg",
                        }
                    },
                },
                "CheckInventoryRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["product_id"],
                    "properties": {
                        "product_id": {"type": "string"},
                        "preferred_size": {"type": "string", "default": "6"},
                        "nearby_stores": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional nearby stores in priority order.",
                        },
                    },
                },
                "CheckAlternateColorRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["anchor_product_id", "occasion"],
                    "properties": {
                        "anchor_product_id": {"type": "string"},
                        "occasion": {"type": "string"},
                        "requested_color": {"type": "string"},
                        "preferred_size": {"type": "string", "default": "6"},
                        "nearby_stores": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
                "BuildOutfitRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["anchor_product_id", "occasion"],
                    "properties": {
                        "anchor_product_id": {"type": "string"},
                        "occasion": {"type": "string"},
                    },
                },
                "ReservePickupRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["product_id", "store_name", "size", "customer_name"],
                    "properties": {
                        "product_id": {"type": "string"},
                        "store_name": {"type": "string"},
                        "size": {"type": "string"},
                        "customer_name": {"type": "string"},
                    },
                },
                "VerifyBackOfficeRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "product_id",
                        "store_name",
                        "size",
                        "customer_name",
                        "invoice_id",
                        "shipped_date",
                    ],
                    "properties": {
                        "product_id": {"type": "string"},
                        "store_name": {"type": "string"},
                        "size": {"type": "string"},
                        "customer_name": {"type": "string"},
                        "invoice_id": {"type": "string"},
                        "shipped_date": {"type": "string"},
                    },
                },
                "ProductMatch": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "product_id": {"type": "string"},
                        "product_name": {"type": "string"},
                        "brand": {"type": "string"},
                        "price": {"type": "integer"},
                        "color": {"type": "string"},
                        "occasion": {"type": "string"},
                        "material": {"type": "string"},
                        "image_url": {"type": "string"},
                        "match_reason": {"type": "string"},
                        "rank": {"type": "integer"},
                    },
                },
                "MatchResponse": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "status": {"type": "string"},
                        "analysis": {"type": "object", "additionalProperties": True},
                        "matches": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/ProductMatch"},
                        },
                    },
                },
                "InventoryRecord": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "store_name": {"type": "string"},
                        "size": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "pickup_available": {"type": "boolean"},
                        "reserve_eligible": {"type": "boolean"},
                        "preferred_size_match": {"type": "boolean"},
                    },
                },
                "InventoryResponse": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "status": {"type": "string"},
                        "product_id": {"type": "string"},
                        "preferred_size": {"type": "string"},
                        "product": {"type": "object", "additionalProperties": True},
                        "inventory": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/InventoryRecord"},
                        },
                    },
                },
                "ShipmentRecord": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "invoice_id": {"type": "string"},
                        "store_name": {"type": "string"},
                        "size": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "invoice_status": {"type": "string"},
                        "shipment_status": {"type": "string"},
                        "shipped_date": {"type": "string"},
                        "eta_window": {"type": "string"},
                        "associate_note": {"type": "string"},
                    },
                },
                "AlternateProduct": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "product_id": {"type": "string"},
                        "product_name": {"type": "string"},
                        "brand": {"type": "string"},
                        "price": {"type": "integer"},
                        "color": {"type": "string"},
                        "image_url": {"type": "string"},
                        "back_office_shipments": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/ShipmentRecord"},
                        },
                    },
                },
                "AlternateColorResponse": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "status": {"type": "string"},
                        "requested_color": {"type": "string"},
                        "anchor_product": {"type": "object", "additionalProperties": True},
                        "alternates": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/AlternateProduct"},
                        },
                    },
                },
                "BuildOutfitResponse": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "status": {"type": "string"},
                        "anchor_product_id": {"type": "string"},
                        "anchor_product_name": {"type": "string"},
                        "anchor_product": {"type": "object", "additionalProperties": True},
                        "recommendations": {
                            "type": "array",
                            "items": {"type": "object", "additionalProperties": True},
                        },
                    },
                },
                "ConfirmationResponse": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "status": {"type": "string"},
                        "reservation_id": {"type": "string"},
                        "invoice_id": {"type": "string"},
                        "customer_name": {"type": "string"},
                        "product_id": {"type": "string"},
                        "reserved_item": {"type": "string"},
                        "verified_item": {"type": "string"},
                        "store_name": {"type": "string"},
                        "size": {"type": "string"},
                        "pickup_window": {"type": "string"},
                        "associate_note": {"type": "string"},
                    },
                },
            }
        },
    }


PRIVACY_POLICY_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>RetailNext Privacy Policy</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px auto; max-width: 720px; padding: 0 20px; line-height: 1.6; color: #111827; }
      h1, h2 { line-height: 1.2; }
    </style>
  </head>
  <body>
    <h1>RetailNext Luxury Assistant Privacy Policy</h1>
    <p>This service is used to support a luxury retail styling workflow. It accepts styling attributes and retail workflow inputs in order to return dress matches, inventory checks, alternate-color shipment lookups, styling recommendations, and reservation confirmations.</p>
    <h2>Data used</h2>
    <p>The service processes the request fields sent by ChatGPT, such as dress attributes, product identifiers, preferred size, store name, customer name, and related workflow details. For demo use, these requests are handled against static sample retail data in this application.</p>
    <h2>Data sharing</h2>
    <p>This demo service does not sell user data. Data is used only to fulfill the retail workflow requests sent to the API.</p>
    <h2>Retention</h2>
    <p>Logs and hosting telemetry may be retained by the hosting provider for operational purposes. Update this policy before production use if you add databases, analytics, or third-party services.</p>
    <h2>Contact</h2>
    <p>Contact the operator of this GPT for questions about this demo deployment.</p>
  </body>
</html>
"""


GPT_ACTION_INSTRUCTIONS = """You are RetailNext Luxury Assistant, a luxury retail associate copilot for women's gala and eveningwear.

When a customer uploads a dress image, inspect the image yourself and then call `matchFromStyleBrief`. Do not ask the API to inspect the file for you.

Flow:
1. Start with top 3 dress matches only.
2. Show the dress images and key details first.
3. After the matches, ask whether the associate wants nearby inventory or a different color.
4. If the customer asks for another color, call `checkAlternateColor`.
5. Offer add-ons only after the dress direction is clear.
6. Only call reservation or back-office verification after the associate confirms.

Tone:
- concise
- polished
- associate-facing
- never overwhelm the customer

If there is an uploaded dress image, summarize the observed silhouette, occasion, color, and style cues before calling `matchFromStyleBrief`.
"""


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
            "openapi_path": "/openapi.json",
            "privacy_path": "/privacy",
        }
    )


async def health(_: Any) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def openapi(_: Request) -> JSONResponse:
    return JSONResponse(_openapi_spec(PUBLIC_BASE_URL))


async def privacy(_: Request) -> HTMLResponse:
    return HTMLResponse(PRIVACY_POLICY_HTML)


async def gpt_instructions(_: Request) -> PlainTextResponse:
    return PlainTextResponse(GPT_ACTION_INSTRUCTIONS)


async def api_match_from_style_brief(request: Request) -> JSONResponse:
    payload = await _json_payload(request)
    required = ["category", "occasion", "color", "style_tags"]
    missing = [field for field in required if field not in payload]
    if missing:
        return _bad_request(f"Missing required fields: {', '.join(missing)}")

    result = match_products_from_style_brief(
        category=str(payload["category"]),
        occasion=str(payload["occasion"]),
        color=str(payload["color"]),
        style_tags=list(payload["style_tags"]),
        visual_summary=payload.get("visual_summary"),
    )
    return JSONResponse(_match_response(result["analysis"], result["matches"]))


async def api_match_from_demo_photo(request: Request) -> JSONResponse:
    payload = await _json_payload(request)
    file_name = str(payload.get("file_name", "gala_inspiration.jpg"))
    analysis = analyze_uploaded_photo(file_name)
    if analysis.get("status") != "success":
        return JSONResponse(analysis, status_code=404)

    matches = search_products(
        category=analysis["category"],
        occasion=analysis["occasion"],
        color=analysis["color"],
        style_tags=analysis["style_tags"],
    )
    return JSONResponse(_match_response(analysis, matches["results"]))


async def api_check_inventory(request: Request) -> JSONResponse:
    payload = await _json_payload(request)
    product_id = payload.get("product_id")
    if not product_id:
        return _bad_request("Missing required field: product_id")

    result = check_inventory(
        product_id=str(product_id),
        preferred_size=str(payload.get("preferred_size", PREFERRED_SIZE)),
        nearby_stores=_stores(payload.get("nearby_stores")),
    )
    return JSONResponse(_inventory_response(result))


async def api_check_alternate_color(request: Request) -> JSONResponse:
    payload = await _json_payload(request)
    anchor_product_id = payload.get("anchor_product_id")
    occasion = payload.get("occasion")
    if not anchor_product_id or not occasion:
        return _bad_request("Missing required fields: anchor_product_id, occasion")

    result = check_alternate_color_back_office(
        anchor_product_id=str(anchor_product_id),
        occasion=str(occasion),
        requested_color=payload.get("requested_color"),
        preferred_size=str(payload.get("preferred_size", PREFERRED_SIZE)),
        nearby_stores=payload.get("nearby_stores"),
    )
    return JSONResponse(_alternate_color_response(result))


async def api_build_outfit(request: Request) -> JSONResponse:
    payload = await _json_payload(request)
    anchor_product_id = payload.get("anchor_product_id")
    occasion = payload.get("occasion")
    if not anchor_product_id or not occasion:
        return _bad_request("Missing required fields: anchor_product_id, occasion")

    result = build_outfit_recommendations(
        anchor_product_id=str(anchor_product_id),
        occasion=str(occasion),
    )
    return JSONResponse(_outfit_response(result))


async def api_reserve_pickup(request: Request) -> JSONResponse:
    payload = await _json_payload(request)
    required = ["product_id", "store_name", "size", "customer_name"]
    missing = [field for field in required if not payload.get(field)]
    if missing:
        return _bad_request(f"Missing required fields: {', '.join(missing)}")

    result = reserve_store_pickup(
        product_id=str(payload["product_id"]),
        store_name=str(payload["store_name"]),
        size=str(payload["size"]),
        customer_name=str(payload["customer_name"]),
    )
    return JSONResponse(result)


async def api_verify_back_office(request: Request) -> JSONResponse:
    payload = await _json_payload(request)
    required = [
        "product_id",
        "store_name",
        "size",
        "customer_name",
        "invoice_id",
        "shipped_date",
    ]
    missing = [field for field in required if not payload.get(field)]
    if missing:
        return _bad_request(f"Missing required fields: {', '.join(missing)}")

    result = verify_back_office_availability(
        product_id=str(payload["product_id"]),
        store_name=str(payload["store_name"]),
        size=str(payload["size"]),
        customer_name=str(payload["customer_name"]),
        invoice_id=str(payload["invoice_id"]),
        shipped_date=str(payload["shipped_date"]),
    )
    return JSONResponse(result)


app = mcp_app
app.mount("/images", StaticFiles(directory=str(BASE_DIR / "images")), name="images")
app.add_route("/", root, methods=["GET"])
app.add_route("/health", health, methods=["GET"])
app.add_route("/openapi.json", openapi, methods=["GET"])
app.add_route("/privacy", privacy, methods=["GET"])
app.add_route("/gpt-instructions.txt", gpt_instructions, methods=["GET"])
app.add_route("/api/match-from-style-brief", api_match_from_style_brief, methods=["POST"])
app.add_route("/api/match-from-demo-photo", api_match_from_demo_photo, methods=["POST"])
app.add_route("/api/check-inventory", api_check_inventory, methods=["POST"])
app.add_route("/api/check-alternate-color", api_check_alternate_color, methods=["POST"])
app.add_route("/api/build-outfit", api_build_outfit, methods=["POST"])
app.add_route("/api/reserve-pickup", api_reserve_pickup, methods=["POST"])
app.add_route("/api/verify-back-office", api_verify_back_office, methods=["POST"])


if __name__ == "__main__":
    uvicorn.run(app, host=MCP_HOST, port=MCP_PORT)
