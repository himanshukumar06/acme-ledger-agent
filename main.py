import os
import json
import logging
from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

QUESTION_DATA_URL = os.getenv("QUESTION_DATA_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing from .env"
    )



# Asia/Kolkata is the business timezone specified by the API.
IST = ZoneInfo("Asia/Kolkata")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# OPENAI CLIENT
# ============================================================

client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Acme Appliances Ledger Agent",
    description="Answers natural-language questions about the Acme order ledger.",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str


# ============================================================
# GLOBAL DATA CACHE
# ============================================================

ORDERS = []
RATES = {}

REGIONS = set()
PRODUCTS = set()
CUSTOMERS = set()

DATA_LOADED = False


# ============================================================
# HELPER: PARSE ISO TIMESTAMP
# ============================================================

def parse_timestamp(timestamp: str) -> datetime:
    """
    Convert ISO timestamp into a timezone-aware datetime.

    Handles both:
        2026-03-01T10:00:00Z
    and:
        2026-03-01T10:00:00+05:30
    """

    if timestamp.endswith("Z"):
        timestamp = timestamp[:-1] + "+00:00"

    return datetime.fromisoformat(timestamp)


# ============================================================
# HELPER: BUSINESS DATE
# ============================================================

def get_business_date(timestamp: str) -> date:
    """
    Convert an order timestamp to the Asia/Kolkata business date.
    """

    dt = parse_timestamp(timestamp)

    return dt.astimezone(IST).date()


# ============================================================
# FETCH ROOT API
# ============================================================

def fetch_root_data():
    """
    Fetch the root questionData endpoint.

    The root response gives us:
        links.orders
        links.rates
        links.export
        notes
    """

    logger.info("Fetching root API...")

    response = requests.get(
        QUESTION_DATA_URL,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    return data


# ============================================================
# FETCH ALL ORDERS
# ============================================================

def fetch_all_orders(orders_url: str):
    """
    Follow the API's pagination using the 'next' URL.

    The API currently reports 85 pages.
    """

    all_orders = []

    url = orders_url
    page_count = 0

    while url:

        page_count += 1

        logger.info("Fetching orders page %s...", page_count)

        response = requests.get(
            url,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        page_orders = data.get("orders", [])

        all_orders.extend(page_orders)

        # IMPORTANT:
        # We follow the API-provided next URL rather than
        # guessing how pagination works.
        url = data.get("next")

    logger.info(
        "Finished downloading orders. Raw rows: %s",
        len(all_orders)
    )

    return all_orders


# ============================================================
# FETCH RATES
# ============================================================

def fetch_rates(rates_url: str):
    """
    Fetch currency conversion rates.

    API structure:

    {
        "usd_per_unit": {
            "USD": 1,
            "EUR": 1.03,
            "INR": 0.01217
        }
    }
    """

    logger.info("Fetching currency rates...")

    response = requests.get(
        rates_url,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    rates = data.get("usd_per_unit", {})

    if not rates:
        raise RuntimeError("No currency rates found")

    logger.info("Currency rates: %s", rates)

    return rates


# ============================================================
# DEDUPLICATE ORDERS
# ============================================================

def deduplicate_orders(orders):
    """
    An order ID may occur multiple times.

    The API rule says:
        the row with the latest updated_at is current.

    Therefore, keep only the latest version of each ID.
    """

    latest_by_id = {}

    for order in orders:

        order_id = order.get("id")

        if not order_id:
            continue

        if order_id not in latest_by_id:
            latest_by_id[order_id] = order
            continue

        old_timestamp = parse_timestamp(
            latest_by_id[order_id]["updated_at"]
        )

        new_timestamp = parse_timestamp(
            order["updated_at"]
        )

        if new_timestamp > old_timestamp:
            latest_by_id[order_id] = order

    result = list(latest_by_id.values())

    logger.info(
        "After deduplication: %s current orders",
        len(result)
    )

    return result


# ============================================================
# MONEY CONVERSION
# ============================================================

def money_to_usd(order):
    """
    Convert order amount into USD.

    rates represent:
        USD value of one unit of each currency.
    """

    currency = order.get("currency")

    amount = Decimal(str(order.get("amount", 0)))

    if currency not in RATES:
        raise ValueError(
            f"Unknown currency: {currency}"
        )

    rate = Decimal(str(RATES[currency]))

    return amount * rate


# ============================================================
# ROUND MONEY
# ============================================================

def round_money(value: Decimal) -> float:
    """
    Round money to exactly two decimal places.
    """

    rounded = value.quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP
    )

    return float(rounded)


# ============================================================
# FILTER ORDERS
# ============================================================

def filter_orders(
    orders,
    region=None,
    product=None,
    customer=None,
    start_date=None,
    end_date=None,
):
    """
    Apply filters to current orders.

    Date filtering uses created_at converted to
    Asia/Kolkata business date.
    """

    result = []

    for order in orders:

        # ----------------------------------------------------
        # REGION
        # ----------------------------------------------------

        if region is not None:

            if order.get("region", "").lower() != region.lower():
                continue

        # ----------------------------------------------------
        # PRODUCT
        # ----------------------------------------------------

        if product is not None:

            if order.get("product", "").lower() != product.lower():
                continue

        # ----------------------------------------------------
        # CUSTOMER
        # ----------------------------------------------------

        if customer is not None:

            if order.get("customer", "").lower() != customer.lower():
                continue

        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        if start_date is not None or end_date is not None:

            order_date = get_business_date(
                order["created_at"]
            )

            if start_date is not None:

                if order_date < start_date:
                    continue

            if end_date is not None:

                # end_date is exclusive
                if order_date >= end_date:
                    continue

        result.append(order)

    return result


# ============================================================
# CALCULATE REVENUE
# ============================================================

def calculate_revenue(orders):

    total = Decimal("0")

    for order in orders:

        # API rule:
        # Only paid orders count as revenue.
        if order.get("status") != "paid":
            continue

        total += money_to_usd(order)

    return round_money(total)


# ============================================================
# CALCULATE REFUNDS
# ============================================================

def calculate_refunds(orders):

    total = Decimal("0")

    for order in orders:

        if order.get("status") != "refunded":
            continue

        total += money_to_usd(order)

    return round_money(total)


# ============================================================
# CALCULATE QUANTITY
# ============================================================

def calculate_quantity(orders):

    total = 0

    for order in orders:

        total += int(order.get("qty", 0))

    return total


# ============================================================
# CALCULATE ORDER COUNT
# ============================================================

def calculate_order_count(orders):

    return len(orders)


# ============================================================
# CALCULATE PAID ORDER COUNT
# ============================================================

def calculate_paid_order_count(orders):

    return sum(
        1
        for order in orders
        if order.get("status") == "paid"
    )


# ============================================================
# CALCULATE REFUNDED ORDER COUNT
# ============================================================

def calculate_refunded_order_count(orders):

    return sum(
        1
        for order in orders
        if order.get("status") == "refunded"
    )


# ============================================================
# LLM QUERY EXTRACTION
# ============================================================

def build_llm_instructions():

    region_list = sorted(REGIONS)
    product_list = sorted(PRODUCTS)

    return f"""
You are a query parser for the Acme Appliances order ledger.

Your ONLY job is to convert a natural-language question into
a structured JSON query.

DO NOT calculate numbers.

DO NOT answer the user's question.

Return ONLY valid JSON.

============================================================
ALLOWED METRICS
============================================================

The metric must be exactly one of:

- revenue
- refunds
- quantity
- orders
- paid_orders
- refunded_orders

============================================================
AVAILABLE REGIONS
============================================================

{json.dumps(region_list)}

============================================================
AVAILABLE PRODUCTS
============================================================

{json.dumps(product_list)}

============================================================
CUSTOMERS
============================================================

Customer IDs look like:

C1001
C1002
C1003
etc.

If a customer ID is explicitly mentioned, extract it.

============================================================
FIELDS
============================================================

Return exactly:

{{
    "metric": "...",
    "region": null,
    "product": null,
    "customer": null,
    "start_date": null,
    "end_date": null
}}

============================================================
METRIC RULES
============================================================

Revenue means:
    sum of amount converted to USD
    ONLY for orders whose current status is "paid".

Refunds means:
    sum of amount converted to USD
    for orders whose current status is "refunded".

Quantity means:
    sum of qty.

Orders means:
    count of matching current orders.

paid_orders means:
    count of matching current orders with status "paid".

refunded_orders means:
    count of matching current orders with status "refunded".

============================================================
DATE RULES
============================================================

Business dates use Asia/Kolkata.

For a month:

March 2026
means:

start_date = "2026-03-01"
end_date = "2026-04-01"

April 2026
means:

start_date = "2026-04-01"
end_date = "2026-05-01"

For a year:

2026
means:

start_date = "2026-01-01"
end_date = "2027-01-01"

For a specific date:

March 15, 2026
means:

start_date = "2026-03-15"
end_date = "2026-03-16"

If there is no date filter:

start_date = null
end_date = null

============================================================
FILTER RULES
============================================================

If the question mentions a region, extract it.

If the question mentions a product, extract it.

If the question mentions a customer, extract it.

If a filter is not mentioned, use null.

============================================================
SYNONYMS
============================================================

"sales", "earnings", "money generated",
"sales revenue", "revenue"
usually mean metric = revenue.

"refund", "refunded amount",
"money returned", "returns"
usually mean metric = refunds.

"how many units", "units sold",
"quantity sold"
usually mean metric = quantity.

"how many orders", "number of orders"
means metric = orders.

============================================================
IMPORTANT
============================================================

Do not invent values.

Use exact product names from the available product list.

Use exact region names from the available region list.

Customer IDs should be preserved exactly.

Return JSON only.
"""


def parse_question_with_llm(question: str):

    instructions = build_llm_instructions()

    completion = client.chat.completions.create(
        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": instructions
            },
            {
                "role": "user",
                "content": question
            }
        ],

        temperature=0,

        response_format={
            "type": "json_object"
        }
    )

    text = completion.choices[0].message.content.strip()

    logger.info(
        "Groq query: %s",
        text
    )

    try:

        query = json.loads(text)

    except json.JSONDecodeError as exc:

        logger.error(
            "Groq returned invalid JSON: %s",
            text
        )

        raise ValueError(
            "Groq returned invalid JSON"
        ) from exc

    return validate_query(query)
    # --------------------------------------------------------
    # Remove accidental markdown fences if model adds them.
    # --------------------------------------------------------

    if text.startswith("```"):

        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    try:
        query = json.loads(text)

    except json.JSONDecodeError as exc:

        logger.error(
            "LLM returned invalid JSON: %s",
            text
        )

        raise ValueError(
            "LLM returned invalid JSON"
        ) from exc

    return validate_query(query)


# ============================================================
# VALIDATE LLM QUERY
# ============================================================

def validate_query(query):

    allowed_metrics = {
        "revenue",
        "refunds",
        "quantity",
        "orders",
        "paid_orders",
        "refunded_orders",
    }

    metric = query.get("metric")

    if metric not in allowed_metrics:
        raise ValueError(
            f"Invalid metric: {metric}"
        )

    region = query.get("region")
    product = query.get("product")
    customer = query.get("customer")

    # --------------------------------------------------------
    # Validate region
    # --------------------------------------------------------

    if region is not None:

        matching_region = next(
            (
                r for r in REGIONS
                if r.lower() == str(region).lower()
            ),
            None
        )

        if matching_region is None:
            raise ValueError(
                f"Unknown region: {region}"
            )

        region = matching_region

    # --------------------------------------------------------
    # Validate product
    # --------------------------------------------------------

    if product is not None:

        matching_product = next(
            (
                p for p in PRODUCTS
                if p.lower() == str(product).lower()
            ),
            None
        )

        if matching_product is None:
            raise ValueError(
                f"Unknown product: {product}"
            )

        product = matching_product

    # --------------------------------------------------------
    # Validate customer
    # --------------------------------------------------------

    if customer is not None:

        customer = str(customer).upper()

        if customer not in CUSTOMERS:
            raise ValueError(
                f"Unknown customer: {customer}"
            )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    start_date = query.get("start_date")
    end_date = query.get("end_date")

    if start_date is not None:

        date.fromisoformat(start_date)

    if end_date is not None:

        date.fromisoformat(end_date)

    return {
        "metric": metric,
        "region": region,
        "product": product,
        "customer": customer,
        "start_date": start_date,
        "end_date": end_date,
    }


# ============================================================
# EXECUTE STRUCTURED QUERY
# ============================================================

def execute_query(query):

    start_date = None
    end_date = None

    if query.get("start_date"):
        start_date = date.fromisoformat(
            query["start_date"]
        )

    if query.get("end_date"):
        end_date = date.fromisoformat(
            query["end_date"]
        )

    matching_orders = filter_orders(
        orders=ORDERS,
        region=query.get("region"),
        product=query.get("product"),
        customer=query.get("customer"),
        start_date=start_date,
        end_date=end_date,
    )

    metric = query["metric"]

    if metric == "revenue":

        return calculate_revenue(
            matching_orders
        )

    if metric == "refunds":

        return calculate_refunds(
            matching_orders
        )

    if metric == "quantity":

        return calculate_quantity(
            matching_orders
        )

    if metric == "orders":

        return calculate_order_count(
            matching_orders
        )

    if metric == "paid_orders":

        return calculate_paid_order_count(
            matching_orders
        )

    if metric == "refunded_orders":

        return calculate_refunded_order_count(
            matching_orders
        )

    raise ValueError(
        f"Unsupported metric: {metric}"
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    global ORDERS
    global RATES
    global REGIONS
    global PRODUCTS
    global CUSTOMERS
    global DATA_LOADED

    if DATA_LOADED:
        return

    logger.info("=" * 60)
    logger.info("LOADING ACME LEDGER DATA")
    logger.info("=" * 60)

    # --------------------------------------------------------
    # ROOT
    # --------------------------------------------------------

    root = fetch_root_data()

    links = root.get("links", {})

    orders_url = links.get("orders")
    rates_url = links.get("rates")

    if not orders_url:
        raise RuntimeError(
            "Root API did not provide an orders URL"
        )

    if not rates_url:
        raise RuntimeError(
            "Root API did not provide a rates URL"
        )

    # --------------------------------------------------------
    # ORDERS
    # --------------------------------------------------------

    raw_orders = fetch_all_orders(
        orders_url
    )

    # --------------------------------------------------------
    # DEDUPLICATION
    # --------------------------------------------------------

    ORDERS = deduplicate_orders(
        raw_orders
    )

    # --------------------------------------------------------
    # RATES
    # --------------------------------------------------------

    RATES = fetch_rates(
        rates_url
    )

    # --------------------------------------------------------
    # BUILD DIMENSION SETS
    # --------------------------------------------------------

    REGIONS = {
        str(order.get("region"))
        for order in ORDERS
        if order.get("region")
    }

    PRODUCTS = {
        str(order.get("product"))
        for order in ORDERS
        if order.get("product")
    }

    CUSTOMERS = {
        str(order.get("customer")).upper()
        for order in ORDERS
        if order.get("customer")
    }

    DATA_LOADED = True

    logger.info("=" * 60)
    logger.info("DATA READY")
    logger.info("Raw orders: %s", len(raw_orders))
    logger.info("Current orders: %s", len(ORDERS))
    logger.info("Regions: %s", sorted(REGIONS))
    logger.info("Products: %s", sorted(PRODUCTS))
    logger.info("Customers: %s", len(CUSTOMERS))
    logger.info("Rates: %s", RATES)
    logger.info("=" * 60)


# ============================================================
# STARTUP
# ============================================================



# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "orders_loaded": len(ORDERS),
        "data_loaded": DATA_LOADED,
    }


# ============================================================
# ROOT GET
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Acme Appliances Ledger Agent",
        "status": "ok",
        "orders_loaded": len(ORDERS),
    }


# ============================================================
# MAIN QUESTION ENDPOINT
# ============================================================

@app.post("/")
def answer_question(request: QuestionRequest):
    load_data()
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="question must not be empty"
        )

    try:

        # ----------------------------------------------------
        # 1. English question → structured query
        # ----------------------------------------------------

        query = parse_question_with_llm(
            question
        )

        # ----------------------------------------------------
        # 2. Structured query → exact calculation
        # ----------------------------------------------------

        answer = execute_query(
            query
        )

        # ----------------------------------------------------
        # 3. Return required JSON
        # ----------------------------------------------------

        return {
            "answer": answer
        }

    except Exception as exc:

        logger.exception(
            "Question processing failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(
            os.getenv("PORT", "8000")
        ),
        reload=False
    )