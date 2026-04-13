"""Wildberries API endpoint registry.
Assumption: endpoints are based on public WB Statistics API routes and can be replaced in one place.
"""

WB_ENDPOINTS = {
    "orders": "/api/v1/supplier/orders",
    "sales": "/api/v1/supplier/sales",
    "stocks": "/api/v1/supplier/stocks",
    # Optional endpoint in some accounts/tariffs. Keep replaceable.
    "income": "/api/v1/supplier/incomes",
    # Financial reports — detailed per-sale breakdown (logistics, storage, penalties, etc.)
    "finance": "/api/v5/supplier/reportDetailByPeriod",
}

# Promotion (Advertising) API — separate host
WB_ADV_HOST = "https://advert-api.wildberries.ru"
WB_ADV_ENDPOINTS = {
    "adv_campaigns": "/adv/v1/promotion/count",      # GET — list campaigns grouped by type/status
    "adv_fullstats": "/adv/v3/fullstats",             # GET — full stats (?ids=...&beginDate=...&endDate=..., max 31 day)
}
