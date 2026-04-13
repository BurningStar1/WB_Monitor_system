"""Wildberries HTTP client with safe logging."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import requests

from config import get_settings
from utils import get_logger
from .base_connector import MarketplaceConnector
from .endpoints import WB_ENDPOINTS, WB_ADV_HOST, WB_ADV_ENDPOINTS


@dataclass
class WBRequest:
    endpoint_key: str
    date_from: date


class WildberriesClient(MarketplaceConnector):
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(self.__class__.__name__, self.settings.log_level)

        if not self.settings.wb_api_token:
            self.logger.warning("WB API token is empty. API calls will fail until token is configured.")

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": self.settings.wb_api_token,
                "Content-Type": "application/json",
            }
        )

    def fetch(self, endpoint_key: str, date_from: date) -> list[dict[str, Any]]:
        req = WBRequest(endpoint_key=endpoint_key, date_from=date_from)
        return self.fetch_by_request(req)

    def fetch_finance(self, date_from: date, date_to: date) -> list[dict[str, Any]]:
        """Fetch financial report with auto-pagination (up to 100k records per page)."""
        path = WB_ENDPOINTS["finance"]
        url = f"{self.settings.wb_api_host}{path}"
        all_records: list[dict[str, Any]] = []
        rrdid = 0

        while True:
            params: dict[str, Any] = {
                "dateFrom": f"{date_from.isoformat()}T00:00:00Z",
                "dateTo": f"{date_to.isoformat()}T23:59:59Z",
                "limit": 100000,
            }
            if rrdid:
                params["rrdid"] = rrdid

            self.logger.info(
                "Requesting finance report dateFrom=%s dateTo=%s rrdid=%s",
                date_from, date_to, rrdid,
            )
            response = self.session.get(url, params=params, timeout=120)

            if response.status_code >= 400:
                self.logger.error("Finance API failed: status=%s", response.status_code)
                response.raise_for_status()

            data = response.json()
            if not isinstance(data, list) or len(data) == 0:
                break

            all_records.extend(data)
            self.logger.info("Finance page fetched: %d records (total: %d)", len(data), len(all_records))

            if len(data) < 100000:
                break
            rrdid = data[-1].get("rrd_id", 0)

        return all_records

    # ── Promotion (Advertising) API ──────────────────────────────

    def fetch_ads_campaigns(self) -> list[dict[str, Any]]:
        """Get list of all advertising campaigns via /adv/v1/promotion/count.

        Returns flat list of dicts with 'advertId' key extracted from the
        nested response grouped by type/status.
        """
        url = f"{WB_ADV_HOST}{WB_ADV_ENDPOINTS['adv_campaigns']}"
        self.logger.info("Requesting ads campaigns list")
        response = self.session.get(url, timeout=60)
        if response.status_code == 204:
            return []
        if response.status_code >= 400:
            self.logger.error("Ads campaigns API failed: status=%s", response.status_code)
            response.raise_for_status()
        data = response.json()
        # Response: {adverts: [{type, status, count, advert_list: [{advertId, changeTime}]}]}
        result: list[dict[str, Any]] = []
        for group in data.get("adverts", []):
            for adv in group.get("advert_list", []):
                adv["type"] = group.get("type")
                adv["status"] = group.get("status")
                result.append(adv)
        return result

    def fetch_ads_fullstats(
        self, campaign_ids: list[int], begin_date: str, end_date: str,
    ) -> list[dict[str, Any]]:
        """Fetch full advertising stats via GET /adv/v3/fullstats.

        Params: ids (comma-separated), beginDate, endDate (YYYY-MM-DD).
        WB limit: max 31-day period per request.
        """
        url = f"{WB_ADV_HOST}{WB_ADV_ENDPOINTS['adv_fullstats']}"
        params = {
            "ids": ",".join(str(cid) for cid in campaign_ids),
            "beginDate": begin_date,
            "endDate": end_date,
        }
        self.logger.info(
            "Requesting ads fullstats: %d campaigns, %s..%s",
            len(campaign_ids), begin_date, end_date,
        )
        for attempt in range(3):
            response = self.session.get(url, params=params, timeout=120)
            if response.status_code == 204:
                return []
            if response.status_code == 429:
                wait = 15 * (attempt + 1)
                self.logger.warning("Rate limited, retrying in %ds...", wait)
                time.sleep(wait)
                continue
            if response.status_code >= 400:
                self.logger.error("Ads fullstats API failed: status=%s body=%s",
                                  response.status_code, response.text[:300])
                response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        response.raise_for_status()
        return []

    def fetch_by_request(self, req: WBRequest) -> list[dict[str, Any]]:
        if req.endpoint_key not in WB_ENDPOINTS:
            raise ValueError(f"Unsupported endpoint_key: {req.endpoint_key}")

        path = WB_ENDPOINTS[req.endpoint_key]
        url = f"{self.settings.wb_api_host}{path}"
        params = {"dateFrom": req.date_from.isoformat()}

        self.logger.info("Requesting endpoint=%s dateFrom=%s", req.endpoint_key, params["dateFrom"])
        response = self.session.get(url, params=params, timeout=60)

        if response.status_code >= 400:
            self.logger.error(
                "WB API request failed: endpoint=%s status=%s", req.endpoint_key, response.status_code
            )
            response.raise_for_status()

        data = response.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []
