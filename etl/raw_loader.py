from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

from api import WildberriesClient
from config import get_settings
from db import RawRepository, get_engine
from utils import get_logger


class RawLoader:
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(self.__class__.__name__, self.settings.log_level)
        self.client = WildberriesClient()
        self.repo = RawRepository(get_engine())

    def load_endpoint(self, endpoint_key: str, date_from: date) -> list[dict[str, Any]]:
        payload = self.client.fetch(endpoint_key=endpoint_key, date_from=date_from)
        self.repo.insert_raw_payload(
            endpoint=endpoint_key,
            request_params={"dateFrom": date_from.isoformat()},
            payload=payload,
            source=self.settings.raw_source,
        )
        self.logger.info("RAW saved for endpoint=%s records=%s", endpoint_key, len(payload))
        return payload

    def load_finance(self, date_from: date, date_to: date) -> list[dict[str, Any]]:
        payload = self.client.fetch_finance(date_from=date_from, date_to=date_to)
        self.repo.insert_raw_payload(
            endpoint="finance",
            request_params={"dateFrom": date_from.isoformat(), "dateTo": date_to.isoformat()},
            payload=payload,
            source=self.settings.raw_source,
        )
        self.logger.info("RAW saved for finance report: %d records", len(payload))
        return payload

    def load_ads(self, date_from: date, date_to: date) -> list[dict[str, Any]]:
        """Fetch advertising stats from WB Promotion API and save to raw."""
        # 1. Get campaign list
        campaigns = self.client.fetch_ads_campaigns()
        if not campaigns:
            self.logger.info("No advertising campaigns found")
            return []

        campaign_ids = [c["advertId"] for c in campaigns if "advertId" in c]
        self.logger.info("Found %d ad campaigns", len(campaign_ids))

        # 2. Fetch stats in 31-day windows (WB limit for GET /adv/v3/fullstats)
        all_stats: list[dict[str, Any]] = []
        window_start = date_from
        first_batch = True
        while window_start <= date_to:
            window_end = min(window_start + timedelta(days=30), date_to)
            if not first_batch:
                time.sleep(10)
            first_batch = False
            try:
                stats = self.client.fetch_ads_fullstats(
                    campaign_ids,
                    begin_date=window_start.isoformat(),
                    end_date=window_end.isoformat(),
                )
                all_stats.extend(stats)
            except Exception as e:
                self.logger.warning("Ads stats batch %s..%s failed: %s",
                                    window_start, window_end, e)
            window_start = window_end + timedelta(days=1)

        self.repo.insert_raw_payload(
            endpoint="ads",
            request_params={"dateFrom": date_from.isoformat(), "dateTo": date_to.isoformat()},
            payload=all_stats,
            source=self.settings.raw_source,
        )
        self.logger.info("RAW saved for ads stats: %d campaign records", len(all_stats))
        return all_stats
