from __future__ import annotations

from datetime import date, timedelta

from api import WB_ENDPOINTS
from config import get_settings
from utils import get_logger
from .raw_loader import RawLoader
from .stg_loader import StgLoader
from .mart_loader import MartLoader


class Pipeline:
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(self.__class__.__name__, self.settings.log_level)
        self.raw_loader = RawLoader()
        self.stg_loader = StgLoader(self.settings.log_level)
        self.mart_loader = MartLoader(self.settings.log_level)

    def run(self, days_back: int = 7) -> None:
        date_from = date.today() - timedelta(days=days_back)
        date_to = date.today()

        for endpoint_key in ["orders", "sales", "stocks"]:
            if endpoint_key in WB_ENDPOINTS:
                self.raw_loader.load_endpoint(endpoint_key=endpoint_key, date_from=date_from)

        if "finance" in WB_ENDPOINTS:
            self.raw_loader.load_finance(date_from=date_from, date_to=date_to)

        try:
            self.raw_loader.load_ads(date_from=date_from, date_to=date_to)
        except Exception as e:
            self.logger.warning("Ads loading skipped: %s", e)

        self.stg_loader.run()
        self.mart_loader.run()
        self.logger.info("Pipeline completed")
