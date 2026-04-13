from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any


class MarketplaceConnector(ABC):
    @abstractmethod
    def fetch(self, endpoint_key: str, date_from: date) -> list[dict[str, Any]]:
        """Fetch data from marketplace API for endpoint and period."""
        raise NotImplementedError
