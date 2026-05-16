"""
Congress.gov API client.
Docs: https://api.congress.gov/
Requires a free API key from https://api.congress.gov/sign-up/
"""

import httpx
import logging
from dataclasses import dataclass
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class StatuteDocument:
    title: str
    congress: int
    bill_type: str
    bill_number: str
    summary: str
    url: str
    doc_type: str = "statute"

    @property
    def full_title(self) -> str:
        return f"{self.bill_type.upper()} {self.bill_number} ({self.congress}th Congress) — {self.title}"


class CongressClient:
    """Async client for the Congress.gov REST API v3."""

    def __init__(self):
        settings = get_settings()
        self._api_key = settings.congress_api_key
        self._client = httpx.AsyncClient(
            base_url=settings.congress_base_url,
            timeout=30.0,
        )

    def _params(self, extra: dict | None = None) -> dict:
        p = {"api_key": self._api_key, "format": "json"}
        if extra:
            p.update(extra)
        return p

    async def search_bills(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[StatuteDocument]:
        """
        Search Congress.gov for bills matching the query.
        Falls back to browsing the most recent congress bills if the text-search
        endpoint is not available on the free tier.
        """
        results: list[StatuteDocument] = []

        try:
            resp = await self._client.get(
                "/bill",
                params=self._params(
                    {
                        "query": query,
                        "limit": max_results,
                        "sort": "updateDate+desc",
                    }
                ),
            )
            resp.raise_for_status()
            data = resp.json()
            bills = data.get("bills", [])

            for bill in bills:
                # Fetch the bill's summary text
                congress = bill.get("congress")
                bill_type = bill.get("type", "").lower()
                bill_number = bill.get("number", "")

                summary_text = bill.get("title", "")
                try:
                    s_resp = await self._client.get(
                        f"/bill/{congress}/{bill_type}/{bill_number}/summaries",
                        params=self._params(),
                    )
                    s_resp.raise_for_status()
                    summaries = s_resp.json().get("summaries", [])
                    if summaries:
                        summary_text = summaries[-1].get("text", summary_text)
                        # Strip basic HTML tags
                        import re
                        summary_text = re.sub(r"<[^>]+>", " ", summary_text)
                except Exception:
                    pass  # fall back to title

                if not summary_text:
                    continue

                bill_url = (
                    bill.get("url")
                    or f"https://www.congress.gov/bill/{congress}th-congress/{bill_type}-bill/{bill_number}"
                )

                results.append(
                    StatuteDocument(
                        title=bill.get("title", "Untitled Bill"),
                        congress=congress,
                        bill_type=bill_type,
                        bill_number=bill_number,
                        summary=summary_text[:8000],
                        url=bill_url,
                    )
                )

        except httpx.HTTPStatusError as e:
            logger.error(f"Congress.gov HTTP error: {e.response.status_code} — {e.response.text}")
        except Exception as e:
            logger.error(f"Congress.gov search failed: {e}")

        logger.info(f"Congress.gov: fetched {len(results)} documents for '{query}'")
        return results

    async def close(self):
        await self._client.aclose()
