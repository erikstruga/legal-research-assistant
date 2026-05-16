"""
CourtListener API client.
Docs: https://www.courtlistener.com/help/api/rest/
Free tier allows unauthenticated access (rate-limited); set COURTLISTENER_API_TOKEN for higher limits.
"""

import httpx
import logging
from typing import Optional
from dataclasses import dataclass

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class CaseLawDocument:
    case_name: str
    court: str
    date_filed: Optional[str]
    citation: Optional[str]
    plain_text: str
    absolute_url: str
    doc_type: str = "case_law"

    @property
    def url(self) -> str:
        return f"https://www.courtlistener.com{self.absolute_url}"

    @property
    def title(self) -> str:
        parts = [self.case_name]
        if self.citation:
            parts.append(f"({self.citation})")
        if self.date_filed:
            parts.append(self.date_filed[:4])
        return " ".join(parts)


class CourtListenerClient:
    """Async client for the CourtListener REST API v3."""

    def __init__(self):
        settings = get_settings()
        headers = {"Accept": "application/json"}
        if settings.courtlistener_api_token:
            headers["Authorization"] = f"Token {settings.courtlistener_api_token}"

        self._client = httpx.AsyncClient(
            base_url=settings.courtlistener_base_url,
            headers=headers,
            timeout=30.0,
        )

    async def search_opinions(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[CaseLawDocument]:
        """
        Search CourtListener for opinions matching the query.
        Returns a list of CaseLawDocument objects with full text.
        """
        results: list[CaseLawDocument] = []

        try:
            # Step 1: Search for matching clusters
            resp = await self._client.get(
                "/search/",
                params={
                    "q": query,
                    "type": "o",          # opinions
                    "order_by": "score desc",
                    "format": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("results", [])[:max_results]

            for hit in hits:
                # Step 2: Fetch opinion text via the opinion endpoint
                opinion_id = hit.get("id")
                plain_text = hit.get("snippet", "")

                if opinion_id and not plain_text:
                    try:
                        op_resp = await self._client.get(f"/opinions/{opinion_id}/")
                        op_resp.raise_for_status()
                        op_data = op_resp.json()
                        plain_text = (
                            op_data.get("plain_text")
                            or op_data.get("html_with_citations")
                            or ""
                        )
                    except Exception as e:
                        logger.warning(f"Failed to fetch opinion {opinion_id}: {e}")

                if not plain_text:
                    continue

                results.append(
                    CaseLawDocument(
                        case_name=hit.get("caseName", "Unknown Case"),
                        court=hit.get("court", ""),
                        date_filed=hit.get("dateFiled"),
                        citation=hit.get("citation", [None])[0] if hit.get("citation") else None,
                        plain_text=plain_text[:8000],  # cap per-doc to avoid token overload
                        absolute_url=hit.get("absolute_url", ""),
                    )
                )

        except httpx.HTTPStatusError as e:
            logger.error(f"CourtListener HTTP error: {e.response.status_code} — {e.response.text}")
        except Exception as e:
            logger.error(f"CourtListener search failed: {e}")

        logger.info(f"CourtListener: fetched {len(results)} documents for '{query}'")
        return results

    async def close(self):
        await self._client.aclose()
