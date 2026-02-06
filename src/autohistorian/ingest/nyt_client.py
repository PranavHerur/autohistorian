"""NYT Article Search API client."""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

import httpx

from .schemas import (
    ArchiveResponse,
    Article,
    ArticleSearchResponse,
    Byline,
    Headline,
    Keyword,
)


class NYTClient:
    """Client for the NYT Article Search API.

    API documentation: https://developer.nytimes.com/docs/articlesearch-product/1/overview
    """

    BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
    ARCHIVE_URL = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
    RATE_LIMIT_DELAY = 12.0  # NYT allows 5 requests per minute

    def __init__(self, api_key: str):
        """Initialize the NYT client.

        Args:
            api_key: NYT API key from https://developer.nytimes.com/
        """
        self.api_key = api_key
        self._last_request_time: Optional[float] = None

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self._last_request_time is not None:
            elapsed = asyncio.get_event_loop().time() - self._last_request_time
            if elapsed < self.RATE_LIMIT_DELAY:
                await asyncio.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    def _parse_article(self, doc: dict) -> Article:
        """Parse a document from the NYT API response into an Article."""
        headline_data = doc.get("headline", {})
        headline = Headline(
            main=headline_data.get("main", ""),
            print_headline=headline_data.get("print_headline"),
        )

        byline_data = doc.get("byline")
        byline = None
        if byline_data:
            byline = Byline(
                original=byline_data.get("original"),
                organization=byline_data.get("organization"),
            )

        keywords = []
        for kw in doc.get("keywords") or []:
            keywords.append(
                Keyword(
                    name=kw.get("name", ""),
                    value=kw.get("value", ""),
                    rank=kw.get("rank", 0),
                    major=kw.get("major", "N"),
                )
            )

        # Parse publication date
        pub_date_str = doc.get("pub_date", "")
        try:
            pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pub_date = datetime.utcnow()

        return Article(
            id=uuid4(),
            web_url=doc.get("web_url", ""),
            snippet=doc.get("snippet"),
            lead_paragraph=doc.get("lead_paragraph"),
            abstract=doc.get("abstract"),
            headline=headline,
            byline=byline,
            source=doc.get("source", "The New York Times"),
            pub_date=pub_date,
            document_type=doc.get("document_type", "article"),
            section_name=doc.get("section_name"),
            subsection_name=doc.get("subsection_name"),
            keywords=keywords,
            word_count=doc.get("word_count", 0),
        )

    async def search(
        self,
        query: Optional[str] = None,
        begin_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        page: int = 0,
        sort: str = "newest",
        filter_query: Optional[str] = None,
        sections: Optional[list[str]] = None,
    ) -> ArticleSearchResponse:
        """Search for articles matching the query.

        Args:
            query: Search query string (optional - can fetch by date/section alone)
            begin_date: Start date for search range
            end_date: End date for search range
            page: Page number (0-indexed, max 100 pages)
            sort: Sort order - "newest", "oldest", or "relevance"
            filter_query: Additional filter query (fq parameter)
            sections: List of section names to filter by (e.g., ["U.S.", "Politics"])

        Returns:
            ArticleSearchResponse with matching articles
        """
        await self._rate_limit()

        params = {
            "api-key": self.api_key,
            "page": page,
            "sort": sort,
        }

        if query:
            params["q"] = query

        if begin_date:
            params["begin_date"] = begin_date.strftime("%Y%m%d")
        if end_date:
            params["end_date"] = end_date.strftime("%Y%m%d")

        # Build filter query
        fq_parts = []
        if filter_query:
            fq_parts.append(filter_query)
        if sections:
            section_filter = " OR ".join(f'"{s}"' for s in sections)
            fq_parts.append(f"section_name:({section_filter})")
        if fq_parts:
            params["fq"] = " AND ".join(fq_parts)

        async with httpx.AsyncClient() as client:
            response = await client.get(self.BASE_URL, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()

        articles = []
        docs = data.get("response", {}).get("docs") or []
        for doc in docs:
            articles.append(self._parse_article(doc))

        total_hits = data.get("response", {}).get("meta", {}).get("hits", 0)

        return ArticleSearchResponse(
            status=data.get("status", "OK"),
            copyright=data.get("copyright", ""),
            articles=articles,
            total_hits=total_hits,
        )

    async def search_all(
        self,
        query: Optional[str] = None,
        begin_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_pages: int = 10,
        sort: str = "newest",
        filter_query: Optional[str] = None,
        sections: Optional[list[str]] = None,
    ) -> list[Article]:
        """Search and return all articles across multiple pages.

        Args:
            query: Search query string (optional)
            begin_date: Start date for search range
            end_date: End date for search range
            max_pages: Maximum number of pages to fetch
            sort: Sort order
            filter_query: Additional filter query
            sections: List of section names to filter by

        Returns:
            List of all articles across pages
        """
        all_articles = []
        page = 0

        while page < max_pages:
            result = await self.search(
                query=query,
                begin_date=begin_date,
                end_date=end_date,
                page=page,
                sort=sort,
                filter_query=filter_query,
                sections=sections,
            )

            all_articles.extend(result.articles)

            # Check if we've fetched all results
            if len(result.articles) < 10:  # NYT returns max 10 per page
                break
            if len(all_articles) >= result.total_hits:
                break

            page += 1

        return all_articles

    async def search_recent(
        self,
        query: Optional[str] = None,
        days: int = 30,
        max_articles: int = 100,
        sections: Optional[list[str]] = None,
    ) -> list[Article]:
        """Search for recent articles, optionally filtered by topic or section.

        Args:
            query: Search query string (optional - if None, fetches all recent articles)
            days: Number of days to look back
            max_articles: Maximum number of articles to return
            sections: List of section names to filter by (e.g., ["U.S.", "Politics"])

        Returns:
            List of recent articles
        """
        end_date = datetime.utcnow()
        begin_date = end_date - timedelta(days=days)

        max_pages = (max_articles + 9) // 10  # Ceiling division

        return await self.search_all(
            query=query,
            begin_date=begin_date,
            end_date=end_date,
            max_pages=max_pages,
            sections=sections,
        )

    async def fetch_recent(
        self,
        days: int = 7,
        max_articles: int = 100,
        sections: Optional[list[str]] = None,
    ) -> list[Article]:
        """Fetch recent articles without a specific query.

        This is the simplest way to ingest recent news and let the system
        automatically discover topics.

        Args:
            days: Number of days to look back (not used - fetches most recent)
            max_articles: Maximum number of articles to return
            sections: Optional list of sections to filter by (not used currently).

        Returns:
            List of recent articles
        """
        max_pages = (max_articles + 9) // 10

        # NYT API requires a query to return results
        # Use "news" as a broad query to get recent general news articles
        return await self.search_all(
            query="news",
            begin_date=None,
            end_date=None,
            max_pages=max_pages,
            sections=None,  # Don't filter by section - just get recent news
        )

    async def fetch_archive(self, year: int, month: int) -> ArchiveResponse:
        """Fetch all articles from a specific month using the Archive API.

        The Archive API returns all articles published in a given month
        in a single request. This is more efficient than paginating through
        the Article Search API for bulk historical data.

        Args:
            year: The year (1851 to present)
            month: The month (1-12)

        Returns:
            ArchiveResponse with all articles from that month
        """
        if month < 1 or month > 12:
            raise ValueError(f"Month must be 1-12, got {month}")
        if year < 1851:
            raise ValueError(f"Year must be 1851 or later, got {year}")

        await self._rate_limit()

        url = self.ARCHIVE_URL.format(year=year, month=month)
        params = {"api-key": self.api_key}

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=60.0)
            response.raise_for_status()
            data = response.json()

        articles = []
        docs = data.get("response", {}).get("docs") or []
        for doc in docs:
            articles.append(self._parse_article(doc))

        return ArchiveResponse(
            copyright=data.get("copyright", ""),
            year=year,
            month=month,
            articles=articles,
            total_hits=len(articles),
        )

    async def fetch_archive_range(
        self,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
    ) -> list[ArchiveResponse]:
        """Fetch archives for a range of months.

        Args:
            start_year: Starting year
            start_month: Starting month (1-12)
            end_year: Ending year
            end_month: Ending month (1-12)

        Returns:
            List of ArchiveResponse objects, one per month
        """
        results = []
        current_year = start_year
        current_month = start_month

        while (current_year, current_month) <= (end_year, end_month):
            result = await self.fetch_archive(current_year, current_month)
            results.append(result)

            # Move to next month
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

        return results
