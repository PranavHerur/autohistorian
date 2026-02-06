"""Event extraction from articles."""

from datetime import datetime
from uuid import uuid4

from ..ingest.schemas import Article
from ..knowledge.models import Event
from ..llm.client import GeminiClient


class EventExtractor:
    """Extract events from articles using LLM."""

    def __init__(self, llm_client: GeminiClient):
        """Initialize the event extractor.

        Args:
            llm_client: Gemini client for LLM calls
        """
        self.llm_client = llm_client

    def _build_article_text(self, article: Article) -> str:
        """Build article text for extraction."""
        parts = [f"Headline: {article.headline.main}"]
        if article.abstract:
            parts.append(f"Abstract: {article.abstract}")
        if article.lead_paragraph:
            parts.append(f"Lead: {article.lead_paragraph}")
        return "\n\n".join(parts)

    def _parse_datetime(self, date_str: str) -> datetime | None:
        """Parse a datetime string."""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    async def extract(self, article: Article) -> list[Event]:
        """Extract events from an article.

        Args:
            article: The article to analyze

        Returns:
            List of Event objects
        """
        article_text = self._build_article_text(article)

        # Extract events using LLM
        raw_events = await self.llm_client.extract_events(article_text)

        # Convert to Event models
        events = []
        for raw in raw_events:
            event = Event(
                id=uuid4(),
                description=raw.get("description", ""),
                event_type=raw.get("event_type", "unknown"),
                valid_time=self._parse_datetime(raw.get("valid_time")),
                observation_time=article.pub_date,  # When the article was published
                participants=raw.get("participants", []),
                location=raw.get("location"),
                source_article_id=article.id,
                source_url=article.web_url,
            )
            events.append(event)

        return events
