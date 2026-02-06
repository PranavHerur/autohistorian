"""Statement extraction from articles."""

from uuid import uuid4

from ..ingest.schemas import Article
from ..knowledge.models import Statement
from ..llm.client import GeminiClient


class StatementExtractor:
    """Extract statements and quotes from articles using LLM."""

    def __init__(self, llm_client: GeminiClient):
        """Initialize the statement extractor.

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

    async def extract(self, article: Article) -> list[Statement]:
        """Extract statements from an article.

        Args:
            article: The article to analyze

        Returns:
            List of Statement objects
        """
        article_text = self._build_article_text(article)

        # Extract statements using LLM
        raw_statements = await self.llm_client.extract_statements(article_text)

        # Convert to Statement models
        statements = []
        for raw in raw_statements:
            statement = Statement(
                id=uuid4(),
                content=raw.get("content", ""),
                speaker=raw.get("speaker", "Unknown"),
                speaker_role=raw.get("speaker_role"),
                stance=raw.get("stance"),
                target=raw.get("target"),
                observation_time=article.pub_date,  # When the article was published
                source_article_id=article.id,
                source_url=article.web_url,
            )
            statements.append(statement)

        return statements
