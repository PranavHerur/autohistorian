"""Entity extraction from articles."""

from uuid import uuid4

from ..ingest.schemas import Article
from ..knowledge.models import Entity
from ..llm.client import GeminiClient


class EntityExtractor:
    """Extract entities from articles using LLM."""

    def __init__(self, llm_client: GeminiClient):
        """Initialize the entity extractor.

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

    async def extract(self, article: Article) -> list[Entity]:
        """Extract entities from an article.

        Args:
            article: The article to analyze

        Returns:
            List of Entity objects
        """
        article_text = self._build_article_text(article)

        # Extract entities using LLM
        raw_entities = await self.llm_client.extract_entities(article_text)

        # Convert to Entity models
        entities = []
        for raw in raw_entities:
            entity = Entity(
                id=uuid4(),
                name=raw.get("name", ""),
                entity_type=raw.get("entity_type", "unknown"),
                description=raw.get("description"),
            )
            entities.append(entity)

        return entities
