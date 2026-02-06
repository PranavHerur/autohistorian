"""Topic extraction from articles."""

from ..ingest.schemas import Article
from ..llm.client import GeminiClient
from ..llm.prompts import TOPIC_EXTRACTION_PROMPT


class TopicExtractor:
    """Extract topics from articles using LLM."""

    def __init__(self, llm_client: GeminiClient):
        """Initialize the topic extractor.

        Args:
            llm_client: Gemini client for LLM calls
        """
        self.llm_client = llm_client

    async def extract_topics(self, article: Article) -> list[dict]:
        """Extract topics from an article.

        Args:
            article: The article to analyze

        Returns:
            List of topic dictionaries with name, category, and relevance
        """
        prompt = TOPIC_EXTRACTION_PROMPT.format(
            headline=article.headline.main,
            abstract=article.abstract or article.snippet or "",
        )

        response = await self.llm_client._generate(prompt)
        result = self.llm_client._extract_json(response)

        if isinstance(result, list):
            return result
        return []
