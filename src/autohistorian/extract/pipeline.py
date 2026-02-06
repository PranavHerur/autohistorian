"""Extraction pipeline for processing articles."""

import asyncio
from typing import Optional

from ..ingest.schemas import Article
from ..knowledge.models import ExtractionResult, ExtractedTopic
from ..llm.client import GeminiClient
from .entity_extractor import EntityExtractor
from .event_extractor import EventExtractor
from .statement_extractor import StatementExtractor
from .topic_extractor import TopicExtractor


class ExtractionPipeline:
    """Pipeline for extracting structured data from articles."""

    def __init__(self, llm_client: GeminiClient):
        """Initialize the extraction pipeline.

        Args:
            llm_client: Gemini client for LLM calls
        """
        self.llm_client = llm_client
        self.event_extractor = EventExtractor(llm_client)
        self.statement_extractor = StatementExtractor(llm_client)
        self.entity_extractor = EntityExtractor(llm_client)
        self.topic_extractor = TopicExtractor(llm_client)

    async def extract(
        self, article: Article, topic: Optional[str] = None
    ) -> ExtractionResult:
        """Extract all information from an article.

        Args:
            article: The article to process
            topic: Optional topic to focus on (if None, auto-detect topics)

        Returns:
            ExtractionResult with events, statements, entities, and topics
        """
        # Determine if we should auto-detect topics
        auto_topics = topic is None
        topics = []

        if auto_topics:
            # Run topic and entity extraction in parallel
            topics_task = self.topic_extractor.extract_topics(article)
            entities_task = self.entity_extractor.extract(article)
            raw_topics, entities = await asyncio.gather(topics_task, entities_task)

            topics = [
                ExtractedTopic(
                    name=t.get("name", "Unknown"),
                    category=t.get("category", "other"),
                    relevance=t.get("relevance", 1.0),
                )
                for t in raw_topics
            ]
        else:
            entities = await self.entity_extractor.extract(article)
            topics = [ExtractedTopic(name=topic, category="other", relevance=1.0)]

        # Extract events and statements in parallel
        events_task = self.event_extractor.extract(article)
        statements_task = self.statement_extractor.extract(article)
        events, statements = await asyncio.gather(events_task, statements_task)

        return ExtractionResult(
            article_id=article.id,
            events=events,
            statements=statements,
            entities=entities,
            topics=topics,
        )

    async def extract_batch(
        self,
        articles: list[Article],
        topic: Optional[str] = None,
        max_concurrent: int = 5,
    ) -> list[ExtractionResult]:
        """Extract from multiple articles with concurrency control.

        Args:
            articles: List of articles to process
            topic: Optional topic to focus on
            max_concurrent: Maximum concurrent extractions

        Returns:
            List of ExtractionResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_with_semaphore(article: Article) -> ExtractionResult:
            async with semaphore:
                return await self.extract(article, topic)

        tasks = [extract_with_semaphore(article) for article in articles]
        return await asyncio.gather(*tasks)
