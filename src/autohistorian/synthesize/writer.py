"""Article synthesis and writing."""

import json
from typing import Optional

from ..knowledge.store import KnowledgeStore
from ..llm.client import GeminiClient


class ArticleWriter:
    """Generate Wikipedia-style articles from extracted data."""

    def __init__(self, llm_client: GeminiClient, store: KnowledgeStore):
        """Initialize the article writer.

        Args:
            llm_client: Gemini client for LLM calls
            store: Knowledge store to read data from
        """
        self.llm_client = llm_client
        self.store = store

    async def generate_article(self, topic: str) -> str:
        """Generate a Wikipedia-style article for a topic.

        Args:
            topic: The topic to write about

        Returns:
            Article text in markdown format
        """
        # Get events and statements for the topic
        events = self.store.get_events_for_topic(topic)
        statements = self.store.get_statements_for_topic(topic)

        # Convert to dicts for the LLM
        events_data = [e.model_dump(mode="json") for e in events]
        statements_data = [s.model_dump(mode="json") for s in statements]

        # Generate the article
        article = await self.llm_client.synthesize_article(
            topic=topic,
            events=events_data,
            statements=statements_data,
        )

        # Add timeline section
        timeline = self._generate_timeline_section(topic)
        if timeline:
            article += "\n\n" + timeline

        return article

    async def generate_with_perspectives(self, topic: str) -> str:
        """Generate an article with multiple perspectives highlighted.

        Args:
            topic: The topic to write about

        Returns:
            Article text with perspectives section
        """
        article = await self.generate_article(topic)

        # Add perspectives section based on statement stances
        statements = self.store.get_statements_for_topic(topic)

        if statements:
            perspectives = "\n\n## Perspectives\n\n"

            # Group by stance
            pro_statements = [s for s in statements if s.stance == "pro"]
            con_statements = [s for s in statements if s.stance == "con"]
            neutral_statements = [s for s in statements if s.stance == "neutral"]

            if pro_statements:
                perspectives += "### Supporting Views\n"
                for s in pro_statements[:3]:
                    perspectives += f'- **{s.speaker}**: "{s.content}"\n'

            if con_statements:
                perspectives += "\n### Opposing Views\n"
                for s in con_statements[:3]:
                    perspectives += f'- **{s.speaker}**: "{s.content}"\n'

            if neutral_statements:
                perspectives += "\n### Neutral Analysis\n"
                for s in neutral_statements[:3]:
                    perspectives += f'- **{s.speaker}**: "{s.content}"\n'

            article += perspectives

        return article

    def _generate_timeline_section(self, topic: str) -> str:
        """Generate a dual-timeline section.

        Args:
            topic: The topic to generate timeline for

        Returns:
            Timeline section in markdown
        """
        valid_items = self.store.get_timeline(topic, use_valid_time=True)
        obs_items = self.store.get_timeline(topic, use_valid_time=False)

        if not valid_items and not obs_items:
            return ""

        section = "## Timeline\n\n"

        # When events happened
        section += "### When Events Occurred\n"
        section += "*Chronological order of when events actually happened*\n\n"
        for item in valid_items[:10]:
            time_str = item.get("time", "Unknown date")
            if time_str and len(time_str) > 10:
                time_str = time_str[:10]
            desc = item.get("description") or item.get("content", "")
            section += f"- **{time_str}**: {desc}\n"

        # When we learned
        section += "\n### When We Learned\n"
        section += "*Order in which information was reported*\n\n"
        for item in obs_items[:10]:
            time_str = item.get("time", "Unknown date")
            if time_str and len(time_str) > 10:
                time_str = time_str[:10]
            desc = item.get("description") or item.get("content", "")
            section += f"- **{time_str}** (reported): {desc}\n"

        return section

    def export_timeline_json(self, topic: str) -> dict:
        """Export timeline in TimelineJS format.

        Args:
            topic: The topic to export

        Returns:
            TimelineJS-compatible JSON structure
        """
        items = self.store.get_timeline(topic, use_valid_time=True)

        events = []
        for item in items:
            time_str = item.get("time")
            if not time_str:
                continue

            # Parse date for TimelineJS format
            try:
                year = time_str[:4]
                month = time_str[5:7] if len(time_str) > 5 else "01"
                day = time_str[8:10] if len(time_str) > 8 else "01"
            except (IndexError, TypeError):
                continue

            event = {
                "start_date": {
                    "year": year,
                    "month": month,
                    "day": day,
                },
                "text": {
                    "headline": item.get("description", "")[:100],
                    "text": item.get("description") or item.get("content", ""),
                },
            }

            if item.get("type") == "statement":
                event["text"]["headline"] = f'{item.get("speaker", "Unknown")}: {event["text"]["headline"]}'

            events.append(event)

        return {
            "title": {
                "text": {
                    "headline": topic,
                    "text": f"Timeline of events related to {topic}",
                }
            },
            "events": events,
        }
