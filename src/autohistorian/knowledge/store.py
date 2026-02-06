"""Knowledge store for persisting extracted data."""

import json
from pathlib import Path
from typing import Optional
from uuid import UUID

from ..ingest.schemas import Article
from .models import Entity, Event, ExtractionResult, Statement, Topic


class KnowledgeStore:
    """Simple file-based knowledge store."""

    def __init__(self, data_dir: str = "data"):
        """Initialize the knowledge store.

        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = Path(data_dir)
        self.articles_dir = self.data_dir / "articles"
        self.extractions_dir = self.data_dir / "extractions"
        self.topics_dir = self.data_dir / "topics"

        # Create directories
        self.articles_dir.mkdir(parents=True, exist_ok=True)
        self.extractions_dir.mkdir(parents=True, exist_ok=True)
        self.topics_dir.mkdir(parents=True, exist_ok=True)

    def save_article(self, article: Article) -> None:
        """Save an article to the store."""
        path = self.articles_dir / f"{article.id}.json"
        path.write_text(article.model_dump_json(indent=2))

    def get_article(self, article_id: UUID) -> Optional[Article]:
        """Retrieve an article by ID."""
        path = self.articles_dir / f"{article_id}.json"
        if not path.exists():
            return None
        return Article.model_validate_json(path.read_text())

    def save_extraction_result(
        self, result: ExtractionResult, topic: Optional[str] = None
    ) -> None:
        """Save extraction results."""
        # Save to extractions directory
        path = self.extractions_dir / f"{result.article_id}.json"
        path.write_text(result.model_dump_json(indent=2))

        # Update topic indices
        for extracted_topic in result.topics:
            self._add_to_topic(extracted_topic.name, extracted_topic.category, result)

    def _add_to_topic(
        self, topic_name: str, category: str, result: ExtractionResult
    ) -> None:
        """Add extraction results to a topic index."""
        safe_name = topic_name.replace("/", "_").replace(" ", "_")[:100]
        topic_file = self.topics_dir / f"{safe_name}.json"

        if topic_file.exists():
            topic_data = json.loads(topic_file.read_text())
        else:
            topic_data = {
                "name": topic_name,
                "category": category,
                "article_ids": [],
                "events": [],
                "statements": [],
            }

        # Add article reference
        article_id_str = str(result.article_id)
        if article_id_str not in topic_data["article_ids"]:
            topic_data["article_ids"].append(article_id_str)

        # Add events
        for event in result.events:
            event_data = event.model_dump(mode="json")
            topic_data["events"].append(event_data)

        # Add statements
        for statement in result.statements:
            statement_data = statement.model_dump(mode="json")
            topic_data["statements"].append(statement_data)

        topic_file.write_text(json.dumps(topic_data, indent=2, default=str))

    def get_topics(self) -> list[str]:
        """Get all topic names."""
        topics = []
        for path in self.topics_dir.glob("*.json"):
            data = json.loads(path.read_text())
            topics.append(data["name"])
        return sorted(topics)

    def get_all_topics_info(self) -> list[dict]:
        """Get info about all topics, sorted by coverage."""
        topics_info = []
        for path in self.topics_dir.glob("*.json"):
            data = json.loads(path.read_text())
            topics_info.append({
                "name": data["name"],
                "category": data.get("category", "other"),
                "article_count": len(data.get("article_ids", [])),
                "event_count": len(data.get("events", [])),
                "statement_count": len(data.get("statements", [])),
            })

        # Sort by total coverage (articles + events + statements)
        topics_info.sort(
            key=lambda t: t["article_count"] + t["event_count"] + t["statement_count"],
            reverse=True,
        )
        return topics_info

    def get_topic_data(self, topic_name: str) -> Optional[dict]:
        """Get all data for a topic."""
        safe_name = topic_name.replace("/", "_").replace(" ", "_")[:100]
        topic_file = self.topics_dir / f"{safe_name}.json"
        if not topic_file.exists():
            return None
        return json.loads(topic_file.read_text())

    def get_events_for_topic(self, topic_name: str) -> list[Event]:
        """Get all events for a topic."""
        data = self.get_topic_data(topic_name)
        if not data:
            return []
        return [Event.model_validate(e) for e in data.get("events", [])]

    def get_statements_for_topic(self, topic_name: str) -> list[Statement]:
        """Get all statements for a topic."""
        data = self.get_topic_data(topic_name)
        if not data:
            return []
        return [Statement.model_validate(s) for s in data.get("statements", [])]

    def get_timeline(
        self, topic_name: str, use_valid_time: bool = True
    ) -> list[dict]:
        """Get timeline items for a topic.

        Args:
            topic_name: Name of the topic
            use_valid_time: If True, sort by when events happened. If False, sort by when reported.

        Returns:
            List of timeline items with time, type, and description
        """
        data = self.get_topic_data(topic_name)
        if not data:
            return []

        items = []

        for event in data.get("events", []):
            time_key = "valid_time" if use_valid_time else "observation_time"
            items.append({
                "time": event.get(time_key) or event.get("observation_time"),
                "type": "event",
                "description": event.get("description", ""),
                "location": event.get("location"),
            })

        for statement in data.get("statements", []):
            time_key = "valid_time" if use_valid_time else "observation_time"
            items.append({
                "time": statement.get(time_key) or statement.get("observation_time"),
                "type": "statement",
                "content": statement.get("content", ""),
                "speaker": statement.get("speaker", "Unknown"),
                "stance": statement.get("stance"),
            })

        # Sort by time
        items.sort(key=lambda x: x.get("time") or "")
        return items

    def get_stats(self) -> dict:
        """Get statistics about the knowledge store."""
        article_count = len(list(self.articles_dir.glob("*.json")))
        extraction_count = len(list(self.extractions_dir.glob("*.json")))
        topic_count = len(list(self.topics_dir.glob("*.json")))

        # Count events and statements across all topics
        total_events = 0
        total_statements = 0
        for path in self.topics_dir.glob("*.json"):
            data = json.loads(path.read_text())
            total_events += len(data.get("events", []))
            total_statements += len(data.get("statements", []))

        return {
            "articles": article_count,
            "extractions": extraction_count,
            "topics": topic_count,
            "events": total_events,
            "statements": total_statements,
        }
