"""Knowledge graph models for events, statements, and entities."""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A named entity (person, organization, location, etc.)."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    entity_type: str  # person, organization, location, etc.
    aliases: list[str] = Field(default_factory=list)
    description: Optional[str] = None


class Event(BaseModel):
    """An event extracted from news articles."""

    id: UUID = Field(default_factory=uuid4)
    description: str
    event_type: str  # arrest, statement, policy_change, etc.

    # Dual timeline
    valid_time: Optional[datetime] = None  # When it actually happened
    observation_time: Optional[datetime] = None  # When it was reported

    # Participants
    participants: list[str] = Field(default_factory=list)
    location: Optional[str] = None

    # Source tracking
    source_article_id: Optional[UUID] = None
    source_url: Optional[str] = None
    confidence: float = 1.0


class Statement(BaseModel):
    """A statement or quote attributed to a speaker."""

    id: UUID = Field(default_factory=uuid4)
    content: str
    speaker: str
    speaker_role: Optional[str] = None

    # Stance detection
    stance: Optional[str] = None  # pro, con, neutral
    target: Optional[str] = None  # What the statement is about

    # Dual timeline
    valid_time: Optional[datetime] = None  # When statement was made
    observation_time: Optional[datetime] = None  # When it was reported

    # Source tracking
    source_article_id: Optional[UUID] = None
    source_url: Optional[str] = None


class Topic(BaseModel):
    """A topic or theme that groups related events and statements."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    category: str = "other"  # politics, law, international, economy, etc.
    description: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)


class ExtractedTopic(BaseModel):
    """A topic extracted from an article."""

    name: str
    category: str = "other"
    relevance: float = 1.0


class ExtractionResult(BaseModel):
    """Result of extracting information from an article."""

    article_id: UUID
    events: list[Event] = Field(default_factory=list)
    statements: list[Statement] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    topics: list[ExtractedTopic] = Field(default_factory=list)
