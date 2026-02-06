"""Schemas for NYT API responses."""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Headline(BaseModel):
    """Article headline from NYT API."""

    main: str
    print_headline: Optional[str] = None


class Byline(BaseModel):
    """Article byline information."""

    original: Optional[str] = None
    organization: Optional[str] = None


class Multimedia(BaseModel):
    """Multimedia item from article."""

    url: str
    type: str
    subtype: Optional[str] = None
    caption: Optional[str] = None


class Keyword(BaseModel):
    """Keyword/tag from article."""

    name: str
    value: str
    rank: int = 0
    major: str = "N"


class Article(BaseModel):
    """A news article from NYT."""

    id: UUID = Field(default_factory=uuid4)
    web_url: str
    snippet: Optional[str] = None
    lead_paragraph: Optional[str] = None
    abstract: Optional[str] = None
    headline: Headline
    byline: Optional[Byline] = None
    source: str = "The New York Times"
    pub_date: datetime
    document_type: str = "article"
    section_name: Optional[str] = None
    subsection_name: Optional[str] = None
    keywords: list[Keyword] = Field(default_factory=list)
    word_count: int = 0

    # Full text (fetched separately if needed)
    full_text: Optional[str] = None

    @property
    def title(self) -> str:
        """Get the main headline."""
        return self.headline.main


class ArticleSearchResponse(BaseModel):
    """Response from NYT Article Search API."""

    status: str
    copyright: str
    articles: list[Article] = Field(default_factory=list)
    total_hits: int = 0


class NYTAPIResponse(BaseModel):
    """Raw NYT API response wrapper."""

    status: str
    copyright: str
    response: dict


class SearchMeta(BaseModel):
    """Metadata about a search query."""

    query: str
    begin_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    page: int = 0
    total_hits: int = 0


class ArchiveResponse(BaseModel):
    """Response from NYT Archive API."""

    copyright: str
    year: int
    month: int
    articles: list[Article] = Field(default_factory=list)
    total_hits: int = 0
