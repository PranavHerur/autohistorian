"""Gemini LLM client for extraction and synthesis tasks."""

import asyncio
import json
import re
import time
from typing import Any

from google import genai
from google.genai import types

from .prompts import (
    ARTICLE_SYNTHESIS_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    EVENT_EXTRACTION_PROMPT,
    OUTLINE_GENERATION_PROMPT,
    STANCE_DETECTION_PROMPT,
    STATEMENT_EXTRACTION_PROMPT,
    SYSTEM_PROMPT,
)


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int = 20):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until we can make a request."""
        async with self._lock:
            now = time.monotonic()
            time_since_last = now - self._last_request_time
            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)
            self._last_request_time = time.monotonic()


class GeminiClient:
    """Client for Gemini API interactions with rate limiting and retries."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        requests_per_minute: int = 20,
        max_retries: int = 3,
    ):
        """Initialize the Gemini client.

        Args:
            api_key: Google AI API key
            model: Model to use (default: gemini-2.0-flash)
            requests_per_minute: Rate limit (default: 20, under Gemini's 25/min)
            max_retries: Max retries on rate limit errors
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.max_retries = max_retries

    def _extract_json(self, text: str) -> Any:
        """Extract JSON from LLM response, handling markdown code blocks."""
        if text is None:
            return None
        # Try to find JSON in code blocks first
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            text = json_match.group(1).strip()

        # Try to parse as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find array or object
            array_match = re.search(r"\[[\s\S]*\]", text)
            if array_match:
                try:
                    return json.loads(array_match.group())
                except json.JSONDecodeError:
                    pass

            obj_match = re.search(r"\{[\s\S]*\}", text)
            if obj_match:
                try:
                    return json.loads(obj_match.group())
                except json.JSONDecodeError:
                    pass

            return None

    async def _generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """Generate a response from the LLM with rate limiting and retries."""
        await self.rate_limiter.acquire()

        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.2,
                    ),
                )
                return response.text
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    # Rate limited, wait and retry
                    wait_time = (attempt + 1) * 10
                    await asyncio.sleep(wait_time)
                    continue
                raise

        return None

    async def extract_events(self, article_text: str) -> list[dict]:
        """Extract events from article text.

        Args:
            article_text: The article text to analyze

        Returns:
            List of event dictionaries
        """
        prompt = EVENT_EXTRACTION_PROMPT.format(article_text=article_text)
        response = await self._generate(prompt)
        result = self._extract_json(response)
        return result if isinstance(result, list) else []

    async def extract_statements(self, article_text: str) -> list[dict]:
        """Extract statements from article text.

        Args:
            article_text: The article text to analyze

        Returns:
            List of statement dictionaries
        """
        prompt = STATEMENT_EXTRACTION_PROMPT.format(article_text=article_text)
        response = await self._generate(prompt)
        result = self._extract_json(response)
        return result if isinstance(result, list) else []

    async def extract_entities(self, article_text: str) -> list[dict]:
        """Extract entities from article text.

        Args:
            article_text: The article text to analyze

        Returns:
            List of entity dictionaries
        """
        prompt = ENTITY_EXTRACTION_PROMPT.format(article_text=article_text)
        response = await self._generate(prompt)
        result = self._extract_json(response)
        return result if isinstance(result, list) else []

    async def detect_stance(
        self, statement: str, speaker: str, context: str
    ) -> dict:
        """Detect stance in a statement.

        Args:
            statement: The statement to analyze
            speaker: Who made the statement
            context: Surrounding context

        Returns:
            Stance detection result
        """
        prompt = STANCE_DETECTION_PROMPT.format(
            statement=statement, speaker=speaker, context=context
        )
        response = await self._generate(prompt)
        result = self._extract_json(response)
        return result if isinstance(result, dict) else {"stance": "neutral"}

    async def generate_outline(
        self, topic: str, events: list[dict], statements: list[dict]
    ) -> dict:
        """Generate an article outline.

        Args:
            topic: The topic to write about
            events: List of relevant events
            statements: List of relevant statements

        Returns:
            Article outline
        """
        events_text = json.dumps(events, indent=2, default=str)
        statements_text = json.dumps(statements, indent=2, default=str)

        prompt = OUTLINE_GENERATION_PROMPT.format(
            topic=topic, events=events_text, statements=statements_text
        )
        response = await self._generate(prompt)
        result = self._extract_json(response)
        return result if isinstance(result, dict) else {}

    async def synthesize_article(
        self, topic: str, events: list[dict], statements: list[dict]
    ) -> str:
        """Synthesize an article section.

        Args:
            topic: The topic to write about
            events: List of relevant events
            statements: List of relevant statements

        Returns:
            Article text in markdown
        """
        events_text = json.dumps(events, indent=2, default=str)
        statements_text = json.dumps(statements, indent=2, default=str)

        prompt = ARTICLE_SYNTHESIS_PROMPT.format(
            topic=topic, events=events_text, statements=statements_text
        )
        return await self._generate(prompt)
