"""Prompts for LLM extraction and synthesis tasks."""

SYSTEM_PROMPT = """You are an expert at analyzing news articles and extracting structured information.
You identify events, statements, entities, and topics with precision and accuracy.
Always respond with valid JSON when asked for structured output."""

EVENT_EXTRACTION_PROMPT = """Analyze this news article and extract all events (things that happened).

For each event, identify:
- description: A clear, factual description of what happened
- event_type: Category (e.g., arrest, policy_change, statement, meeting, protest, legal_action, etc.)
- valid_time: When the event actually occurred (ISO format if known, null if unknown)
- participants: List of people/organizations involved
- location: Where it happened (if mentioned)

Article:
{article_text}

Respond with a JSON array of events:
[{{"description": "...", "event_type": "...", "valid_time": "...", "participants": [...], "location": "..."}}]"""

STATEMENT_EXTRACTION_PROMPT = """Analyze this news article and extract all notable statements or quotes.

For each statement, identify:
- content: The actual quote or paraphrased statement
- speaker: Who said it
- speaker_role: Their role/title if mentioned
- stance: Their position (pro, con, neutral) on the topic
- target: What the statement is about

Article:
{article_text}

Respond with a JSON array of statements:
[{{"content": "...", "speaker": "...", "speaker_role": "...", "stance": "...", "target": "..."}}]"""

ENTITY_EXTRACTION_PROMPT = """Analyze this news article and extract all named entities.

For each entity, identify:
- name: The entity's name
- entity_type: Category (person, organization, location, law, event_name, etc.)
- description: Brief description based on the article

Article:
{article_text}

Respond with a JSON array of entities:
[{{"name": "...", "entity_type": "...", "description": "..."}}]"""

TOPIC_EXTRACTION_PROMPT = """Analyze this news article and identify the main topics it covers.

For each topic, provide:
- name: A clear, specific topic name (e.g., "ICE Operations in Minnesota" not just "Immigration")
- category: One of: politics, law, international, economy, science, social, other
- relevance: How central this topic is to the article (0.0-1.0)

Article headline: {headline}
Article abstract: {abstract}

Respond with a JSON array of topics (max 5):
[{{"name": "...", "category": "...", "relevance": 0.9}}]"""

STANCE_DETECTION_PROMPT = """Analyze the following statement and determine the speaker's stance.

Statement: "{statement}"
Speaker: {speaker}
Context: {context}

Classify the stance as one of:
- pro: Supports or agrees with the subject
- con: Opposes or disagrees with the subject
- neutral: No clear position or balanced view

Respond with JSON: {{"stance": "...", "confidence": 0.9, "reasoning": "..."}}"""

OUTLINE_GENERATION_PROMPT = """Generate a Wikipedia-style article outline for the topic: {topic}

Based on the following events and statements:

Events:
{events}

Statements:
{statements}

Create an outline with:
1. A lead section summarizing the topic
2. Relevant sections organized chronologically or thematically
3. A timeline section with dual timelines (when events happened vs when reported)

Respond with a JSON outline:
{{
  "title": "...",
  "lead": "...",
  "sections": [
    {{"title": "...", "content_notes": "..."}}
  ]
}}"""

ARTICLE_SYNTHESIS_PROMPT = """Write a Wikipedia-style article section about: {topic}

Use these events and statements as source material:

Events:
{events}

Statements:
{statements}

Guidelines:
- Write in encyclopedic, neutral tone
- Cite sources using [Source: date] format
- Distinguish between when events happened and when they were reported
- Include notable quotes with attribution
- Be factual and avoid speculation

Write the article section in markdown format."""
