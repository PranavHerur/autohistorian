# AutoHistorian

Synthesize Wikipedia-style articles from news sources with dual timelines.

## Features

- **NYT API Integration**: Fetch articles via Article Search API or Archive API
- **Automatic Topic Discovery**: Extract topics from article content using LLM
- **Event Extraction**: Identify events with dual timeline (when happened vs when reported)
- **Statement Attribution**: Track quotes with speaker and stance detection
- **Article Synthesis**: Generate Wikipedia-style articles with perspectives

## Installation

```bash
uv sync
```

## Configuration

Create a `.env` file:

```
AUTOHISTORIAN_NYT_API_KEY=your_nyt_key
AUTOHISTORIAN_GEMINI_API_KEY=your_gemini_key
```

## Usage

### Crawl NYT Archives

Download historical archives (1 API call per month, back to 1851):

```bash
autohistorian crawl-archive -y 2025 -m 12
```

### Ingest from Archives

Process downloaded archives through the extraction pipeline:

```bash
autohistorian ingest-archive 2025 12 -m 100
```

### Generate Articles

Create a Wikipedia-style article for a discovered topic:

```bash
autohistorian generate "Donald Trump"
```

### View Topics

List all auto-discovered topics:

```bash
autohistorian topics
```

## Rate Limits

NYT API: 5 requests/minute, 500 requests/day
