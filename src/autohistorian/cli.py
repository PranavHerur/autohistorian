"""Command-line interface for AutoHistorian."""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import get_settings

app = typer.Typer(
    name="autohistorian",
    help="Synthesize Wikipedia-style articles from news sources with dual timelines.",
)
console = Console()


def get_api_keys() -> tuple[str, str]:
    """Get API keys from settings (loaded from .env or environment)."""
    settings = get_settings()

    if not settings.nyt_api_key:
        console.print("[red]Error: AUTOHISTORIAN_NYT_API_KEY not set[/red]")
        console.print("Set it in .env file or as environment variable")
        raise typer.Exit(1)
    if not settings.gemini_api_key:
        console.print("[red]Error: AUTOHISTORIAN_GEMINI_API_KEY not set[/red]")
        console.print("Set it in .env file or as environment variable")
        raise typer.Exit(1)

    return settings.nyt_api_key, settings.gemini_api_key


# Available NYT sections for filtering
NYT_SECTIONS = ["U.S.", "World", "Politics", "Business", "Technology", "Science", "Health", "Sports", "Arts", "Opinion"]


@app.command()
def ingest(
    query: Optional[str] = typer.Argument(None, help="Search query (optional - if omitted, fetches recent news)"),
    days: int = typer.Option(7, "--days", "-d", help="Number of days to look back"),
    max_articles: int = typer.Option(50, "--max", "-m", help="Maximum articles to fetch"),
    sections: Optional[str] = typer.Option(None, "--sections", "-s", help="Comma-separated sections (e.g., 'Politics,U.S.')"),
    model: Optional[str] = typer.Option(None, "--model", help="Gemini model to use (e.g., gemini-2.5-flash-preview-05-20)"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Data directory"),
):
    """Ingest articles from the last N days and auto-discover topics.

    If no query is provided, fetches recent articles from major news sections.
    Topics are automatically extracted from article content.

    Examples:
        autohistorian ingest --days 7                    # Recent news, auto-discover topics
        autohistorian ingest "immigration" --days 30    # Search for specific topic
        autohistorian ingest --sections "Politics,U.S." # Filter by sections
    """
    nyt_key, gemini_key = get_api_keys()
    settings = get_settings()
    data_dir = data_dir or settings.data_dir

    # Parse sections
    section_list = None
    if sections:
        section_list = [s.strip() for s in sections.split(",")]

    async def run():
        from .extract.pipeline import ExtractionPipeline
        from .ingest.nyt_client import NYTClient
        from .knowledge.store import KnowledgeStore
        from .llm.client import GeminiClient

        nyt_client = NYTClient(nyt_key)
        llm_client = GeminiClient(gemini_key, model=model or settings.gemini_model)
        store = KnowledgeStore(data_dir)
        pipeline = ExtractionPipeline(llm_client)

        discovered_topics: set[str] = set()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Fetch articles
            if query:
                task = progress.add_task(f"Fetching articles for '{query}'...", total=None)
                articles = await nyt_client.search_recent(
                    query=query, days=days, max_articles=max_articles, sections=section_list
                )
            else:
                task = progress.add_task(f"Fetching recent articles (last {days} days)...", total=None)
                articles = await nyt_client.fetch_recent(
                    days=days, max_articles=max_articles, sections=section_list
                )
            progress.update(task, description=f"Found {len(articles)} articles")

            if not articles:
                console.print("[yellow]No articles found[/yellow]")
                return

            # Save all articles first
            for article in articles:
                store.save_article(article)

            # Process articles in parallel batches
            task = progress.add_task(f"Extracting from {len(articles)} articles (parallel)...", total=None)
            results = await pipeline.extract_batch(articles, topic=None)

            # Save results and track topics
            for result in results:
                for t in result.topics:
                    discovered_topics.add(t.name)
                store.save_extraction_result(result, topic=None)

            progress.update(task, description=f"Processed {len(articles)} articles")

        # Print summary
        stats = store.get_stats()
        console.print()
        console.print("[green]Ingestion complete![/green]")
        console.print(f"  Articles processed: {len(articles)}")
        console.print(f"  Total articles: {stats['articles']}")
        console.print(f"  Total events: {stats['events']}")
        console.print(f"  Total statements: {stats['statements']}")
        console.print(f"  Total topics: {stats['topics']}")

        if discovered_topics:
            console.print()
            console.print("[cyan]Discovered topics:[/cyan]")
            for topic in sorted(discovered_topics)[:15]:
                console.print(f"  - {topic}")
            if len(discovered_topics) > 15:
                console.print(f"  ... and {len(discovered_topics) - 15} more (use 'autohistorian topics' to see all)")

    asyncio.run(run())


@app.command()
def generate(
    topic: str = typer.Argument(..., help="Topic to generate article for"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    perspectives: bool = typer.Option(False, "--perspectives", "-p", help="Include perspectives section"),
    model: Optional[str] = typer.Option(None, "--model", help="Gemini model to use"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Data directory"),
):
    """Generate a Wikipedia-style article for a topic."""
    _, gemini_key = get_api_keys()
    settings = get_settings()
    data_dir = data_dir or settings.data_dir

    async def run():
        from .knowledge.store import KnowledgeStore
        from .llm.client import GeminiClient
        from .synthesize.writer import ArticleWriter

        llm_client = GeminiClient(gemini_key, model=model or settings.gemini_model)
        store = KnowledgeStore(data_dir)
        writer = ArticleWriter(llm_client, store)

        # Check if topic exists
        if topic not in store.get_topics():
            console.print(f"[yellow]Topic '{topic}' not found. Available topics:[/yellow]")
            for t in store.get_topics():
                console.print(f"  - {t}")
            raise typer.Exit(1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating article...", total=None)

            if perspectives:
                article = await writer.generate_with_perspectives(topic)
            else:
                article = await writer.generate_article(topic)

            progress.update(task, description="Article generated!")

        if output:
            Path(output).write_text(article)
            console.print(f"[green]Article saved to {output}[/green]")
        else:
            console.print()
            console.print(article)

    asyncio.run(run())


@app.command()
def timeline(
    topic: str = typer.Argument(..., help="Topic to show timeline for"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, timelinejs"),
    valid_time: bool = typer.Option(True, "--valid-time/--obs-time", help="Use valid time (when happened) or observation time (when reported)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    model: Optional[str] = typer.Option(None, "--model", help="Gemini model to use"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Data directory"),
):
    """Display or export timeline for a topic."""
    from .knowledge.store import KnowledgeStore
    from .llm.client import GeminiClient
    from .synthesize.writer import ArticleWriter

    settings = get_settings()
    data_dir = data_dir or settings.data_dir
    store = KnowledgeStore(data_dir)

    # Check if topic exists
    if topic not in store.get_topics():
        console.print(f"[yellow]Topic '{topic}' not found. Available topics:[/yellow]")
        for t in store.get_topics():
            console.print(f"  - {t}")
        raise typer.Exit(1)

    items = store.get_timeline(topic, use_valid_time=valid_time)

    if format == "json":
        output_data = json.dumps(items, indent=2)
        if output:
            Path(output).write_text(output_data)
            console.print(f"[green]Timeline saved to {output}[/green]")
        else:
            console.print(output_data)

    elif format == "timelinejs":
        # Need LLM client for writer
        _, gemini_key = get_api_keys()
        llm_client = GeminiClient(gemini_key, model=model or settings.gemini_model)
        writer = ArticleWriter(llm_client, store)
        output_data = json.dumps(writer.export_timeline_json(topic), indent=2)
        if output:
            Path(output).write_text(output_data)
            console.print(f"[green]TimelineJS data saved to {output}[/green]")
        else:
            console.print(output_data)

    else:  # table format
        time_label = "When Happened" if valid_time else "When Reported"
        table = Table(title=f"Timeline: {topic}")
        table.add_column(time_label, style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Description")

        for item in items:
            time_str = item["time"][:10] if item["time"] else "Unknown"
            item_type = item["type"].capitalize()
            desc = item.get("description") or item.get("content", "")
            if len(desc) > 80:
                desc = desc[:77] + "..."
            table.add_row(time_str, item_type, desc)

        console.print(table)


@app.command()
def topics(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum topics to show"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Data directory"),
):
    """List all auto-discovered topics in the knowledge store.

    Topics are automatically extracted from articles and sorted by coverage.
    """
    from .knowledge.store import KnowledgeStore

    settings = get_settings()
    data_dir = data_dir or settings.data_dir
    store = KnowledgeStore(data_dir)
    topics_info = store.get_all_topics_info()

    if not topics_info:
        console.print("[yellow]No topics found. Use 'autohistorian ingest' to add articles.[/yellow]")
        return

    # Filter by category if specified
    if category:
        topics_info = [t for t in topics_info if t["category"] == category]

    # Limit results
    topics_info = topics_info[:limit]

    table = Table(title="Discovered Topics (sorted by coverage)")
    table.add_column("Topic", style="cyan", max_width=50)
    table.add_column("Category", style="magenta")
    table.add_column("Articles", justify="right")
    table.add_column("Events", justify="right")
    table.add_column("Statements", justify="right")

    for info in topics_info:
        table.add_row(
            info["name"],
            info["category"],
            str(info["article_count"]),
            str(info["event_count"]),
            str(info["statement_count"]),
        )

    console.print(table)

    # Show available categories
    all_topics = store.get_all_topics_info()
    categories = set(t["category"] for t in all_topics)
    if len(categories) > 1:
        console.print()
        console.print(f"[dim]Categories: {', '.join(sorted(categories))}[/dim]")
        console.print("[dim]Use --category to filter[/dim]")


@app.command()
def stats(
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Data directory"),
):
    """Show statistics about the knowledge store."""
    from .knowledge.store import KnowledgeStore

    settings = get_settings()
    data_dir = data_dir or settings.data_dir
    store = KnowledgeStore(data_dir)
    store_stats = store.get_stats()

    table = Table(title="Knowledge Store Statistics")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="green")

    for key, value in store_stats.items():
        table.add_row(key.capitalize(), str(value))

    console.print(table)


@app.command("ingest-archive")
def ingest_archive(
    year: int = typer.Argument(..., help="Year of archive to ingest"),
    month: int = typer.Argument(..., help="Month of archive to ingest (1-12)"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Filter articles by keyword in headline/abstract"),
    sections: Optional[str] = typer.Option(None, "--sections", "-s", help="Comma-separated sections to filter"),
    max_articles: int = typer.Option(100, "--max", "-m", help="Maximum articles to process"),
    model: Optional[str] = typer.Option(None, "--model", help="Gemini model to use"),
    archive_dir: Optional[str] = typer.Option(None, "--archive-dir", help="Archive directory"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Data directory for knowledge store"),
):
    """Ingest articles from a local archive file into the knowledge store.

    Loads articles from a previously downloaded archive and runs them through
    the extraction pipeline to identify events, statements, and topics.

    Examples:
        autohistorian ingest-archive 2025 12                    # All of Dec 2025
        autohistorian ingest-archive 2025 12 -q "immigration"   # Filter by topic
        autohistorian ingest-archive 2025 12 -s "Politics,U.S." # Filter by section
        autohistorian ingest-archive 2025 12 -m 50              # Limit to 50 articles
    """
    _, gemini_key = get_api_keys()
    settings = get_settings()
    data_dir = data_dir or settings.data_dir
    archive_dir = Path(archive_dir) if archive_dir else Path(settings.data_dir) / "archive"

    # Load archive file
    archive_file = archive_dir / f"{year}-{month:02d}.json"
    if not archive_file.exists():
        console.print(f"[red]Archive file not found: {archive_file}[/red]")
        console.print("Run 'autohistorian crawl-archive' to download archives first.")
        raise typer.Exit(1)

    console.print(f"[cyan]Loading archive: {archive_file}[/cyan]")

    with open(archive_file) as f:
        archive_data = json.load(f)

    # Parse sections filter
    section_list = None
    if sections:
        section_list = [s.strip().lower() for s in sections.split(",")]

    # Filter and convert articles
    from .ingest.schemas import Article, Byline, Headline, Keyword

    articles = []
    skipped_empty = 0
    for article_data in archive_data["articles"]:
        # Skip articles with no meaningful content
        headline_text = article_data.get("headline", {}).get("main", "")
        abstract_text = article_data.get("abstract") or ""
        if not headline_text and not abstract_text:
            skipped_empty += 1
            continue

        # Apply section filter
        if section_list:
            section = (article_data.get("section_name") or "").lower()
            if section not in section_list:
                continue

        # Apply query filter
        if query:
            query_lower = query.lower()
            headline = article_data.get("headline", {}).get("main", "").lower()
            abstract = (article_data.get("abstract") or "").lower()
            snippet = (article_data.get("snippet") or "").lower()
            if query_lower not in headline and query_lower not in abstract and query_lower not in snippet:
                continue

        # Convert to Article model
        headline_data = article_data.get("headline", {})
        headline = Headline(
            main=headline_data.get("main", ""),
            print_headline=headline_data.get("print_headline"),
        )

        byline_data = article_data.get("byline")
        byline = None
        if byline_data:
            byline = Byline(
                original=byline_data.get("original"),
                organization=byline_data.get("organization"),
            )

        keywords = []
        for kw in article_data.get("keywords") or []:
            keywords.append(Keyword(
                name=kw.get("name", ""),
                value=kw.get("value", ""),
                rank=kw.get("rank", 0),
                major=kw.get("major", "N"),
            ))

        from datetime import datetime
        pub_date_str = article_data.get("pub_date", "")
        try:
            pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pub_date = datetime.utcnow()

        from uuid import UUID
        article = Article(
            id=UUID(article_data["id"]),
            web_url=article_data.get("web_url", ""),
            snippet=article_data.get("snippet"),
            lead_paragraph=article_data.get("lead_paragraph"),
            abstract=article_data.get("abstract"),
            headline=headline,
            byline=byline,
            source=article_data.get("source", "The New York Times"),
            pub_date=pub_date,
            document_type=article_data.get("document_type", "article"),
            section_name=article_data.get("section_name"),
            subsection_name=article_data.get("subsection_name"),
            keywords=keywords,
            word_count=article_data.get("word_count", 0),
        )
        articles.append(article)

        if len(articles) >= max_articles:
            break

    if not articles:
        console.print("[yellow]No articles matched the filters[/yellow]")
        raise typer.Exit(1)

    console.print(f"[cyan]Found {len(articles)} articles to process[/cyan]")
    if skipped_empty:
        console.print(f"[dim]Skipped {skipped_empty} articles with empty content[/dim]")

    async def run():
        from .extract.pipeline import ExtractionPipeline
        from .knowledge.store import KnowledgeStore
        from .llm.client import GeminiClient

        llm_client = GeminiClient(gemini_key, model=model or settings.gemini_model)
        store = KnowledgeStore(data_dir)
        pipeline = ExtractionPipeline(llm_client)

        discovered_topics: set[str] = set()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Save all articles first
            for article in articles:
                store.save_article(article)

            # Process articles in parallel batches
            task = progress.add_task(f"Extracting from {len(articles)} articles...", total=None)
            results = await pipeline.extract_batch(articles, topic=None)

            # Save results and track topics
            for result in results:
                for t in result.topics:
                    discovered_topics.add(t.name)
                store.save_extraction_result(result, topic=None)

            progress.update(task, description=f"Processed {len(articles)} articles")

        # Print summary
        stats = store.get_stats()
        console.print()
        console.print("[green]Ingestion complete![/green]")
        console.print(f"  Articles processed: {len(articles)}")
        console.print(f"  Total articles: {stats['articles']}")
        console.print(f"  Total events: {stats['events']}")
        console.print(f"  Total statements: {stats['statements']}")
        console.print(f"  Total topics: {stats['topics']}")

        if discovered_topics:
            console.print()
            console.print("[cyan]Discovered topics:[/cyan]")
            for topic in sorted(discovered_topics)[:15]:
                console.print(f"  - {topic}")
            if len(discovered_topics) > 15:
                console.print(f"  ... and {len(discovered_topics) - 15} more")

    asyncio.run(run())


@app.command("crawl-archive")
def crawl_archive(
    start_year: int = typer.Option(2026, "--start-year", "-y", help="Year to start crawling from"),
    start_month: int = typer.Option(1, "--start-month", "-m", help="Month to start crawling from (1-12)"),
    end_year: int = typer.Option(1851, "--end-year", help="Year to stop at"),
    end_month: int = typer.Option(9, "--end-month", help="Month to stop at"),
    daily_limit: int = typer.Option(500, "--limit", "-l", help="Daily request limit"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory for archive files"),
):
    """Crawl the NYT Archive API backwards from a starting date.

    Downloads all articles month by month, saving each as a JSON file.
    Stops when hitting the daily limit and outputs the resume command.

    Examples:
        autohistorian crawl-archive                          # Start from Jan 2026
        autohistorian crawl-archive -y 2024 -m 6             # Start from June 2024
        autohistorian crawl-archive -y 2020 -m 1 -o ./data   # Custom output dir
    """
    from .config import get_settings

    settings = get_settings()

    if not settings.nyt_api_key:
        console.print("[red]Error: AUTOHISTORIAN_NYT_API_KEY not set[/red]")
        raise typer.Exit(1)

    # Set output directory
    archive_dir = Path(output_dir) if output_dir else Path(settings.data_dir) / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    async def run():
        from .ingest.nyt_client import NYTClient

        client = NYTClient(settings.nyt_api_key)
        requests_made = 0
        current_year = start_year
        current_month = start_month

        console.print(f"[cyan]Starting archive crawl from {start_year}/{start_month:02d}[/cyan]")
        console.print(f"[cyan]Saving to: {archive_dir}[/cyan]")
        console.print()

        while (current_year, current_month) >= (end_year, end_month):
            # Check if we've hit the daily limit
            if requests_made >= daily_limit:
                console.print()
                console.print("[yellow]Daily limit reached![/yellow]")
                console.print()
                console.print("[green]To resume tomorrow, run:[/green]")
                console.print(f"  autohistorian crawl-archive -y {current_year} -m {current_month}")
                return

            # Check if file already exists (skip if already downloaded)
            output_file = archive_dir / f"{current_year}-{current_month:02d}.json"
            if output_file.exists():
                console.print(f"[dim]Skipping {current_year}/{current_month:02d} (already exists)[/dim]")
            else:
                # Fetch the archive
                try:
                    console.print(f"Fetching {current_year}/{current_month:02d}...", end=" ")
                    archive = await client.fetch_archive(current_year, current_month)
                    requests_made += 1

                    # Save to file
                    output_data = {
                        "year": archive.year,
                        "month": archive.month,
                        "total_articles": archive.total_hits,
                        "articles": [a.model_dump(mode="json") for a in archive.articles],
                    }
                    output_file.write_text(json.dumps(output_data, indent=2, default=str))

                    console.print(f"[green]{archive.total_hits} articles[/green] ({requests_made}/{daily_limit} requests)")

                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    console.print()
                    console.print("[green]To resume, run:[/green]")
                    console.print(f"  autohistorian crawl-archive -y {current_year} -m {current_month}")
                    return

            # Move to previous month
            current_month -= 1
            if current_month < 1:
                current_month = 12
                current_year -= 1

        console.print()
        console.print("[green]Archive crawl complete![/green]")
        console.print(f"Total requests: {requests_made}")

    asyncio.run(run())


if __name__ == "__main__":
    app()
