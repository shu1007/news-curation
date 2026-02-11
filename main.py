#!/usr/bin/env python3
"""RSS Feed Aggregator with LLM Summarization via Ollama."""

import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import requests
import yaml
from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader

# Paths
BASE_DIR = Path(__file__).resolve().parent
FEEDS_FILE = BASE_DIR / "feeds.yaml"
ARTICLES_FILE = BASE_DIR / "data" / "articles.json"
OUTPUT_FILE = BASE_DIR / "docs" / "index.html"
TEMPLATES_DIR = BASE_DIR / "templates"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"
RETENTION_DAYS = 7
LOG_DIR = BASE_DIR / "logs"


def setup_logging() -> None:
    """Configure logging to both stdout and a timestamped log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"run_{timestamp}.log"

    log = logging.getLogger()
    log.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    # Stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    logging.info(f"Log file: {log_file}")


def load_feeds() -> list[dict]:
    """Load feed definitions from feeds.yaml."""
    with open(FEEDS_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("feeds", [])


def load_existing_articles() -> list[dict]:
    """Load previously saved articles from JSON."""
    if not ARTICLES_FILE.exists():
        return []
    with open(ARTICLES_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_articles(articles: list[dict]) -> None:
    """Save articles to JSON file."""
    ARTICLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTICLES_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)


def is_japanese(text: str) -> bool:
    """Check if text contains Japanese characters (hiragana, katakana, kanji)."""
    return bool(re.search(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", text))


def parse_published(entry) -> datetime | None:
    """Parse published date from a feed entry."""
    for attr in ("published_parsed", "updated_parsed"):
        tp = getattr(entry, attr, None)
        if tp:
            from time import mktime

            return datetime.fromtimestamp(mktime(tp), tz=timezone.utc)
    return None


def extract_image_from_entry(entry) -> str:
    """Extract image URL from RSS entry fields."""
    # hatena_imageurl (はてなブックマーク)
    hatena_img = entry.get("hatena_imageurl", "")
    if hatena_img:
        return hatena_img

    # media_content
    media = entry.get("media_content", [])
    if media:
        for m in media:
            if m.get("medium") == "image" or (m.get("type", "").startswith("image")):
                return m.get("url", "")
        if media[0].get("url"):
            return media[0]["url"]

    # media_thumbnail
    media_thumb = entry.get("media_thumbnail", [])
    if media_thumb and media_thumb[0].get("url"):
        return media_thumb[0]["url"]

    # enclosure with image type
    for enc in entry.get("enclosures", []):
        if enc.get("type", "").startswith("image"):
            return enc.get("href", "")

    # <img> tag in summary/description
    summary_html = entry.get("summary", entry.get("description", ""))
    if summary_html:
        imgs = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', summary_html)
        if imgs:
            return imgs[0]

    return ""


def fetch_feed_articles(feed: dict, cutoff: datetime) -> list[dict]:
    """Fetch articles from a single feed, filtering by cutoff date."""
    logging.info(f"  Fetching: {feed['name']} ({feed['url']})")
    try:
        d = feedparser.parse(feed["url"])
    except Exception as e:
        logging.info(f"    Error parsing feed: {e}")
        return []

    articles = []
    for entry in d.entries:
        published = parse_published(entry)
        if published is None:
            published = datetime.now(tz=timezone.utc)
        if published < cutoff:
            continue

        title = entry.get("title", "No Title")
        url = entry.get("link", "")
        description = entry.get("summary", entry.get("description", ""))
        # Strip HTML tags from description
        description = re.sub(r"<[^>]+>", "", description).strip()

        image_url = extract_image_from_entry(entry)

        articles.append(
            {
                "feed_name": feed["name"],
                "title": title,
                "title_ja": "",
                "url": url,
                "published": published.isoformat(),
                "summary": "",
                "description": description[:1000],
                "is_english": not is_japanese(title),
                "image_url": image_url,
            }
        )

    logging.info(f"    Found {len(articles)} articles within {RETENTION_DAYS} days")
    return articles


CONTENT_MAX_CHARS = 3000

_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RSSAggregator/1.0)",
}


def fetch_article_content(url: str) -> tuple[str, str]:
    """Fetch article URL and extract main text content and og:image from HTML.

    Returns:
        (text_content, og_image_url)
    """
    try:
        resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        logging.info(f"    -> 本文取得: 失敗({e.__class__.__name__})")
        return "", ""

    soup = BeautifulSoup(resp.content, "html.parser")

    # Extract og:image before decomposing elements
    og_image = ""
    og_tag = soup.find("meta", property="og:image")
    if og_tag and og_tag.get("content"):
        og_image = og_tag["content"]

    # Remove non-content elements
    for tag in soup.select("script, style, nav, header, footer, aside, form, iframe, noscript"):
        tag.decompose()

    # Try common article containers first
    article_el = (
        soup.select_one("article")
        or soup.select_one('[role="main"]')
        or soup.select_one("main")
        or soup.select_one(".post-content, .entry-content, .article-body, .content")
    )

    target = article_el if article_el else soup.body if soup.body else soup
    text = target.get_text(separator="\n", strip=True)

    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text[:CONTENT_MAX_CHARS], og_image


def call_ollama(prompt: str) -> str:
    """Call Ollama API and return generated text."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        logging.warning("Ollama is not running. Skipping LLM summarization.")
        return ""
    except Exception as e:
        logging.warning(f"Ollama API error: {e}")
        return ""


def summarize_article(article: dict) -> str:
    """Generate a Japanese summary of an article using Ollama."""
    title = article["title"]
    desc = article.get("description", "")
    content = f"タイトル: {title}\n内容: {desc}" if desc else f"タイトル: {title}"

    prompt = (
        "以下の記事を日本語で要約してください。\n"
        "制約:\n"
        "- 100〜200字程度\n"
        "- 必ず文末を「。」で終えること\n"
        "- 要約のみを出力（前置きや補足は不要）\n\n"
        f"{content}"
    )
    return call_ollama(prompt)


def generate_labels(article: dict) -> list[str]:
    """Generate 2-3 labels for an article using Ollama."""
    title = article["title"]
    summary = article.get("summary", "")
    prompt = (
        "以下の記事にふさわしいラベルを2〜3個生成してください。\n"
        "制約:\n"
        "- 英語の小文字で出力（例: programming, rust, apple）\n"
        "- カンマ区切りで出力（例: ai, programming, python）\n"
        "- 汎用的なカテゴリ（例: programming, tech, security, devops）と、\n"
        "  記事固有のトピック（例: rust, iphone, aws, docker）を混ぜること\n"
        "- ラベルのみを出力（前置きや説明は不要）\n\n"
        f"タイトル: {title}\n要約: {summary}"
    )
    result = call_ollama(prompt)
    if not result:
        return []
    # Parse comma-separated labels, clean up
    labels = [l.strip().lower().strip('"\'') for l in result.split(",")]
    labels = [l for l in labels if l and len(l) < 30 and " " not in l.strip()]
    return labels[:3]


def translate_to_japanese(text: str) -> str:
    """Translate English text to Japanese using Ollama."""
    prompt = (
        "以下の英語テキストを自然な日本語に翻訳してください。翻訳のみを出力してください。\n\n"
        f"{text}"
    )
    return call_ollama(prompt)


def _save_progress(articles: list[dict]) -> None:
    """Save intermediate progress (strips description field from copies)."""
    to_save = [{k: v for k, v in a.items() if k != "description"} for a in articles]
    save_articles(to_save)


def process_articles(new_articles: list[dict], existing_articles: list[dict]) -> list[dict]:
    """Process new articles: summarize and translate, merge with existing."""
    existing_urls = {a["url"] for a in existing_articles}

    to_process = [a for a in new_articles if a["url"] not in existing_urls]
    logging.info(f"\nNew articles to process: {len(to_process)}")

    processed = list(existing_articles)
    total = len(to_process)

    for i, article in enumerate(to_process, 1):
        label = "EN" if article["is_english"] else "JA"
        logging.info(f"  [{i}/{total}] [{label}] {article['title'][:60]}")

        # Fetch full article content from HTML
        content, og_image = fetch_article_content(article["url"])
        if content:
            article["description"] = content
            logging.info(f"    -> 本文取得: OK ({len(content)}字)")
        else:
            logging.info(f"    -> 本文取得: フォールバック(RSS概要を使用)")

        # Use og:image as fallback if no RSS image
        if not article.get("image_url") and og_image:
            article["image_url"] = og_image
            logging.info(f"    -> 画像: og:image取得")
        elif article.get("image_url"):
            logging.info(f"    -> 画像: RSS取得")
        else:
            logging.info(f"    -> 画像: なし")

        # Summarize
        summary = summarize_article(article)
        article["summary"] = summary
        if summary:
            logging.info(f"    -> 要約: OK ({len(article['summary'])}字)")
        else:
            logging.info(f"    -> 要約: スキップ")

        # Translate English articles
        if article["is_english"] and article["title"]:
            title_ja = translate_to_japanese(article["title"])
            article["title_ja"] = title_ja if title_ja else article["title"]
            if title_ja:
                logging.info(f"    -> 翻訳: OK — {article['title_ja'][:40]}")
            else:
                logging.info(f"    -> 翻訳: スキップ")
        else:
            article["title_ja"] = article["title"]

        # Generate labels
        labels = generate_labels(article)
        article["labels"] = labels
        if labels:
            logging.info(f"    -> ラベル: {', '.join(labels)}")
        else:
            logging.info(f"    -> ラベル: スキップ")

        processed.append(article)

        # Save progress every 5 articles
        if i % 5 == 0:
            logging.info(f"  --- 中間保存 ({i}/{total}) ---")
            _save_progress(processed)

    return processed


def prune_old_articles(articles: list[dict], cutoff: datetime) -> list[dict]:
    """Remove articles older than the cutoff date."""
    result = []
    for a in articles:
        try:
            pub = datetime.fromisoformat(a["published"])
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if pub >= cutoff:
                result.append(a)
        except (ValueError, KeyError):
            result.append(a)
    return result


def render_html(articles: list[dict]) -> None:
    """Render HTML from template and articles."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Sort by published date descending
    sorted_articles = sorted(
        articles,
        key=lambda a: a.get("published", ""),
        reverse=True,
    )

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)
    template = env.get_template("index.html.j2")

    # Collect unique feed names (in order of appearance) and labels (by frequency)
    seen_feeds = []
    label_count: dict[str, int] = {}
    for a in sorted_articles:
        fn = a.get("feed_name", "")
        if fn and fn not in seen_feeds:
            seen_feeds.append(fn)
        for l in a.get("labels", []):
            label_count[l] = label_count.get(l, 0) + 1
    all_labels = sorted(label_count.keys(), key=lambda l: -label_count[l])

    html = template.render(
        articles=sorted_articles,
        updated_at=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        total_count=len(sorted_articles),
        feed_names=seen_feeds,
        all_labels=all_labels,
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    logging.info(f"\nGenerated: {OUTPUT_FILE}")


def main():
    setup_logging()
    logging.info("=== RSS Feed Aggregator ===")

    # Load feeds
    feeds = load_feeds()
    logging.info(f"Loaded {len(feeds)} feeds\n")

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=RETENTION_DAYS)

    # Fetch articles from all feeds
    logging.info("Fetching articles...")
    new_articles = []
    for feed in feeds:
        new_articles.extend(fetch_feed_articles(feed, cutoff))
    logging.info(f"\nTotal fetched: {len(new_articles)} articles")

    # Load existing and process
    existing = load_existing_articles()
    logging.info(f"Existing articles: {len(existing)}")

    all_articles = process_articles(new_articles, existing)

    # Prune old articles
    all_articles = prune_old_articles(all_articles, cutoff)
    logging.info(f"Articles after pruning: {len(all_articles)}")

    # Remove description field (only used for LLM input)
    for a in all_articles:
        a.pop("description", None)

    # Save and render
    save_articles(all_articles)
    logging.info(f"Saved {len(all_articles)} articles to {ARTICLES_FILE}")
    render_html(all_articles)

    logging.info("\nDone!")


if __name__ == "__main__":
    main()
