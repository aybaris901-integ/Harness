"""Article pipeline: fetch a web page and pull out the readable text.

`trafilatura` does the heavy lifting (strips navigation, ads, comments). The
module knows nothing about LLMs — it returns an `Article` for the summarizer.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx
import trafilatura

logger = logging.getLogger(__name__)

# Two identities, tried in order. Wikipedia and friends 403 anything that
# claims to be a browser but does not behave like one, while many news sites
# 403 anything that admits to being a bot — so try honest first, then browser.
_BOT_UA = "HarnessBot/0.2 (personal Telegram summarizer; +https://github.com/local/harness-bot)"
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_BASE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "kk,en;q=0.9,ru;q=0.8",
}
_RETRY_WITH_BROWSER_UA = {401, 403, 406, 429}
_MAX_HTML_BYTES = 5 * 1024 * 1024
# Below this the "article" is probably a paywall stub, a consent wall or a JS app shell.
MIN_ARTICLE_CHARS = 200


class ArticleError(RuntimeError):
    """User-presentable failure of the article pipeline."""


@dataclass(slots=True)
class Article:
    url: str
    title: str | None
    text: str
    author: str | None = None
    date: str | None = None
    site: str | None = None


async def fetch_html(url: str, *, timeout: float = 30.0) -> str:
    try:
        async with httpx.AsyncClient(
            headers=_BASE_HEADERS, follow_redirects=True, timeout=timeout
        ) as client:
            response = await client.get(url, headers={"User-Agent": _BOT_UA})
            if response.status_code in _RETRY_WITH_BROWSER_UA:
                logger.info(
                    "%s answered %d to the bot UA, retrying as a browser", url, response.status_code
                )
                response = await client.get(url, headers={"User-Agent": _BROWSER_UA})
    except httpx.HTTPError as exc:
        raise ArticleError(f"could not fetch the page: {exc}") from exc

    if response.status_code >= 400:
        raise ArticleError(f"the site answered HTTP {response.status_code}")

    content_type = response.headers.get("content-type", "")
    if content_type and not any(t in content_type for t in ("html", "xml", "text/plain")):
        raise ArticleError(f"not a web page (content-type: {content_type.split(';')[0]})")

    if len(response.content) > _MAX_HTML_BYTES:
        raise ArticleError("the page is too large to process")
    return response.text


def extract_article(html: str, url: str) -> Article:
    """Run trafilatura over already-fetched HTML. Synchronous and CPU-bound."""
    doc = trafilatura.bare_extraction(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
        with_metadata=True,
    )
    text = (doc.text or "").strip() if doc is not None else ""
    if len(text) < MIN_ARTICLE_CHARS:
        raise ArticleError(
            "could not find article text on that page (paywall, login wall or a "
            "JavaScript-only site?)"
        )
    assert doc is not None
    return Article(
        url=url,
        title=(doc.title or "").strip() or None,
        text=text,
        author=(doc.author or None),
        date=(doc.date or None),
        site=(doc.sitename or None),
    )


async def load_article(url: str) -> Article:
    """Fetch + extract. The extraction step runs in a worker thread."""
    html = await fetch_html(url)
    article = await asyncio.to_thread(extract_article, html, url)
    logger.info("article %s: %d chars, title=%r", url, len(article.text), article.title)
    return article
