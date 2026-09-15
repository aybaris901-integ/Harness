"""URL detection and routing between the article and video pipelines.

Pure functions, no I/O. The bot layer uses `extract_urls` to decide whether a
message is a link at all; the harness uses `classify_url` to pick a pipeline.
"""

from __future__ import annotations

import re
from enum import StrEnum
from urllib.parse import urlsplit

# Deliberately loose: anything that looks like a scheme://host or www.host.
# Trailing punctuation that people type after a link is trimmed below.
_URL_RE = re.compile(
    r"""(?ix)
    \b(?:https?://|www\.)
    [^\s<>"'«»()]+
    """
)
_TRAILING_PUNCT = ".,;:!?)]}»'\""


class LinkKind(StrEnum):
    ARTICLE = "article"
    VIDEO = "video"


# Hosts whose links are (almost always) media yt-dlp can handle. Everything
# else is treated as an article; the harness falls back to the article path if
# yt-dlp reports that one of these URLs has no media after all (a text tweet).
_VIDEO_HOSTS = {
    "youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
    "tiktok.com",
    "vm.tiktok.com",
    "instagram.com",
    "vimeo.com",
    "twitch.tv",
    "dailymotion.com",
    "rutube.ru",
    "ok.ru",
    "vk.com",
    "vkvideo.ru",
    "twitter.com",
    "x.com",
    "facebook.com",
    "fb.watch",
}


def extract_urls(text: str | None) -> list[str]:
    """All URLs in `text`, in order, deduplicated, with a scheme prepended if missing."""
    if not text:
        return []
    found: list[str] = []
    for match in _URL_RE.finditer(text):
        url = match.group(0).rstrip(_TRAILING_PUNCT)
        if url.lower().startswith("www."):
            url = "https://" + url
        if url not in found:
            found.append(url)
    return found


def _host(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def classify_url(url: str) -> LinkKind:
    host = _host(url)
    if not host:
        return LinkKind.ARTICLE
    # Match the host and any subdomain (m.youtube.com, music.youtube.com, ...).
    for video_host in _VIDEO_HOSTS:
        if host == video_host or host.endswith("." + video_host):
            return LinkKind.VIDEO
    return LinkKind.ARTICLE


def is_video_url(url: str) -> bool:
    return classify_url(url) is LinkKind.VIDEO
