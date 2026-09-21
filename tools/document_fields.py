"""Structured field extraction from raw OCR text (CLAUDE.md §9 "Document field
extraction", Phase 3).

Per-document-type regex/heuristics, not a model call: a passport or a Kazakh
vehicle registration has a predictable-enough layout that regex is sufficient,
and it keeps PII out of any LLM call entirely (CLAUDE.md §5). `extract_fields`
never calls an LLM — the optional `extract_with_local_llm` fallback exists for
text regex genuinely can't parse, but it talks to a local Ollama-compatible
server only, never `llm_router.py`, and is disabled by default (see
DOCUMENT_LOCAL_LLM_* in .env.example).

`score_query` implements the natural-language document search (CLAUDE.md §7
Phase 3 "Query handler"): also pure keyword/synonym matching, never an LLM —
PII values are never sent anywhere to answer a search.
"""

# `_LOCAL_LLM_PROMPT` is copied verbatim from CLAUDE.md §9; do not reflow it.
# ruff: noqa: E501

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExtractedDocument:
    document_type: str
    fields: dict[str, str] = field(default_factory=dict)
    raw_text: str = ""


# -- document type detection / query synonyms ---------------------------------

# Reused both to guess a newly OCR'd document's type and to score a search
# query against an already-stored one, so the two stay in sync automatically.
DOCUMENT_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "passport": (
        "passport",
        "паспорт",
        "төлқұжат",
        "жеке куәлік",
        "identity card",
        # ICAO data-page labels: Kazakhstan passports print these in Latin
        # script alongside Kazakh/Russian, and they OCR far more reliably
        # than the Cyrillic under a mismatched language pack (see
        # `_extract_mrz` below for the even more reliable MRZ signal).
        "given names",
        "date of birth",
        "date of expiry",
        "nationality",
        "signature of bearer",
    ),
    "vehicle_registration": (
        "vehicle registration",
        "техпаспорт",
        "тех паспорт",
        "свидетельство о регистрации",
        "мемлекеттік тіркеу туралы куәлік",
        "vin",
        "гос номер",
        "мемлекеттік нөмір",
        "көлік",
    ),
    "contract": ("шарт", "договор", "contract", "келісімшарт", "agreement"),
    "receipt": ("чек", "квитанция", "receipt", "түбіртек", "касса"),
}


_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")


def _keyword_regex(keyword: str) -> re.Pattern[str]:
    """Compile a keyword into a regex tolerant of Kazakh/Russian inflection.

    Russian/Kazakh grammatical cases don't just append a suffix, they can
    replace a word's final letter(s) — "квитанция" (nominative) becomes
    "квитанции" (genitive, as in "№ квитанции"), so a literal substring
    check silently stops matching depending on which case the surrounding
    sentence happens to use. Every Cyrillic/Kazakh word of 5+ letters gets
    truncated by its last 2 characters (its likely-invariant stem, e.g.
    "квитанц") and matched with a trailing `\\w*` so any case ending is
    accepted; short words and ASCII words (vin, receipt, ...) don't have
    this problem and are matched as-is. This is generic — it applies to
    every keyword automatically, not just ones we've hit a bug report for.
    """
    parts = []
    for word in keyword.split():
        stem = word[:-2] if _CYRILLIC_RE.search(word) and len(word) >= 5 else word
        parts.append(re.escape(stem) + r"\w*")
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b", re.IGNORECASE | re.UNICODE)


_DOCUMENT_TYPE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    doc_type: tuple(_keyword_regex(kw) for kw in keywords)
    for doc_type, keywords in DOCUMENT_TYPE_KEYWORDS.items()
}


# How many leading lines count as the document's "title" region for keyword
# detection below — chosen to comfortably fit a heading that wraps onto a
# second line plus a blank line or two from OCR noise, without reaching so
# far down that a form's field values start counting as titles.
_TITLE_LINES = 5

# How much slack (beyond the matched keyword itself) a non-title line is
# allowed before it stops counting as a short, label-like line — see
# `detect_document_type`.
_LABEL_LINE_SLACK = 10


def detect_document_type(text: str) -> str:
    """Keyword-only guess, used when no structural pattern below fired.

    This is the fallback layer — fragile by nature, since it requires the
    OCR'd text to literally contain a recognizable trigger word (see
    `_keyword_regex` for how "recognizable" tolerates inflection). A badly
    OCR'd Cyrillic page (wrong Tesseract language pack, poor scan) can lose
    every one of these words while still yielding clean digits/MRZ data;
    `extract_fields` below only calls this when nothing more specific matched.

    A keyword hit only counts when it's near the top of the document (the
    title region, `_TITLE_LINES`) or when it's on its own short, label-like
    line (`_LABEL_LINE_SLACK`) — not just anywhere in the body. Two real
    cases motivate this split:
      - A parcel label's field asking for ID details can contain a full
        sentence like "паспорт номер 1111 222222 выдан УФМС" — a bare
        substring hit there would misclassify the *label* as a passport.
        That line is long relative to the matched word, so it's excluded.
      - A Kaspi transfer receipt has no "receipt"-ish word in its own
        heading ("Перевод успешно совершен"), only in a field of its own,
        "№ квитанции" — but that line IS just the label, short and
        self-describing, so it correctly still counts.
    A genuinely well-scanned passport page prints "PASSPORT / ПАСПОРТ" as
    its literal heading (title region, always counts); a badly-OCR'd one is
    expected to rely on `_extract_mrz` for its type instead (see
    `extract_fields`, which checks MRZ before ever falling back to this
    function).
    """
    lines = text.splitlines()
    scores = {doc_type: 0 for doc_type in DOCUMENT_TYPE_KEYWORDS}
    for i, line in enumerate(lines):
        in_title = i < _TITLE_LINES
        for doc_type, patterns in _DOCUMENT_TYPE_PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(line)
                if not match:
                    continue
                if in_title or len(line.strip()) <= len(match.group(0)) + _LABEL_LINE_SLACK:
                    scores[doc_type] += 1
    best_type, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_type if best_score > 0 else "unknown"


# -- field concepts: shared by label canonicalization (extraction time) and
# -- synonym matching (search time) — see FIELD_CONCEPTS below for why these
# -- must be one table, not two.

_IIN_RE = re.compile(r"\b\d{12}\b")
_VIN_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
_DATE_RE = re.compile(r"\b(\d{2}[.\-/]\d{2}[.\-/]\d{4})\b")
_PLATE_RE = re.compile(r"\b\d{3}\s?[A-Z]{3}\s?\d{2,3}\b", re.IGNORECASE)
# "Label: value" / "Label - value" lines — the shape most receipts, contracts
# and registration certificates print their fields in. The separator must
# NOT sit between two digits: without that guard, a value-only line like
# "31.08.2026 21:48" (a date and time, no label at all) gets split on the
# colon inside "21:48" as if it were itself a "label: value" pair, producing
# a garbage field instead of leaving the line alone.
_LABEL_VALUE_RE = re.compile(r"^\s*([\w \-()/.]{2,40}?)\s*(?<!\d)[:\-–](?!\d)\s*(.+?)\s*$")

# ICAO 9303 TD3 machine-readable zone (passport data page): two 44-character
# lines in a fixed-width, OCR-friendly font. It is checked BEFORE the label
# scan because it typically survives even when the rest of the page's
# Cyrillic/Kazakh text is badly misread under the wrong Tesseract language
# pack — the single most reliable, script-independent passport signal
# available, and exactly the case that caused extraction and document_type to
# disagree (fields came from MRZ-adjacent digits; type detection needed
# literal Cyrillic keywords that OCR had mangled).
_MRZ_LINE2_RE = re.compile(
    r"^(?P<doc_number>[A-Z0-9<]{9})(?P<doc_check>[0-9<])"
    r"(?P<nationality>[A-Z<]{3})"
    r"(?P<birth>[0-9]{6})(?P<birth_check>[0-9<])"
    r"(?P<sex>[MF<])"
    r"(?P<expiry>[0-9]{6})(?P<expiry_check>[0-9<])"
    r"(?P<personal>[A-Z0-9<]{14})(?P<personal_check>[0-9<])"
    r"(?P<composite_check>[0-9<])$"
)

MAX_FIELDS = 20

# Canonical field key -> alias phrases (Kazakh/Russian/English) that mean the
# same concept. Used both to fold a freshly-OCR'd label into one stable key
# regardless of which language it happened to print in (extraction time,
# `_canonical_field_key`) and to match a search query in any of those same
# languages against whichever key a given document ended up with (query
# time, `score_query`). One table, not two — LANGUAGE_POLICY governs what the
# bot *outputs*, not what language a label was scanned in or a user searches
# in, so both directions need every alias (CLAUDE.md §8).
FIELD_CONCEPTS: dict[str, tuple[str, ...]] = {
    "iin": ("иин", "жеке сәйкестендіру нөмірі", "iin"),
    "vin": ("vin", "вин", "кузов нөмірі", "шасси нөмірі"),
    "plate_number": ("мемлекеттік нөмір", "гос номер", "plate", "нөмір белгісі", "мем нөмір"),
    "document_number": ("құжат нөмірі", "номер документа", "document number", "серия", "series"),
    "surname": ("тегі", "фамилия", "surname", "family name"),
    "given_names": ("аты", "имя", "given names", "given name", "first name"),
    "nationality": ("азаматтығы", "гражданство", "nationality"),
    "sex": ("жынысы", "пол", "sex"),
    "date_of_birth": ("туған күні", "дата рождения", "date of birth", "birth date", "born"),
    "date_of_issue": ("берілген күні", "дата выдачи", "date of issue", "issue date", "issued"),
    "date_of_expiry": (
        "жарамдылық мерзімі",
        "мерзімі",
        "срок действия",
        "date of expiry",
        "expiry date",
        "expiry",
    ),
    "authority": ("берген орган", "выдан", "authority"),
    "owner_name": ("иесі", "владелец", "owner"),
    # Generic catch-all — checked last so a more specific date_of_* concept
    # above always wins for a label like "Issue date" (which also contains
    # the word "date").
    "date": ("күні", "дата", "date"),
}

# Same inflection problem as DOCUMENT_TYPE_KEYWORDS/`_keyword_regex` applies
# here too — a label search must match "дата рождения" whether the OCR'd
# page (or the user's search query) happens to use that exact case or not.
_FIELD_CONCEPT_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    canonical: tuple(_keyword_regex(alias) for alias in aliases)
    for canonical, aliases in FIELD_CONCEPTS.items()
}


def _slugify(label: str) -> str:
    key = re.sub(r"[^\w]+", "_", label.strip().lower())
    return key.strip("_")[:40]


def _canonical_field_key(label: str) -> str:
    """Map a raw OCR'd label to a stable key, regardless of its language.

    Falls back to slugifying the label verbatim when it matches no known
    concept — an unrecognized field is still stored and still searchable by
    its own literal text, just without the multilingual synonym boost.
    """
    lowered = label.strip().lower()
    for canonical, patterns in _FIELD_CONCEPT_PATTERNS.items():
        if any(pattern.search(lowered) for pattern in patterns):
            return canonical
    return _slugify(label)


def _mrz_date(raw: str) -> str | None:
    if not raw.isdigit() or len(raw) != 6:
        return None
    yy, mm, dd = raw[:2], raw[2:4], raw[4:6]
    # TD3 doesn't encode the century. 00-49 as 2000s covers every document
    # issued so far, and an expiry date is always in the future anyway.
    century = "20" if int(yy) <= 49 else "19"
    return f"{dd}.{mm}.{century}{yy}"


def _extract_mrz(text: str) -> tuple[dict[str, str], str | None]:
    """TD3 MRZ line 2 -> fields, plus a "passport" type hint when it matches.

    Only line 2 is parsed: it is fully self-describing (fixed-width, no
    separators to get misread) and already carries everything line 1 would
    add except the name, which line 1's OCR noise makes far less reliable to
    parse. Returns ({}, None) when no line matches the TD3 layout.
    """
    for raw_line in text.splitlines():
        candidate = raw_line.strip().replace(" ", "")
        match = _MRZ_LINE2_RE.match(candidate)
        if not match:
            continue
        fields: dict[str, str] = {"document_number": match.group("doc_number").rstrip("<")}
        nationality = match.group("nationality").rstrip("<")
        if nationality:
            fields["nationality"] = nationality
        birth = _mrz_date(match.group("birth"))
        if birth:
            fields["date_of_birth"] = birth
        sex = match.group("sex")
        if sex in ("M", "F"):
            fields["sex"] = sex
        expiry = _mrz_date(match.group("expiry"))
        if expiry:
            fields["date_of_expiry"] = expiry
        personal = match.group("personal").rstrip("<")
        if personal:
            fields["iin"] = personal
        return fields, "passport"
    return {}, None


def _extract_generic(text: str) -> tuple[dict[str, str], str | None]:
    """Patterns with no assumption about document type, plus label:value lines.

    Returns a type hint alongside the fields for the few patterns that *are*
    unambiguous evidence of one type (a VIN or plate number only ever appears
    on a vehicle document) — see `extract_fields` for why that hint matters.
    """
    fields: dict[str, str] = {}
    type_hint: str | None = None

    iin = _IIN_RE.search(text)
    if iin:
        fields["iin"] = iin.group(0)

    vin = _VIN_RE.search(text)
    if vin:
        fields["vin"] = vin.group(0)
        type_hint = "vehicle_registration"

    plate = _PLATE_RE.search(text)
    if plate:
        fields["plate_number"] = plate.group(0).upper().replace(" ", "")
        type_hint = type_hint or "vehicle_registration"

    dates = _DATE_RE.findall(text)
    if dates:
        fields["date"] = dates[0]
        if len(dates) > 1 and dates[1] != dates[0]:
            fields["date_2"] = dates[1]

    for line in text.splitlines():
        if len(fields) >= MAX_FIELDS:
            break
        match = _LABEL_VALUE_RE.match(line)
        if not match:
            continue
        label, value = match.group(1).strip(), match.group(2).strip()
        if not value or len(label) < 2:
            continue
        key = _canonical_field_key(label)
        if key and key not in fields:
            fields[key] = value

    return fields, type_hint


# -- receipt-specific extraction (Kaspi-style transfer/payment receipts) -----
#
# Same per-type-parser philosophy as `_extract_mrz` above, not a generalized
# mechanism: Kaspi's app renders a receipt as a two-column (label, value)
# layout that Tesseract frequently linearizes into two separate blocks —
# every label, then every value, in reading order — with no ':'/'-' left on
# a shared line for `_LABEL_VALUE_RE` to find at all. These patterns search
# forward from a label's line for the next line shaped like *that field's*
# value (a long digit run for a receipt number, a "Surname I." name for a
# sender/recipient), skipping past any other label line in between — the
# value's shape is specific enough to not need an explicit list of labels to
# skip. Only invoked once `document_type` is already "receipt" (see
# `extract_fields`); the generic label:value matcher above is untouched, so
# passport/vehicle/Kazpochta-label behaviour can't regress from this.

_AMOUNT_RE = re.compile(r"(\d[\d ]*\d|\d)\s*(?:₸|тенге|теңге|т\.?)\b", re.IGNORECASE)
_RECEIPT_NUMBER_LABEL_RE = re.compile(r"квитанц\w*", re.IGNORECASE)
_RECEIPT_NUMBER_VALUE_RE = re.compile(r"\b(\d{8,20})\b")
_SENDER_LABEL_RE = re.compile(r"\b(?:отправител\w*|жіберуш\w*)\b", re.IGNORECASE)
_RECIPIENT_LABEL_RE = re.compile(r"\b(?:кому|кімге)\b", re.IGNORECASE)
# "Surname I." — a Cyrillic/Kazakh name abbreviated to one initial, the shape
# Kaspi prints sender/recipient names in. No digits anywhere in the line.
_NAME_VALUE_RE = re.compile(r"^[^\d]{2,40}\s[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ]\.$")


def _find_value_after(lines: list[str], label_index: int, value_re: re.Pattern[str]) -> str | None:
    """First line after `label_index` shaped like `value_re`, however far
    below the label it sits — tolerates the labels-then-values block split
    described above instead of assuming label and value share a line."""
    for line in lines[label_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        match = value_re.search(stripped)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    return None


def _extract_receipt_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}

    amount = _AMOUNT_RE.search(text)
    if amount:
        fields["amount"] = f"{amount.group(1).strip()} ₸"

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "receipt_number" not in fields and _RECEIPT_NUMBER_LABEL_RE.search(line):
            value = _find_value_after(lines, i, _RECEIPT_NUMBER_VALUE_RE)
            if value:
                fields["receipt_number"] = value
        if "sender" not in fields and _SENDER_LABEL_RE.search(line):
            value = _find_value_after(lines, i, _NAME_VALUE_RE)
            if value:
                fields["sender"] = value
        if "recipient" not in fields and _RECIPIENT_LABEL_RE.search(line):
            value = _find_value_after(lines, i, _NAME_VALUE_RE)
            if value:
                fields["recipient"] = value

    return fields


def extract_fields(raw_text: str) -> ExtractedDocument:
    """Regex/heuristic extraction. Never calls any LLM (CLAUDE.md §5/§9).

    A handful of patterns are unambiguous evidence of one specific document
    type — an MRZ block only appears on a passport, a VIN/plate only on a
    vehicle document. When one of those fires, its verdict on
    `document_type` wins over the generic keyword scan (`detect_document_type`),
    so "a type-specific field was extracted" and "document_type" can no
    longer disagree the way they did for an OCR-garbled passport whose
    Cyrillic labels didn't survive but whose MRZ digits did.
    """
    mrz_fields, type_hint = _extract_mrz(raw_text)
    generic_fields, generic_hint = _extract_generic(raw_text)
    type_hint = type_hint or generic_hint

    fields = dict(mrz_fields)
    for key, value in generic_fields.items():
        fields.setdefault(key, value)

    document_type = type_hint or detect_document_type(raw_text)

    if document_type == "receipt":
        for key, value in _extract_receipt_fields(raw_text).items():
            fields.setdefault(key, value)

    return ExtractedDocument(document_type=document_type, fields=fields, raw_text=raw_text)


def needs_local_llm_fallback(extracted: ExtractedDocument) -> bool:
    """True only when regex genuinely found nothing to go on (CLAUDE.md §9)."""
    return extracted.document_type == "unknown" and not extracted.fields


# -- optional local-LLM fallback (never llm_router.py, never cloud) ----------

_LOCAL_LLM_PROMPT = """The text below is raw OCR output from a personal document (passport, vehicle registration, contract, receipt, etc). OCR errors are possible.

Return ONLY valid JSON, no prose, no markdown fences:
{{
  "document_type": "...",
  "fields": {{"field_name": "value", ...}},
  "raw_text": "..."
}}

Rules:
- Use field names a person would search for later (e.g. "iin", "document_number", "issue_date", "vin"), not generic labels like "field_1".
- If a field is unreadable or ambiguous due to OCR noise, set its value to null — do not guess.
- "raw_text" is the original OCR text, kept as a fallback for fields not explicitly modeled above.

OCR text:
{ocr_text}"""


async def extract_with_local_llm(
    raw_text: str, *, base_url: str, model: str, timeout: float = 60.0
) -> ExtractedDocument | None:
    """Best-effort fallback via a local Ollama-compatible server.

    Deliberately bypasses `llm_router.py` — that chain is a cloud fallback
    pool, and raw PII must never leave this machine (CLAUDE.md §5). Returns
    None on any failure (server not running, bad JSON, ...) so the caller can
    keep the regex-only result instead of crashing the upload.
    """
    prompt = _LOCAL_LLM_PROMPT.format(ocr_text=raw_text)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
            )
            response.raise_for_status()
            payload = response.json()
        data = json.loads(payload["response"])
    except Exception as exc:  # local server down, bad JSON, wrong shape, ...
        logger.warning("local LLM field extraction unavailable/failed: %s", exc)
        return None

    document_type = str(data.get("document_type") or "unknown")
    raw_fields = data.get("fields") or {}
    fields = {str(k): str(v) for k, v in raw_fields.items() if v is not None}
    return ExtractedDocument(document_type=document_type, fields=fields, raw_text=raw_text)


# -- query matching (natural-language search, CLAUDE.md §7 Phase 3) ----------


def score_query(
    query: str, *, document_type: str, field_names: list[str]
) -> tuple[int, list[str]]:
    """Heuristic keyword match — never an LLM call (CLAUDE.md §5/§8).

    Returns (score, matched_field_names); a positive score means the query
    plausibly refers to this document, higher meaning a better match. Scoring
    a document is never conditional on `document_type` having been
    recognized: the type-keyword term below can add 0 for an "unknown"
    document, but any of its extracted fields can still match on their own —
    a document with real field-level data must stay findable even when
    classification itself failed (CLAUDE.md §7 Phase 3).
    """
    lowered = query.lower()
    score = sum(
        3 for pattern in _DOCUMENT_TYPE_PATTERNS.get(document_type, ()) if pattern.search(lowered)
    )

    matched_fields: list[str] = []
    for field_name in field_names:
        if not field_name:
            continue
        if field_name in lowered or field_name.replace("_", " ") in lowered:
            score += 2
            matched_fields.append(field_name)
            continue
        patterns = _FIELD_CONCEPT_PATTERNS.get(field_name, ())
        if any(pattern.search(lowered) for pattern in patterns):
            score += 2
            matched_fields.append(field_name)

    return score, matched_fields
