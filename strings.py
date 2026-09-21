"""Static, non-LLM user-facing strings (CLAUDE.md §8).

Everything here is text that reaches the user without going through
`llm_router.py`: command replies, static notices, BotCommand menu
descriptions, and messages raised as `HarnessError` / `LinkError`.

None of these are triggered by user-authored text (they answer commands,
unsupported input, access checks, or pipeline failures), so there is no
language to detect from. Per LANGUAGE_POLICY (see `harness/prompts.py`) they
default to Kazakh. Never Russian, no exceptions.
"""

from __future__ import annotations

# -- bot/handlers/common.py --------------------------------------------------

START_TEXT = (
    "Сәлем! Мен сенің репетиторыңмын.\n\n"
    "Түсінгің келетін тақырыпты жаз — мысалы, «рекурсия» немесе «Ом заңы» — "
    "мен оны мысалмен түсіндіріп, тексеру үшін сұрақ қоямын.\n\n"
    "Немесе мақала не бейне сілтемесін жібер — қысқаша мазмұндама жасаймын.\n\n"
    "/help — не істей алатынымды көр"
)

HELP_TEXT = (
    "Қазір не істей аламын (Phase 3):\n\n"
    "• Тақырыпты жаз — сабақ басталады.\n"
    "• /tutor <тақырып> — сабақты нақты бастау.\n"
    "• Мақала сілтемесін жібер — негізгі тұстары бар қысқаша мазмұндама аласың.\n"
    "• Бейне сілтемесін жібер (YouTube, TikTok, Instagram…) — субтитр бойынша "
    "мазмұндама, ал субтитр болмаса — дауысты тану арқылы (бұл баяуырақ).\n"
    "• /summarize <сілтеме> — дәл солай, нақты түрде.\n"
    "• Құжат фотосын жібер (төлқұжат, техпаспорт, шарт, чек) — оқып, шифрланған "
    "күйде сақтаймын.\n"
    "• /find <сұрау> — сақталған құжаттардан іздеу (мысалы, /find көлік нөмірі).\n"
    "• /reset — біздің жазысуымызды ұмытып, жаңадан бастау.\n"
    "• /cancel — ағымдағы сабақтан шығу.\n\n"
    "Қазақша жауап беремін; ағылшын тілінде жазсаң — ағылшынша жауап беремін."
)

RESET_DONE = "Тарих тазаланды ({count} хабарлама). Енді не туралы сөйлесеміз?"

LESSON_CANCELLED = "Сабақ тоқтатылды. Жалғастырғың келгенде жаңа тақырып жаз."

# -- bot/handlers/tutor.py ----------------------------------------------------

ASK_TOPIC = "Қай тақырыпты қарастырамыз? Оны бір хабарламамен жаз."

GENERIC_ERROR = "Бірдеңе дұрыс болмады. Қайта көріп көр."

UNSUPPORTED_MESSAGE = (
    "Мен мәтінді, сілтемені және құжат фотосын түсінемін. "
    "Дауыс хабары мен файл түрлері әлі қолдау таппайды."
)

# -- bot/handlers/links.py -----------------------------------------------------

VIDEO_PROCESSING = "⏳ Бейне қабылданды, өңдеп жатырмын…"

LINK_GENERIC_ERROR = "Сілтемені өңдеу мүмкін болмады. Қайта көріп көр."

SUMMARIZE_USAGE = "Сілтеме жібер: /summarize https://…"

MULTIPLE_URLS_NOTICE = "Хабарламадағы бірінші сілтемені аламын, қалғанын өткізіп жіберемін."

# -- bot/handlers/documents.py, harness/documents.py (DocumentError) ----------

DOCUMENT_PROCESSING = "⏳ Құжатты өңдеп жатырмын…"

DOCUMENT_GENERIC_ERROR = "Құжатты өңдеу мүмкін болмады. Қайта көріп көр."

DOCUMENT_OCR_FAILED = "Суреттен мәтінді оқу мүмкін болмады: {error}"

DOCUMENT_STORAGE_ERROR = "Құжат қоймасында қате шықты: {error}"

DOCUMENT_NO_TEXT_FOUND = (
    "Суреттен мәтін таныла алмады. Анығырақ әрі жарығы жақсы фото жібер."
)

DOCUMENT_SAVED = "✅ Сақталды: {type} ({count} өріс табылды)."

DOCUMENT_FIND_USAGE = "Нені іздеу керек? Мысалы: /find көлік тіркеу нөмірі"

DOCUMENT_NOT_FOUND = (
    "Сәйкес құжат табылмады. Басқаша сөзбен сұрап көр немесе құжат түрін ата "
    "(мысалы, «төлқұжат», «техпаспорт»)."
)

DOCUMENT_MATCH_HEADER = "📄 Табылды: {type}"

# Internal document_type slugs (tools/document_fields.py) shown to the user.
DOCUMENT_TYPE_LABELS = {
    "passport": "Төлқұжат",
    "vehicle_registration": "Көлік құжаты (техпаспорт)",
    "contract": "Шарт",
    "receipt": "Түбіртек",
    "unknown": "Белгісіз құжат",
}

# -- harness/orchestrator.py, harness/links.py (HarnessError / LinkError) -----

LINKS_NOT_CONFIGURED = "Сілтемелерді мазмұндау мүмкіндігі бапталмаған."

ALL_PROVIDERS_FAILED = (
    "Барлық LLM-провайдерлер қазір қолжетімсіз. Бір минуттан кейін қайта көріп көр."
)

ARTICLE_READ_FAILED = "Парақты оқу мүмкін болмады: {error}"

VIDEO_FETCH_FAILED = "Бейнені алу мүмкін болмады: {error}"

LIVESTREAM_NOT_SUPPORTED = "Бұл — тікелей эфир, ол жүріп жатқанда мазмұндау мүмкін емес."

SUBTITLES_FOUND = "Субтитр табылды ({lang}), оқып жатырмын…"

BUILDING_SUMMARY = "Мазмұндама құрастырып жатырмын…"

STT_NOT_CONFIGURED = (
    "Бейнеде субтитр жоқ, ал сөзді тану бапталмаған "
    "(GROQ_API_KEY немесе faster-whisper керек)."
)

VIDEO_TOO_LONG_FOR_STT = (
    "Бейнеде субтитр жоқ, әрі ол тануға тым ұзақ "
    "({minutes:.0f} мин, шегі — {limit:.0f} мин)."
)

DOWNLOADING_AUDIO = "Субтитр жоқ. Аудионы жүктеп жатырмын{length}…"
AUDIO_LENGTH_SUFFIX = " ({minutes:.0f} мин)"

TRANSCRIBING = "Сөзді танып жатырмын…"
TRANSCRIBING_PART = "Сөзді танып жатырмын… {done} бөлігі"

AUDIO_DOWNLOAD_FAILED = "Аудионы жүктеу мүмкін болмады: {error}"

TRANSCRIPTION_FAILED = "Сөзді тану мүмкін болмады: {error}"

# -- bot/middlewares.py --------------------------------------------------------

ACCESS_DENIED = (
    "Бұл бот жеке пайдалануға арналған. Егер ол сен үшін жұмыс істеуі керек "
    "болса, {user_id} ID-ін ALLOWED_USER_IDS-қа қос."
)

# -- bot/__init__.py (Telegram BotCommand menu descriptions) -------------------

CMD_TUTOR_DESC = "Тақырып бойынша сабақ бастау"
CMD_SUMMARIZE_DESC = "Сілтеме бойынша мақаланы немесе бейнені мазмұндау"
CMD_FIND_DESC = "Сақталған құжаттардан іздеу"
CMD_RESET_DESC = "Диалог тарихын тазалау"
CMD_CANCEL_DESC = "Ағымдағы сабақтан шығу"
CMD_HELP_DESC = "Бот не істей алады"
