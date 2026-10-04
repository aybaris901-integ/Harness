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
    "Қазір не істей аламын (Phase 4):\n\n"
    "• Тақырыпты жаз — сабақ басталады.\n"
    "• /tutor <тақырып> — сабақты нақты бастау.\n"
    "• Мақала сілтемесін жібер — негізгі тұстары бар қысқаша мазмұндама аласың.\n"
    "• Бейне сілтемесін жібер (YouTube, TikTok, Instagram…) — субтитр бойынша "
    "мазмұндама, ал субтитр болмаса — дауысты тану арқылы (бұл баяуырақ).\n"
    "• /summarize <сілтеме> — дәл солай, нақты түрде.\n"
    "• /download <сілтеме> [mp3] — кез келген сілтеме бойынша медианы жүктеп, "
    "файл түрінде әрі сипаттамамен қайтарамын.\n"
    "• Құжат фотосын жібер (төлқұжат, техпаспорт, шарт, чек) — оқып, шифрланған "
    "күйде сақтаймын.\n"
    "• /find <сұрау> — сақталған құжаттардан іздеу (мысалы, /find көлік нөмірі).\n"
    "• Скриншотты «/note» қолтаңбасымен жібер (тақтаға жазылған дәрісхана, код "
    "фотосы т.б.) — таза Markdown жазбаға айналдырып, сақтаймын.\n"
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

# -- bot/handlers/media.py, harness/media.py (MediaError) ----------------------

MEDIA_PROCESSING = "⏳ Сілтеме қабылданды, жүктеп жатырмын…"

MEDIA_DOWNLOADING = "Жүктеп жатырмын…"

MEDIA_DESCRIBING = "Сипаттама құрастырып жатырмын…"

MEDIA_GENERIC_ERROR = "Медианы өңдеу мүмкін болмады. Қайта көріп көр."

DOWNLOAD_USAGE = (
    "Сілтеме жібер: /download https://…\n"
    "Дыбысын mp3-ге түрлендіру керек болса, соңына «mp3» деп қос: "
    "/download https://… mp3"
)

MEDIA_PROBE_FAILED = "Сілтемені тексеру мүмкін болмады: {error}"

MEDIA_DOWNLOAD_FAILED = "Жүктеу мүмкін болмады: {error}"

MEDIA_TOO_LARGE = "Файл тым үлкен ({size:.0f} МБ, шегі — {limit:.0f} МБ)."

NOTE_PROCESSING = "⏳ Скриншотты оқып жатырмын…"

NOTE_SAVED = "📝 {title}"

NOTE_USAGE = (
    "Скриншотты жазбаға айналдыру үшін фотоны «/note» деген қолтаңбамен (caption) жібер."
)

NOTE_OCR_FAILED = "Суреттен мәтінді оқу мүмкін болмады: {error}"

NOTE_NO_TEXT_FOUND = "Суреттен мәтін таныла алмады. Анығырақ әрі жарығы жақсы фото жібер."

NOTE_FORMAT_FAILED = "Жазбаны құрастыру мүмкін болмады: {error}"

# PII safety net: a /note-captioned photo that looks like a personal document
# (passport, vehicle registration) is redirected into the encrypted document
# archive instead of the unencrypted notes table — see harness/media.py.
NOTE_PII_REDIRECTED = (
    "⚠️ Бұл скриншот жеке құжатқа ұқсайды, сондықтан оны шифрланбаған жазба ретінде "
    "сақтамадым — орнына шифрланған құжат қоймасына сақтадым."
)

NOTE_PII_NO_ARCHIVE = (
    "⚠️ Бұл скриншот жеке құжатқа ұқсайды, сондықтан оны шифрланбаған жазба ретінде "
    "сақтай алмаймын. Оны «/note» қолтаңбасынсыз, қарапайым фото ретінде қайта жібер — "
    "құжат ретінде шифрланған күйде сақтаймын."
)

NOTE_PII_ARCHIVE_FAILED = (
    "Бұл скриншот жеке құжатқа ұқсайды, бірақ оны құжат қоймасына сақтау кезінде қате "
    "шықты: {error}"
)

# -- bot/middlewares.py --------------------------------------------------------

ACCESS_DENIED = (
    "Бұл бот жеке пайдалануға арналған. Егер ол сен үшін жұмыс істеуі керек "
    "болса, {user_id} ID-ін ALLOWED_USER_IDS-қа қос."
)

# -- bot/__init__.py (Telegram BotCommand menu descriptions) -------------------

CMD_TUTOR_DESC = "Тақырып бойынша сабақ бастау"
CMD_SUMMARIZE_DESC = "Сілтеме бойынша мақаланы немесе бейнені мазмұндау"
CMD_DOWNLOAD_DESC = "Кез келген сілтеме бойынша медианы жүктеп алу"
CMD_FIND_DESC = "Сақталған құжаттардан іздеу"
CMD_RESET_DESC = "Диалог тарихын тазалау"
CMD_CANCEL_DESC = "Ағымдағы сабақтан шығу"
CMD_HELP_DESC = "Бот не істей алады"
