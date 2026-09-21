import asyncio
from config import load_settings
from tools.ocr import extract_text
from tools import document_fields

KAZPOCHTA = r"8d1017ef-17f5-4b6a-97d0-9b9bade0f459\scratchpad\5298748651578533273.jpg"
RECEIPT = r"8d1017ef-17f5-4b6a-97d0-9b9bade0f459\scratchpad\img.png"

SAMPLE_PASSPORT_OCR = """
PASSPORT / ПАСПОРТ
IIN: 900101300123
Surname: Bekov
Date of birth: 01.01.1990
"""


async def run_image(label, path, settings):
    img = open(path, "rb").read()
    text = await extract_text(
        img,
        tesseract_cmd=settings.tesseract_cmd,
        lang=settings.ocr_lang,
        tessdata_dir=settings.ocr_tessdata_dir,
    )
    doc = document_fields.extract_fields(text)
    print(f"=== {label} ===")
    print("document_type:", doc.document_type)
    print("fields:", doc.fields)
    print()


async def main():
    settings = load_settings()
    await run_image("Kazpochta label", KAZPOCHTA, settings)
    await run_image("Kaspi receipt", RECEIPT, settings)

    print("=== Passport (sample, no real PII/no image) ===")
    doc = document_fields.extract_fields(SAMPLE_PASSPORT_OCR)
    print("document_type:", doc.document_type)
    print("fields:", doc.fields)


asyncio.run(main())
