"""
Builds web/src/labs/tokens/data/flores.json: a handful of sentences from
FLORES+ (openlanguagedata/flores_plus, dev split) in ~30 languages, for the
Tokenizer lab's cross-language comparison.

Why FLORES+ rather than sentences we translate ourselves (docs/03-decisions.md
D7, D13): every sentence is a professional human translation of the SAME
English source, so when Nepali costs 8x the tokens of English the difference is
the tokenizer's, not a translator's.

Licence: FLORES+ is CC BY-SA 4.0. The generated file carries the attribution
and the same licence; the lab shows it next to the table.

The dataset is gated on the Hub: accept its terms once, log in with
`huggingface-cli login`, then run:

    uv run python scripts/build_flores_sample.py
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "openlanguagedata/flores_plus"
SENTENCE_IDS = [3, 4, 5, 8, 11]  # short, plain, few proper nouns

# (FLORES code, English name, native name, script)
LANGUAGES = [
    ("eng_Latn", "English", "English", "Latin"),
    ("deu_Latn", "German", "Deutsch", "Latin"),
    ("fra_Latn", "French", "Français", "Latin"),
    ("spa_Latn", "Spanish", "Español", "Latin"),
    ("ita_Latn", "Italian", "Italiano", "Latin"),
    ("por_Latn", "Portuguese", "Português", "Latin"),
    ("fin_Latn", "Finnish", "Suomi", "Latin"),
    ("tur_Latn", "Turkish", "Türkçe", "Latin"),
    ("vie_Latn", "Vietnamese", "Tiếng Việt", "Latin"),
    ("ind_Latn", "Indonesian", "Bahasa Indonesia", "Latin"),
    ("swh_Latn", "Swahili", "Kiswahili", "Latin"),
    ("yor_Latn", "Yoruba", "Yorùbá", "Latin"),
    ("rus_Cyrl", "Russian", "Русский", "Cyrillic"),
    ("ukr_Cyrl", "Ukrainian", "Українська", "Cyrillic"),
    ("ell_Grek", "Greek", "Ελληνικά", "Greek"),
    ("heb_Hebr", "Hebrew", "עברית", "Hebrew"),
    ("arb_Arab", "Arabic", "العربية", "Arabic"),
    ("pes_Arab", "Persian", "فارسی", "Arabic"),
    ("urd_Arab", "Urdu", "اردو", "Arabic"),
    ("hin_Deva", "Hindi", "हिन्दी", "Devanagari"),
    ("npi_Deva", "Nepali", "नेपाली", "Devanagari"),
    ("ben_Beng", "Bengali", "বাংলা", "Bengali"),
    ("tam_Taml", "Tamil", "தமிழ்", "Tamil"),
    ("tel_Telu", "Telugu", "తెలుగు", "Telugu"),
    ("sin_Sinh", "Sinhala", "සිංහල", "Sinhala"),
    ("tha_Thai", "Thai", "ไทย", "Thai"),
    ("mya_Mymr", "Burmese", "မြန်မာ", "Myanmar"),
    ("khm_Khmr", "Khmer", "ខ្មែរ", "Khmer"),
    ("amh_Ethi", "Amharic", "አማርኛ", "Ethiopic"),
    ("cmn_Hans", "Chinese (Simplified)", "中文", "Han"),
    ("jpn_Jpan", "Japanese", "日本語", "Han"),
    ("kor_Hang", "Korean", "한국어", "Hangul"),
]

OUT = Path(__file__).resolve().parents[1] / "web/src/labs/tokens/data/flores.json"


def main() -> None:
    langs = []
    for code, name, native, script in LANGUAGES:
        path = hf_hub_download(REPO, f"dev/{code}.jsonl", repo_type="dataset")
        rows = {r["id"]: r["text"] for r in map(json.loads, Path(path).read_text().splitlines())}
        langs.append(
            {"code": code, "name": name, "native": native, "script": script, "sentences": [rows[i] for i in SENTENCE_IDS]}
        )
    OUT.write_text(
        json.dumps(
            {
                "source": "FLORES+ (openlanguagedata/flores_plus), dev split, sentence ids " + ", ".join(map(str, SENTENCE_IDS)),
                "license": "CC BY-SA 4.0",
                "url": "https://huggingface.co/datasets/openlanguagedata/flores_plus",
                "languages": langs,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {OUT} ({len(langs)} languages x {len(SENTENCE_IDS)} sentences)")


if __name__ == "__main__":
    main()
