from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECK_DATE = date(2026, 10, 1)

FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
          "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
          "Issue_organization_ru"]
NAME_FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru"]
TEXT_FIELDS = ["Birth_place_ru", "Issue_organization_ru"]
MRZ_LINE2_FIELDS = ["Licence_number", "Birth_date", "Issue_date", "Issue_organisation_code", "Sex_ru"]

# Every threshold below was chosen on 5-fold CV over train (see cv.py and REPORT.md).
# Per-source values exist because confidence is not calibrated: mobile_v2 reports
# conf=1.0 for 97% of characters.
PARAMS = {
    # min char confidence for a text word that nothing else confirms
    "text_min_conf": {"web": 0.93, "mobile": 0.93, "mobile_v2": 0.999},
    # a word seen in train labels at least known_word_min_count times may be less confident
    "known_word_min_conf": {"web": 0.9, "mobile": 0.9, "mobile_v2": None},
    "known_word_min_count": 3,
    # a word found in the Russian spelling dictionary may be less confident still
    "dictionary_word_min_conf": 0.8,
    # replace one unsure character to get a word seen in train
    "word_fix": True,
    "word_fix_max_conf": 0.9,
    "word_fix_min_count": 3,
    # OCR drops whole characters ("С" for "С.", "ОССИИ"); a confident word is suspicious
    # if a word one character longer is near_ratio times more frequent in train
    "near_ratio": 10,
    "near_min": 5,
    # restore a dropped "." or "-" when the bare word never occurs in train labels
    "punct_restore": True,
    "punct_restore_min": 20,
    # "21.052007" -> "21.05.2007", "ЖЕН" -> "ЖЕН."
    "fix_format": True,
    # issuing authority: snap to a name seen in train for the same code
    "org_fix_max_edits": 3,
    "org_fix_max_conf": 0.95,
    # names: a viz/MRZ mismatch is settled when one hypothesis is name_ratio times more
    # likely and its confusion was seen at least name_min_count times in train
    "name_min_count": 2,
    "name_ratio": 5.0,
    "name_eps": 0.01,
    "name_fix_max_conf": 0.98,
    "mrz_only_name_min_conf": 0.99,
    # confirm line-2 fields by aligning a broken MRZ line with the one built from viz
    "mrz2_align": True,
    # no usable MRZ: accept a field on viz alone at this confidence (0.98 caused errors)
    "viz_only_min_conf": {"web": 0.99, "mobile": 0.99, "mobile_v2": None},
}
