"""MRZ of the Russian internal passport: two lines of 44 characters.

Line 1: PNRUS + LAST<<FIRST<MIDDLE, transliterated (Й->Q, Ч->3, Ш->4, Я->8, ...).
Line 2: series[0:3]+number, check, RUS, birth YYMMDD, check, sex, 7x'<',
        then series[3] + issue YYMMDD + code, '<', check, composite check.
The mrz package cannot validate this layout (no expiry date, digits in names),
so it is used only for ICAO 9303 check digits.
"""
import re
from difflib import SequenceMatcher

from mrz.base.functions import hash_string

from .config import CHECK_DATE
from .viz import parse_date

CYR2LAT = dict(zip("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
                   "A B V G D E E J Z I Q K L M N O P R S T U F H C 3 4 W X Y 9 6 7 8".split()))
LAT2CYR = {v: k for k, v in CYR2LAT.items() if k != "Ё"}

# positions of each field in line 2 and of the check digit that covers it
LINE2_SPANS = {
    "Licence_number": (list(range(0, 9)) + [28], 9),
    "Birth_date": (list(range(13, 19)), 19),
    "Sex_ru": ([20], 43),
    "Issue_date": (list(range(29, 35)), 42),
    "Issue_organisation_code": (list(range(35, 41)), 42),
}


def lines(mrz):
    return (mrz or {}).get("lines") or []


def translit(s):
    return "".join(CYR2LAT.get(c, "<") for c in s)


def line1(last, first, middle):
    return ("PNRUS" + translit(last) + "<<" + translit(first) + "<" + translit(middle)).ljust(44, "<")[:44]


def check_digit(s):
    try:
        return hash_string(s)
    except ValueError:
        return None


def to_date(yymmdd):
    """YYMMDD -> DD.MM.YYYY, picking the century that keeps the date in the past."""
    if not re.fullmatch(r"\d{6}", yymmdd):
        return None
    yy = int(yymmdd[:2])
    year = 2000 + yy if 2000 + yy <= CHECK_DATE.year else 1900 + yy
    s = f"{yymmdd[4:6]}.{yymmdd[2:4]}.{year}"
    return s if parse_date(s) else None


def parse_line2(mrz):
    """Line-2 values in label format, each paired with "is it covered by a valid check digit"."""
    texts = [line["text"] for line in lines(mrz)]
    if len(texts) < 2 or len(texts[1]) != 44:
        return {}
    l2 = texts[1]
    num, opt = l2[0:9], l2[28:41]
    ok_num = check_digit(num) == l2[9]
    ok_opt = check_digit(l2[28:42]) == l2[42]
    licence = f"{num[:2]} {num[2]}{opt[0]} {num[3:]}" if num.isdigit() and opt[0].isdigit() else None
    code = f"{opt[7:10]}-{opt[10:13]}" if opt[7:13].isdigit() else None
    return {
        "Licence_number": (licence, ok_num and ok_opt),
        "Birth_date": (to_date(l2[13:19]), check_digit(l2[13:19]) == l2[19]),
        "Issue_date": (to_date(opt[1:7]), ok_opt),
        "Issue_organisation_code": (code, ok_opt),
        "Sex_ru": ({"M": "МУЖ.", "F": "ЖЕН."}.get(l2[20]),
                   check_digit(l2[0:10] + l2[13:20] + l2[21:43]) == l2[43]),
    }


def expected_line2(values):
    """Line 2 as it should read given the viz values; unknown parts are '?'."""
    def yymmdd(s):
        return f"{s[8:10]}{s[3:5]}{s[0:2]}" if s else "??????"

    def check(s):
        return "?" if "?" in s else check_digit(s)

    licence = re.sub(r"\D", "", values.get("Licence_number") or "") or "?" * 10
    num, series_digit = licence[0:3] + licence[4:10], licence[3]
    birth = yymmdd(values.get("Birth_date"))
    sex = {"МУЖ.": "M", "ЖЕН.": "F"}.get(values.get("Sex_ru"), "?")
    code = re.sub(r"\D", "", values.get("Issue_organisation_code") or "") or "??????"
    opt = series_digit + yymmdd(values.get("Issue_date")) + code
    line = num + check(num) + "RUS" + birth + check(birth) + sex + "<" * 7 + opt + "<" + check(opt + "<")
    return line + check(line[0:10] + line[13:20] + line[21:43])


def confirmed_by_alignment(mrz, values):
    """viz fields that a damaged line 2 still repeats character for character.

    An extra or lost character, or an error in a neighbouring field, breaks the check
    digits, but the rest of the line is still an independent second reading.
    """
    texts = lines(mrz)
    if len(texts) < 2:
        return set()
    ops = SequenceMatcher(None, expected_line2(values), texts[1]["text"], autojunk=False).get_opcodes()
    same = {i for tag, i1, i2, _, _ in ops if tag == "equal" for i in range(i1, i2)}
    return {f for f, (span, _) in LINE2_SPANS.items()
            if values.get(f) is not None and all(i in same for i in span)}
