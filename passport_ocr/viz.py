"""Reading the visual zone: text with per-character confidence and format fixes."""
import re
from datetime import date


def norm(s):
    return " ".join(str(s or "").split()).upper()


def viz_chars(viz, field):
    """(text, confidence of every character), or (None, None) if the field was not detected.

    Lines are joined with a space and whitespace is collapsed like norm() does, keeping
    the confidence list aligned with the text.
    """
    v = (viz or {}).get(field)
    if not v or not v.get("lines"):
        return None, None
    chars, confs = [], []
    for line in v["lines"]:
        if chars:
            chars.append(" ")
            confs.append(1.0)
        chars += line["text"].upper()
        confs += line["char_conf"][:len(line["text"])]
    text, out = "", []
    for ch, c in zip(chars, confs):
        if ch == " " and (not text or text[-1] == " "):
            continue
        text += ch
        out.append(c)
    if text.endswith(" "):
        text, out = text[:-1], out[:-1]
    return text, out


def lines_missing(viz, field):
    """The read lines do not cover the field box, so a whole line was lost."""
    v = (viz or {}).get(field)
    if not v or not v.get("lines"):
        return False
    boxes = [line["box"] for line in v["lines"]]
    line_height = sum(b[3] for b in boxes) / len(boxes)
    covered = max(b[1] + b[3] for b in boxes) - min(b[1] for b in boxes)
    return len(boxes) < round(v["box"][3] / line_height) or covered < 0.8 * v["box"][3]


def parse_date(s):
    m = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})", s or "")
    if not m:
        return None
    try:
        return date(int(m[3]), int(m[2]), int(m[1]))
    except ValueError:
        return None


def fmt_date(s, fix):
    if parse_date(s):
        return s
    digits = re.sub(r"\D", "", s or "")
    if fix and s and re.fullmatch(r"[\d. ]+", s) and len(digits) == 8:
        fixed = f"{digits[0:2]}.{digits[2:4]}.{digits[4:8]}"
        return fixed if parse_date(fixed) else None
    return None


def fmt_sex(s, fix):
    if s in ("МУЖ.", "ЖЕН."):
        return s
    return s + "." if fix and s in ("МУЖ", "ЖЕН") else None


def fmt_licence(s):
    digits = re.sub(r"\D", "", s or "")
    if len(digits) != 10 or re.sub(r"[\d ]", "", s or ""):
        return None
    return f"{digits[:2]} {digits[2:4]} {digits[4:]}"


def fmt_code(s):
    digits = re.sub(r"\D", "", s or "")
    if len(digits) != 6 or re.sub(r"[\d\-]", "", s or ""):
        return None
    return f"{digits[:3]}-{digits[3:]}"


def formatted_values(viz, fix_format):
    """viz values of the MRZ-duplicated fields in label format, None if unreadable."""
    raw = {f: viz_chars(viz, f)[0] for f in
           ["Licence_number", "Birth_date", "Issue_date", "Issue_organisation_code", "Sex_ru"]}
    return {
        "Licence_number": fmt_licence(raw["Licence_number"]),
        "Birth_date": fmt_date(raw["Birth_date"], fix_format),
        "Issue_date": fmt_date(raw["Issue_date"], fix_format),
        "Issue_organisation_code": fmt_code(raw["Issue_organisation_code"]),
        "Sex_ru": fmt_sex(raw["Sex_ru"], fix_format),
    }
