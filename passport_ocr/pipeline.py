"""accept / reject for one document.

A document is accepted only if every field is confirmed by an independent source
(MRZ, train dictionaries or confident OCR), the fields have the label format, and
the passport is valid on CHECK_DATE.
"""
import re

from pydantic import ValidationError

from .config import CHECK_DATE, FIELDS, MRZ_LINE2_FIELDS, PARAMS
from .mrz_zone import confirmed_by_alignment, parse_line2
from .names import resolve_names
from .schema import Answer, PassportFields
from .text_fields import resolve_org, resolve_text
from .viz import formatted_values, viz_chars


def resolve_line2_field(field, viz_value, viz_conf, mrz_value, mrz_ok, aligned, viz_only_thr):
    if viz_value is not None and viz_value == mrz_value:
        return viz_value, None
    if mrz_value is not None and mrz_ok:
        return mrz_value, None
    if field in aligned:
        return viz_value, None
    if viz_value is not None and viz_only_thr is not None and viz_conf >= viz_only_thr:
        return viz_value, None
    return None, f"{field}: не подтверждено MRZ"


def resolve(doc, model, params):
    viz, source, mrz = doc.get("viz") or {}, doc.get("source"), doc.get("mrz")
    fields, why = resolve_names(viz, mrz, source, model, params)

    viz_values = formatted_values(viz, params["fix_format"])
    line2 = parse_line2(mrz)
    aligned = confirmed_by_alignment(mrz, viz_values) if params["mrz2_align"] else set()
    for field in MRZ_LINE2_FIELDS:
        mrz_value, mrz_ok = line2.get(field, (None, False))
        viz_conf = min(viz_chars(viz, field)[1] or [0.0])
        value, err = resolve_line2_field(field, viz_values[field], viz_conf, mrz_value, mrz_ok,
                                         aligned, params["viz_only_min_conf"][source])
        if err:
            why.append(err)
        else:
            fields[field] = value

    for field, (value, err) in [
        ("Issue_organization_ru", resolve_org(fields.get("Issue_organisation_code"), viz, source, model, params)),
        ("Birth_place_ru", resolve_text("Birth_place_ru", viz, source, model, params)),
    ]:
        if err:
            why.append(err)
        else:
            fields[field] = value
    return fields, why


def add_years(d, n):
    try:
        return d.replace(year=d.year + n)
    except ValueError:  # 29 February
        return d.replace(year=d.year + n, month=3, day=1)


def is_valid_on(birth, issue, on=CHECK_DATE):
    """Issued at 14, replaced at 20 and 45. The 90-day grace period counts as invalid."""
    if issue < add_years(birth, 14) or issue > on:
        return False
    for age in (20, 45):
        deadline = add_years(birth, age)
        if issue < deadline:
            return on < deadline
    return True


def logic_errors(passport):
    why = []
    if not is_valid_on(passport.birth(), passport.issue()):
        why.append("паспорт недействителен на дату проверки")
    middle, sex = passport.Middle_name_ru, passport.Sex_ru
    if re.search(r"(ВИЧ|ИЧ|ОГЛЫ|УУЛУ)$", middle) and sex != "МУЖ.":
        why.append("пол не сходится с отчеством")
    if re.search(r"(ВНА|ЧНА|КЫЗЫ|ГЫЗЫ)$", middle) and sex != "ЖЕН.":
        why.append("пол не сходится с отчеством")
    return why


def process(doc, model, params=PARAMS):
    fields, why = resolve(doc, model, params)
    if why:
        return Answer(doc_id=doc["doc_id"], decision="reject", reasons=why)
    try:
        passport = PassportFields(**{f: fields[f] for f in FIELDS})
    except ValidationError as e:
        return Answer(doc_id=doc["doc_id"], decision="reject",
                      reasons=[f"формат: {err['loc'][0]}" for err in e.errors()])
    why = logic_errors(passport)
    if why:
        return Answer(doc_id=doc["doc_id"], decision="reject", reasons=why)
    return Answer(doc_id=doc["doc_id"], decision="accept", fields=passport)
