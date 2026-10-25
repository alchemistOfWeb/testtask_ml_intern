"""Постобработка OCR паспорта РФ: python postprocess.py <predictions.jsonl> <answers.jsonl>

По каждому документу решает accept / reject. Документ принимается, только если каждое
из 10 полей подтверждено независимым источником (MRZ, словарь, уверенность OCR),
пройдены логические проверки и паспорт действителен на CHECK_DATE.

Всё, что выучено из данных (словари «Кем выдан» и слов, таблицы типичных путаниц OCR),
собирается при запуске из train (data/train_*.jsonl) функцией build_model. Для
кросс-валидации её вызывают на части train — см. cv.py.
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
          "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
          "Issue_organization_ru"]
NAME_FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru"]
TEXT_FIELDS = ["Birth_place_ru", "Issue_organization_ru"]
CHECK_DATE = date(2026, 10, 1)
ROOT = Path(__file__).resolve().parent

PARAMS = {
    # Минимальный char_conf для текста, который нечем подтвердить, кроме самого OCR.
    # У mobile_v2 97% символов имеют conf=1.0, поэтому порог у неё почти единица.
    "text_min_conf": {"web": 0.93, "mobile": 0.93, "mobile_v2": 0.999},
    # Слово из словаря train принимаем с уверенностью пониже (None — не используем).
    "known_word_min_conf": {"web": 0.9, "mobile": 0.9, "mobile_v2": None},
    "known_word_min_count": 3,
    # Исправление слова на словарное заменой одного символа с уверенностью ниже
    # word_fix_max_conf; кандидат должен встречаться в train не реже word_fix_min_count.
    # Вставки/удаления не исправляем: на CV они давали ошибки («Р-НА» -> «Р-Н»).
    "word_fix": True,
    "word_fix_max_conf": 0.9,
    "word_fix_min_count": 3,
    # OCR часто теряет символ целиком («С» вместо «С.», «ОССИИ»), и уверенность это не
    # ловит. Слово подозрительно, если на расстоянии 1 есть слово, которое в train
    # встречается в near_ratio раз чаще (и не реже near_min раз).
    "near_ratio": 10,
    "near_min": 5,
    # «Кем выдан»: исправление на название из словаря по коду подразделения.
    "org_fix_max_edits": 3,
    "org_fix_max_conf": 0.95,
    # ФИО: расхождение viz/MRZ решаем по таблицам путаниц и уверенности символов.
    # Побеждает сторона, чья ошибка в train встречалась хотя бы name_min_count раз и
    # чья гипотеза в name_ratio раз правдоподобнее. name_eps — доля ошибок даже при
    # conf=1.0 (у mobile_v2 conf почти всегда 1.0).
    "name_min_count": 2,
    "name_ratio": 5.0,
    "name_eps": 0.01,
    # Удалять из viz символы по MRZ можно, только если viz в них не уверена; вставлять
    # в viz символы, которых она не видела, нельзя («НОВИНКО» -> «ОВИНКО» на CV).
    "name_fix_max_conf": 0.98,
    # Поле ФИО не найдено в viz: берём из MRZ, если все его символы там уверенные
    # (None — не берём).
    "mrz_only_name_min_conf": 0.99,
    # Если MRZ под контрольной цифрой расходится с viz, брать значение из MRZ.
    "trust_mrz_on_mismatch": True,
    # MRZ нет или нужная её часть нечитаема: поля, которые она дублирует, принимаем
    # по одной viz с уверенностью не ниже этой (None — не принимаем).
    # На CV: 0.98 даёт ошибки, 0.985 и 0.99 — нет. У mobile_v2 conf не информативна.
    "viz_only_min_conf": {"web": 0.99, "mobile": 0.99, "mobile_v2": None},
    # В разметке train 43 кода, которых нет в справочнике (он от 2026-07), поэтому
    # отсутствие кода в справочнике само по себе не ошибка.
    "require_code_in_reference": False,
}

# ---------------------------------------------------------------- MRZ

CYR2LAT = dict(zip("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
                   "A B V G D E E J Z I Q K L M N O P R S T U F H C 3 4 W X Y 9 6 7 8".split()))
LAT2CYR = {v: k for k, v in CYR2LAT.items() if k != "Ё"}


def translit(s):
    """Кириллица -> алфавит MRZ; дефис и пробел становятся '<'."""
    return "".join(CYR2LAT.get(c, "<") for c in s)


def mrz_line1(last, first, middle):
    return ("PNRUS" + translit(last) + "<<" + translit(first) + "<" + translit(middle)).ljust(44, "<")[:44]


def check_digit(s):
    total = 0
    for i, c in enumerate(s):
        if c.isdigit():
            v = int(c)
        elif "A" <= c <= "Z":
            v = ord(c) - 55
        elif c == "<":
            v = 0
        else:
            return None
        total += v * (7, 3, 1)[i % 3]
    return str(total % 10)


def mrz_date(yymmdd):
    """ГГММДД -> ДД.ММ.ГГГГ; век выбираем так, чтобы дата не была в будущем."""
    if not re.fullmatch(r"\d{6}", yymmdd):
        return None
    yy = int(yymmdd[:2])
    year = 2000 + yy if 2000 + yy <= CHECK_DATE.year else 1900 + yy
    s = f"{yymmdd[4:6]}.{yymmdd[2:4]}.{year}"
    return s if parse_date(s) else None


def parse_mrz_line2(mrz):
    """Значения полей строки 2 и флаги, какие из них защищены контрольными цифрами."""
    lines = [ln["text"] for ln in (mrz or {}).get("lines") or []]
    if len(lines) < 2 or len(lines[1]) != 44:
        return {}
    l2 = lines[1]
    num, opt = l2[0:9], l2[28:41]
    return {
        "licence": f"{num[:2]} {num[2]}{opt[0]} {num[3:]}" if num.isdigit() and opt[0].isdigit() else None,
        "birth": mrz_date(l2[13:19]),
        "sex": {"M": "МУЖ.", "F": "ЖЕН."}.get(l2[20]),
        "issue": mrz_date(opt[1:7]),
        "code": f"{opt[7:10]}-{opt[10:13]}" if opt[7:13].isdigit() else None,
        "ok_num": check_digit(num) == l2[9],
        "ok_birth": check_digit(l2[13:19]) == l2[19],
        "ok_opt": check_digit(l2[28:42]) == l2[42],
        "ok_all": check_digit(l2[0:10] + l2[13:20] + l2[21:43]) == l2[43],
    }


# ---------------------------------------------------------------- модель из train

def diff_segments(truth, read):
    """Пары (правильный кусок, прочитанный кусок) там, где строки расходятся."""
    ops = SequenceMatcher(None, truth, read, autojunk=False).get_opcodes()
    return [(truth[i1:i2], read[j1:j2]) for tag, i1, i2, j1, j2 in ops if tag != "equal"]


def build_model(docs, labels):
    """docs — предсказания train, labels — {doc_id: fields} к ним."""
    sub = defaultdict(set)
    with open(ROOT / "reference" / "subdivisions.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sub[row["code"]].add(norm(row["name"]))

    orgs = defaultdict(Counter)             # код -> Counter(название)
    words = {f: Counter() for f in TEXT_FIELDS}
    viz_err = Counter()                     # (как надо, как прочитала viz), кириллица
    mrz_err = Counter()                     # (как надо, как прочитала MRZ), латиница
    for doc in docs:
        g = labels[doc["doc_id"]]
        orgs[g["Issue_organisation_code"]][norm(g["Issue_organization_ru"])] += 1
        for f in TEXT_FIELDS:
            words[f].update(norm(g[f]).split())
        for f in FIELDS:
            text, _ = viz_chars(doc.get("viz"), f)
            if text is not None:
                viz_err.update(diff_segments(norm(g[f]), text))
        lines = (doc.get("mrz") or {}).get("lines") or []
        if lines:
            want = mrz_line1(norm(g["Last_name_ru"]), norm(g["First_name_ru"]), norm(g["Middle_name_ru"]))
            mrz_err.update(diff_segments(want, lines[0]["text"]))
    # слова справочника DaData в словарь не берём: формулировки там другие, а частоты
    # («ОБЛ.» тысячи раз) ломают правило near_ratio
    return {"subdivisions": sub, "orgs": orgs, "words": words,
            "word_index": {f: deletion_index(c) for f, c in words.items()},
            "viz_err": viz_err, "mrz_err": mrz_err}


def load_model():
    with open(ROOT / "data" / "train_predictions.jsonl", encoding="utf-8") as fh:
        docs = [json.loads(line) for line in fh]
    with open(ROOT / "data" / "train_labels.jsonl", encoding="utf-8") as fh:
        labels = {d["doc_id"]: d["fields"] for d in map(json.loads, fh)}
    return build_model(docs, labels)


# ---------------------------------------------------------------- визуальная зона

def norm(s):
    return " ".join(str(s or "").split()).upper()


def viz_chars(viz, field):
    """(текст, уверенность по каждому символу) или (None, None), если поля нет."""
    v = (viz or {}).get(field)
    if not v or not v.get("lines"):
        return None, None
    chars, confs = [], []
    for ln in v["lines"]:
        if chars:
            chars.append(" ")
            confs.append(1.0)
        chars += ln["text"].upper()
        confs += ln["char_conf"][:len(ln["text"])]
    # схлопываем пробелы вместе с их уверенностью, как norm()
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
    """Строки не покрывают рамку поля: часть текста (обычно целая строка) потеряна."""
    v = (viz or {}).get(field)
    if not v or not v.get("lines"):
        return False
    boxes = [ln["box"] for ln in v["lines"]]
    line_h = sum(b[3] for b in boxes) / len(boxes)
    covered = max(b[1] + b[3] for b in boxes) - min(b[1] for b in boxes)
    return len(boxes) < round(v["box"][3] / line_h) or covered < 0.8 * v["box"][3]


def parse_date(s):
    m = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})", s or "")
    if not m:
        return None
    try:
        return date(int(m[3]), int(m[2]), int(m[1]))
    except ValueError:
        return None


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


# ---------------------------------------------------------------- ФИО: viz + MRZ

def viz_only_names(viz, source, params):
    """MRZ нет: ФИО только по уверенности viz. Пустое отчество тогда не отличить от
    ненайденного, поэтому отчество обязано быть."""
    thr = params["viz_only_min_conf"][source]
    if thr is None:
        return {}, ["ФИО: нет MRZ"]
    got = {}
    for f in NAME_FIELDS:
        text, confs = viz_chars(viz, f)
        if text is None or min(confs, default=0.0) < thr or "Ё" in text:
            return {}, [f"{f}: нет MRZ, viz не уверена"]
        got[f] = text
    return got, []


def doubled(text, idx):
    """Лишние символы viz повторяют соседний символ."""
    return all(0 < i < len(text) and text[i] == text[i - 1] or i + 1 < len(text) and text[i] == text[i + 1]
               for i in idx)


def mrz_only_names(line, texts, params):
    """Части ФИО из строки 1 MRZ для полей, которых нет в viz."""
    thr = params["mrz_only_name_min_conf"]
    text, conf = line["text"], line["char_conf"]
    if thr is None or len(text) != 44 or not text.startswith("PNRUS") or not text.endswith("<"):
        return {}
    body = text[5:].rstrip("<")
    if "<<" not in body:
        return {}
    last, rest = body.split("<<", 1)
    first, _, middle = rest.partition("<")
    starts = [5, 5 + len(last) + 2, 5 + len(last) + 2 + len(first) + 1]
    found = {}
    for f, part, start in zip(NAME_FIELDS, [last, first, middle], starts):
        if texts[f] is not None or not part or any(c not in LAT2CYR for c in part):
            continue
        if min(conf[start:start + len(part)]) >= thr:
            found[f] = "".join(LAT2CYR[c] for c in part)
    return found


def resolve_names(viz, mrz, source, model, params):
    """Сверяем ФИО из viz со строкой 1 MRZ посимвольно.

    Строим строку 1 так, как она должна выглядеть по viz, и выравниваем с прочитанной.
    Каждое расхождение — либо ошибка viz, либо ошибка MRZ; какая из двух, решаем по
    тому, как часто такая путаница встречалась у каждого источника в train.
    """
    lines = (mrz or {}).get("lines") or []
    if not lines:
        return viz_only_names(viz, source, params)
    observed = lines[0]["text"]

    texts = {f: viz_chars(viz, f)[0] for f in NAME_FIELDS}
    confs = {f: viz_chars(viz, f)[1] or [] for f in NAME_FIELDS}
    for f, text in mrz_only_names(lines[0], texts, params).items():
        texts[f], confs[f] = text, [1.0] * len(text)
    # ожидаемая строка и откуда взялся каждый её символ: (поле, индекс) или None
    expected, origin = "PNRUS", [None] * 5
    for k, f in enumerate(NAME_FIELDS):
        sep = ["", "<<", "<"][k]
        expected += sep
        origin += [None] * len(sep)
        t = texts[f] or ""
        expected += translit(t)
        origin += [(f, i) for i in range(len(t))]
    origin_full = origin
    expected, origin = expected.ljust(44, "<")[:44], (origin + [None] * 44)[:44]

    out = {f: list(texts[f] or "") for f in NAME_FIELDS}
    why = []
    ops = SequenceMatcher(None, expected, observed, autojunk=False).get_opcodes()
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal" or (i2 <= 5 and j2 <= 5):  # "PNRUS" к именам не относится
            continue
        e, o = expected[i1:i2], observed[j1:j2]
        own = [origin[i] for i in range(i1, i2) if origin[i]]
        after = origin[i1 - 1] if i1 > 0 else None    # символ поля перед расхождением
        before = origin[i2] if i2 < 44 else None      # символ поля после него
        if own:
            fields = {f for f, _ in own}
        elif after or before:
            fields = {(after or before)[0]}
        else:  # расхождение целиком в заполнителе: относим к предыдущему полю
            near = [origin[i] for i in range(i1 - 1, -1, -1) if origin[i]]
            fields = {near[0][0]} if near else {"Last_name_ru"}
        # гипотеза A: viz права, MRZ прочитала e как o
        votes_a = model["mrz_err"][(e, o)]
        # гипотеза B: MRZ права, viz прочитала правильный текст как свой кусок
        v = "".join(out[f][i] for f, i in own)
        right = "".join(LAT2CYR.get(c, "?") for c in o)
        votes_b = 0 if "?" in right or len(fields) > 1 else model["viz_err"][(right, v)]
        c_viz = min((confs[f][i] for f, i in own), default=1.0)
        c_mrz = min(lines[0]["char_conf"][j1:j2], default=1.0)
        eps = params["name_eps"]
        swap = len(e) == len(o) and len(own) == len(e)
        if swap:
            # замена символа: частота путаницы * (ошибся неуверенный, прав уверенный)
            like_a = (votes_a + 0.5) * (1 - c_mrz + eps) * c_viz
            like_b = (votes_b + 0.5) * (1 - c_viz + eps) * c_mrz
        elif own and not doubled(texts[next(iter(fields))], [i for _, i in own]):
            # лишние символы в viz: у пропавшего в MRZ символа уверенности нет
            like_a = (votes_a + 0.5) * c_viz
            like_b = (votes_b + 0.5) * (1 - c_viz + eps)
        else:
            # лишние символы в MRZ (часто мусорная K в заполнителе) или удвоенная буква
            # в viz («АЛИЯННОРА»), где уверенность viz ничего не говорит: только частоты
            like_a, like_b = votes_a + 0.5, votes_b + 0.5
        need, ratio = params["name_min_count"], params["name_ratio"]
        if votes_a >= need and like_a >= ratio * like_b:
            continue
        if swap and "?" in right and votes_b == 0 and c_viz >= params["text_min_conf"][source] \
                and "Ё" not in v:
            continue  # в MRZ символ, которого в именах не бывает (1, 0, 2, 5): MRZ ошиблась
        allowed = own and (swap or c_viz < params["name_fix_max_conf"])
        if votes_b >= need and like_b >= ratio * like_a and allowed:
            (f,) = fields
            # out[f] — список кусков по символам viz, правка не сдвигает индексы
            out[f][own[0][1]] = right
            for _, i in own[1:]:
                out[f][i] = ""
            continue
        why.append(f"{'/'.join(sorted(fields))}: viz/MRZ расходятся ({e}|{o})")

    # хвост длинного ФИО в MRZ не поместился: его подтверждает только уверенность viz
    cut = [o for o in origin_full[44:] if o]
    thr = params["text_min_conf"][source]
    for f, i in cut:
        if confs[f][i] < thr:
            why.append(f"{f}: обрезано в MRZ, низкая уверенность")
            break

    got = {}
    for f in NAME_FIELDS:
        if any(f in w.split(":")[0] for w in why):
            continue
        if texts[f] is None and f != "Middle_name_ru":
            why.append(f"{f}: не найдено")
            continue
        # в MRZ Е и Ё неразличимы, а в разметке Ё почти не встречается
        got[f] = "".join(out[f]).replace("Ё", "Е")
    return got, why


def resolve_mrz_field(field, viz_value, viz_conf, mrz_value, mrz_ok, thr, params):
    """Поле, продублированное в строке 2 MRZ под контрольной цифрой."""
    if viz_value is not None and viz_value == mrz_value:
        return viz_value, None
    if mrz_value is not None and mrz_ok and params["trust_mrz_on_mismatch"]:
        return mrz_value, None
    if mrz_value is not None and mrz_ok:
        return None, f"{field}: viz={viz_value} MRZ={mrz_value}"
    if viz_value is not None and thr is not None and viz_conf >= thr:
        return viz_value, None
    return None, f"{field}: не подтверждено MRZ"


# ---------------------------------------------------------------- свободный текст

def deletion_index(counter):
    """Индекс для поиска слов на расстоянии 1: вариант с одним удалённым символом -> слова."""
    index = defaultdict(set)
    for w in counter:
        index[w].add(w)
        for i in range(len(w)):
            index[w[:i] + w[i + 1:]].add(w)
    return index


def neighbours(word, field, model):
    """Словарные слова на расстоянии 1 и правка до них: (слово, как надо, как прочитано, позиция)."""
    index = model["word_index"][field]
    variants = {word} | {word[:i] + word[i + 1:] for i in range(len(word))}
    for cand in set().union(*(index.get(v, ()) for v in variants)) - {word}:
        ops = [op for op in SequenceMatcher(None, cand, word, autojunk=False).get_opcodes() if op[0] != "equal"]
        if len(ops) == 1 and max(ops[0][2] - ops[0][1], ops[0][4] - ops[0][3]) == 1:
            _, i1, i2, j1, j2 = ops[0]
            yield cand, cand[i1:i2], word[j1:j2], j1


def looks_misread(word, field, model, params):
    """Уверенно прочитанное слово, в котором, похоже, пропал символ.

    Замену символа ловит уверенность, а у пропавшего символа уверенности нет. Поэтому
    смотрим только на словарные слова, длиннее нашего на один символ.
    """
    if re.search(r"\d", word):  # номера («№2», «11») похожи друг на друга законно
        return False
    counts = model["words"][field]
    return any(len(cand) == len(word) + 1 and counts[cand] >= params["near_min"]
               and counts[cand] >= params["near_ratio"] * (counts[word] + 1)
               for cand, *_ in neighbours(word, field, model))


def fix_word(word, confs, field, model, params):
    """Словарное слово, отличающееся заменой одного неуверенного символа."""
    if len(word) < 3:  # «Г» -> «2»: у коротких слов слишком много соседей
        return None
    counts = model["words"][field]
    scored, others = [], 0
    for cand, right, read, j in neighbours(word, field, model):
        if len(right) != 1 or len(read) != 1:
            # сосед через вставку/удаление: исправлять так не будем, но он делает
            # замену неоднозначной («ПМВД»: «ОМВД» или «МВД»)
            others = max(others, counts[cand])
            continue
        if confs[j] >= params["word_fix_max_conf"] or counts[cand] < params["word_fix_min_count"]:
            continue
        seen = model["viz_err"][(right, read)]
        if seen:
            scored.append((seen * counts[cand], counts[cand], cand))
    scored.sort(reverse=True)
    if not scored or others * 5 > scored[0][1]:
        return None
    if len(scored) == 1 or scored[0][0] >= 5 * scored[1][0]:
        return scored[0][2]
    return None


def resolve_text(field, viz, source, model, params):
    """Место рождения / Кем выдан: уверенность OCR плюс исправление по словарю слов."""
    text, confs = viz_chars(viz, field)
    if text is None:
        return None, f"{field}: не найдено"
    if lines_missing(viz, field):
        return None, f"{field}: потеряна строка"
    thr = params["text_min_conf"][source]
    known_thr = params["known_word_min_conf"][source]
    out, pos = [], 0
    for word in text.split(" "):
        wc = confs[pos:pos + len(word)]
        pos += len(word) + 1
        if min(wc, default=1.0) >= thr:
            if looks_misread(word, field, model, params):
                return None, f"{field}: «{word}» похоже на потерю символа"
            out.append(word)
            continue
        if known_thr is not None and model["words"][field][word] >= params["known_word_min_count"] \
                and min(wc) >= known_thr:
            out.append(word)
            continue
        fixed = fix_word(word, wc, field, model, params) if params["word_fix"] else None
        if fixed is None:
            return None, f"{field}: низкая уверенность в «{word}» ({min(wc):.3f})"
        out.append(fixed)
    return " ".join(out), None


def resolve_org(code, viz, source, model, params):
    """Кем выдан: сначала словарь названий по коду, потом как обычный текст."""
    text, confs = viz_chars(viz, "Issue_organization_ru")
    known = model["orgs"].get(code, {}) if code else {}
    if text is not None and text in known:
        return text, None
    if text is not None and known and not lines_missing(viz, "Issue_organization_ru"):
        fits = []
        for name in known:
            ops = [op for op in SequenceMatcher(None, name, text, autojunk=False).get_opcodes() if op[0] != "equal"]
            edits = sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in ops)
            unsure = all(min(confs[max(j1 - 1, 0):max(j2, j1 + 1)] or [1.0]) < params["org_fix_max_conf"]
                         for _, i1, i2, j1, j2 in ops)
            if edits <= params["org_fix_max_edits"] and unsure:
                fits.append(name)
        if len(fits) == 1:
            return fits[0], None
    return resolve_text("Issue_organization_ru", viz, source, model, params)


# ---------------------------------------------------------------- всё вместе

def resolve(doc, model, params):
    viz, source = doc.get("viz") or {}, doc.get("source")
    fields, why = resolve_names(viz, doc.get("mrz"), source, model, params)

    l2 = parse_mrz_line2(doc.get("mrz"))
    raw = {f: viz_chars(viz, f)[0] for f in FIELDS}
    viz_vals = {
        "Licence_number": fmt_licence(raw["Licence_number"]),
        "Birth_date": raw["Birth_date"] if parse_date(raw["Birth_date"]) else None,
        "Issue_date": raw["Issue_date"] if parse_date(raw["Issue_date"]) else None,
        "Issue_organisation_code": fmt_code(raw["Issue_organisation_code"]),
        "Sex_ru": raw["Sex_ru"] if raw["Sex_ru"] in ("МУЖ.", "ЖЕН.") else None,
    }
    checks = {
        "Licence_number": ("licence", l2.get("ok_num") and l2.get("ok_opt")),
        "Birth_date": ("birth", l2.get("ok_birth")),
        "Issue_date": ("issue", l2.get("ok_opt")),
        "Issue_organisation_code": ("code", l2.get("ok_opt")),
        "Sex_ru": ("sex", l2.get("ok_all")),
    }
    for field, (key, ok) in checks.items():
        conf = min(viz_chars(viz, field)[1] or [0.0])
        val, err = resolve_mrz_field(field, viz_vals[field], conf, l2.get(key), ok,
                                     params["viz_only_min_conf"][source], params)
        if err:
            why.append(err)
        else:
            fields[field] = val

    for field, (val, err) in [
        ("Issue_organization_ru", resolve_org(fields.get("Issue_organisation_code"), viz, source, model, params)),
        ("Birth_place_ru", resolve_text("Birth_place_ru", viz, source, model, params)),
    ]:
        if err:
            why.append(err)
        else:
            fields[field] = val
    return fields, why


def add_years(d, n):
    try:
        return d.replace(year=d.year + n)
    except ValueError:  # 29 февраля
        return d.replace(year=d.year + n, month=3, day=1)


def is_valid_on(birth, issue, on=CHECK_DATE):
    """Паспорт выдаётся в 14, меняется в 20 и 45. Льготные 90 дней не считаем."""
    if issue < add_years(birth, 14) or issue > on:
        return False
    for age in (20, 45):
        deadline = add_years(birth, age)
        if issue < deadline:
            return on < deadline
    return True


def logic_errors(fields, model, params):
    why = []
    birth, issue = parse_date(fields["Birth_date"]), parse_date(fields["Issue_date"])
    if not is_valid_on(birth, issue):
        why.append("паспорт недействителен на дату проверки")
    if params["require_code_in_reference"] and fields["Issue_organisation_code"] not in model["subdivisions"]:
        why.append("кода нет в справочнике")
    mid = fields["Middle_name_ru"]
    if re.search(r"(ВИЧ|ИЧ|ОГЛЫ|УУЛУ)$", mid) and fields["Sex_ru"] != "МУЖ.":
        why.append("пол не сходится с отчеством")
    if re.search(r"(ВНА|ЧНА|КЫЗЫ|ГЫЗЫ)$", mid) and fields["Sex_ru"] != "ЖЕН.":
        why.append("пол не сходится с отчеством")
    return why


def process(doc, model, params=PARAMS):
    fields, why = resolve(doc, model, params)
    if not why:
        why = logic_errors(fields, model, params)
    if why:
        return {"doc_id": doc["doc_id"], "decision": "reject", "reasons": why}
    return {"doc_id": doc["doc_id"], "decision": "accept", "fields": {f: fields[f] for f in FIELDS}}


def main(in_path, out_path, with_reasons=False):
    model = load_model()
    stats = Counter()
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for doc in map(json.loads, fin):
            ans = process(doc, model)
            stats[ans["decision"]] += 1
            if not with_reasons:
                ans.pop("reasons", None)
            fout.write(json.dumps(ans, ensure_ascii=False) + "\n")
    print(f"accept: {stats['accept']}, reject: {stats['reject']}")


if __name__ == "__main__":
    # --reasons: писать причины отказа в ответ (для разбора ошибок)
    args = [a for a in sys.argv[1:] if a != "--reasons"]
    main(args[0], args[1], with_reasons="--reasons" in sys.argv)
