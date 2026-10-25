"""Last, first and middle name: viz checked character by character against MRZ line 1."""
from difflib import SequenceMatcher

from .config import NAME_FIELDS
from .mrz_zone import LAT2CYR, lines, translit
from .viz import viz_chars


def viz_only_names(viz, source, params):
    """No MRZ: names on viz confidence alone. A missing middle name cannot be told apart
    from an undetected one here, so it is required."""
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


def mrz_only_names(line, texts, params):
    """Name parts taken from MRZ line 1 for fields viz did not detect."""
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


def doubled(text, idx):
    """The extra viz characters repeat a neighbour ("АЛИЯННОРА")."""
    return all(0 < i < len(text) and text[i] == text[i - 1] or i + 1 < len(text) and text[i] == text[i + 1]
               for i in idx)


def expected_line1(texts):
    """Line 1 built from viz, and for every position the (field, index) it came from."""
    expected, origin = "PNRUS", [None] * 5
    for sep, f in zip(["", "<<", "<"], NAME_FIELDS):
        expected += sep
        origin += [None] * len(sep)
        t = texts[f] or ""
        expected += translit(t)
        origin += [(f, i) for i in range(len(t))]
    return expected, origin


def mismatch_fields(i1, i2, origin):
    """viz characters inside the mismatch, and the name fields it belongs to."""
    own = [origin[i] for i in range(i1, i2) if origin[i]]
    if own:
        return own, {f for f, _ in own}
    neighbour = (origin[i1 - 1] if i1 > 0 else None) or (origin[i2] if i2 < 44 else None)
    if neighbour:
        return own, {neighbour[0]}
    earlier = [origin[i] for i in range(i1 - 1, -1, -1) if origin[i]]
    return own, {earlier[0][0]} if earlier else {"Last_name_ru"}


def resolve_names(viz, mrz, source, model, params):
    """Every viz/MRZ mismatch is either a viz error or an MRZ error.

    Likelihood of each = how often that confusion happened to that source in train,
    times "the unsure side erred and the confident side is right". The winner needs
    a name_ratio margin; otherwise the field is rejected.
    """
    if not lines(mrz):
        return viz_only_names(viz, source, params)
    mrz_line = lines(mrz)[0]
    observed = mrz_line["text"]

    texts = {f: viz_chars(viz, f)[0] for f in NAME_FIELDS}
    confs = {f: viz_chars(viz, f)[1] or [] for f in NAME_FIELDS}
    for f, text in mrz_only_names(mrz_line, texts, params).items():
        texts[f], confs[f] = text, [1.0] * len(text)
    expected, origin_full = expected_line1(texts)
    expected, origin = expected.ljust(44, "<")[:44], (origin_full + [None] * 44)[:44]

    out = {f: list(texts[f] or "") for f in NAME_FIELDS}
    why = []
    eps, need, ratio = params["name_eps"], params["name_min_count"], params["name_ratio"]
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, expected, observed, autojunk=False).get_opcodes():
        if tag == "equal" or (i2 <= 5 and j2 <= 5):
            continue
        e, o = expected[i1:i2], observed[j1:j2]
        own, fields = mismatch_fields(i1, i2, origin)
        v = "".join(out[f][i] for f, i in own)
        right = "".join(LAT2CYR.get(c, "?") for c in o)
        votes_mrz_wrong = model["mrz_err"][(e, o)]
        votes_viz_wrong = 0 if "?" in right or len(fields) > 1 else model["viz_err"][(right, v)]
        c_viz = min((confs[f][i] for f, i in own), default=1.0)
        c_mrz = min(mrz_line["char_conf"][j1:j2], default=1.0)
        swap = len(e) == len(o) and len(own) == len(e)
        if swap:
            like_mrz_wrong = (votes_mrz_wrong + 0.5) * (1 - c_mrz + eps) * c_viz
            like_viz_wrong = (votes_viz_wrong + 0.5) * (1 - c_viz + eps) * c_mrz
        elif own and not doubled(texts[next(iter(fields))], [i for _, i in own]):
            # extra characters in viz; the character MRZ lost has no confidence
            like_mrz_wrong = (votes_mrz_wrong + 0.5) * c_viz
            like_viz_wrong = (votes_viz_wrong + 0.5) * (1 - c_viz + eps)
        else:
            # extra MRZ characters (often a stray K in the filler) or a doubled viz letter:
            # confidence says nothing here, only the counts do
            like_mrz_wrong, like_viz_wrong = votes_mrz_wrong + 0.5, votes_viz_wrong + 0.5

        if votes_mrz_wrong >= need and like_mrz_wrong >= ratio * like_viz_wrong:
            continue
        impossible_in_names = "?" in right  # 0, 1, 2, 5 never occur in transliterated names
        if swap and impossible_in_names and votes_viz_wrong == 0 and "Ё" not in v \
                and c_viz >= params["text_min_conf"][source]:
            continue
        may_edit_viz = own and (swap or c_viz < params["name_fix_max_conf"])
        if votes_viz_wrong >= need and like_viz_wrong >= ratio * like_mrz_wrong and may_edit_viz:
            (f,) = fields
            out[f][own[0][1]] = right
            for _, i in own[1:]:
                out[f][i] = ""
            continue
        why.append(f"{'/'.join(sorted(fields))}: viz/MRZ расходятся ({e}|{o})")

    # the tail of a long name did not fit into MRZ; only viz confidence vouches for it
    if any(confs[f][i] < params["text_min_conf"][source] for f, i in (o for o in origin_full[44:] if o)):
        why.append("ФИО: обрезано в MRZ, низкая уверенность")

    got = {}
    for f in NAME_FIELDS:
        if any(f in w.split(":")[0] for w in why):
            continue
        if texts[f] is None and f != "Middle_name_ru":
            why.append(f"{f}: не найдено")
            continue
        # MRZ cannot tell Е from Ё, and labels almost never use Ё
        got[f] = "".join(out[f]).replace("Ё", "Е")
    return got, why
