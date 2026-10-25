"""Birth place and issuing authority: free text that MRZ does not duplicate."""
import re
from difflib import SequenceMatcher
from functools import lru_cache

from spellchecker import SpellChecker

from .viz import lines_missing, viz_chars


@lru_cache(maxsize=1)
def russian_dictionary():
    return SpellChecker(language="ru")


def in_russian_dictionary(word):
    if not re.fullmatch(r"[А-ЯЁ]+(-[А-ЯЁ]+)*", word):
        return False
    return all(part.lower() in russian_dictionary() for part in word.split("-"))


def neighbours(word, field, model):
    """Train words one edit away: (word, expected piece, read piece, position in our word)."""
    index = model["word_index"][field]
    variants = {word} | {word[:i] + word[i + 1:] for i in range(len(word))}
    for cand in set().union(*(index.get(v, ()) for v in variants)) - {word}:
        ops = [op for op in SequenceMatcher(None, cand, word, autojunk=False).get_opcodes() if op[0] != "equal"]
        if len(ops) == 1 and max(ops[0][2] - ops[0][1], ops[0][4] - ops[0][3]) == 1:
            _, i1, i2, j1, j2 = ops[0]
            yield cand, cand[i1:i2], word[j1:j2], j1


def restore_punct(word, field, model, params):
    """Restore a dropped "." or "-" ("С" -> "С.", "РН" -> "Р-Н") if the bare word never occurs in train."""
    counts = model["words"][field]
    if counts[word]:
        return None
    cands = {cand for cand, right, read, _ in neighbours(word, field, model)
             if right in (".", "-") and not read and counts[cand] >= params["punct_restore_min"]}
    return cands.pop() if len(cands) == 1 else None


def looks_misread(word, field, model, params):
    """A confident word that probably lost a character.

    A substituted character shows up in confidence, a lost one does not, so only
    train words one character longer are considered.
    """
    if re.search(r"\d", word):  # "№2" and "№12" are legitimately close
        return False
    counts = model["words"][field]
    return any(len(cand) == len(word) + 1 and counts[cand] >= params["near_min"]
               and counts[cand] >= params["near_ratio"] * (counts[word] + 1)
               for cand, *_ in neighbours(word, field, model))


def fix_word(word, confs, field, model, params):
    """The train word reached by replacing one unsure character, if it is unambiguous."""
    if len(word) < 3:  # short words have too many neighbours: "Г" -> "2"
        return None
    counts = model["words"][field]
    scored, via_indel = [], 0
    for cand, right, read, j in neighbours(word, field, model):
        if len(right) != 1 or len(read) != 1:
            # not fixable by a substitution, but makes it ambiguous ("ПМВД": "ОМВД" or "МВД")
            via_indel = max(via_indel, counts[cand])
            continue
        if confs[j] >= params["word_fix_max_conf"] or counts[cand] < params["word_fix_min_count"]:
            continue
        seen = model["viz_err"][(right, read)]
        if seen:
            scored.append((seen * counts[cand], counts[cand], cand))
    scored.sort(reverse=True)
    if not scored or via_indel * 5 > scored[0][1]:
        return None
    if len(scored) == 1 or scored[0][0] >= 5 * scored[1][0]:
        return scored[0][2]
    return None


def accept_word(word, confs, field, source, model, params):
    """(word to output, None) or (None, reason to reject the document)."""
    lowest = min(confs, default=1.0)
    if lowest >= params["text_min_conf"][source]:
        restored = restore_punct(word, field, model, params) if params["punct_restore"] else None
        if restored:
            return restored, None
        if looks_misread(word, field, model, params):
            return None, f"«{word}» похоже на потерю символа"
        return word, None
    known_thr = params["known_word_min_conf"][source]
    if known_thr is not None and lowest >= known_thr \
            and model["words"][field][word] >= params["known_word_min_count"]:
        return word, None
    fixed = fix_word(word, confs, field, model, params) if params["word_fix"] else None
    if fixed:
        return fixed, None
    dictionary_thr = params["dictionary_word_min_conf"]
    if dictionary_thr is not None and lowest >= dictionary_thr and in_russian_dictionary(word):
        return word, None
    return None, f"низкая уверенность в «{word}» ({lowest:.3f})"


def resolve_text(field, viz, source, model, params):
    text, confs = viz_chars(viz, field)
    if text is None:
        return None, f"{field}: не найдено"
    if lines_missing(viz, field):
        return None, f"{field}: потеряна строка"
    out, pos = [], 0
    for word in text.split(" "):
        word_confs = confs[pos:pos + len(word)]
        pos += len(word) + 1
        accepted, why = accept_word(word, word_confs, field, source, model, params)
        if why:
            return None, f"{field}: {why}"
        out.append(accepted)
    return " ".join(out), None


def resolve_org(code, viz, source, model, params):
    """Issuing authority: a name seen in train for the same code first, plain text otherwise."""
    field = "Issue_organization_ru"
    text, confs = viz_chars(viz, field)
    known = model["orgs"].get(code, {}) if code else {}
    if text is not None and text in known:
        return text, None
    if text is not None and known and not lines_missing(viz, field):
        fits = []
        for name in known:
            ops = [op for op in SequenceMatcher(None, name, text, autojunk=False).get_opcodes() if op[0] != "equal"]
            edits = sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in ops)
            all_unsure = all(min(confs[max(j1 - 1, 0):max(j2, j1 + 1)] or [1.0]) < params["org_fix_max_conf"]
                             for _, i1, i2, j1, j2 in ops)
            if edits <= params["org_fix_max_edits"] and all_unsure:
                fits.append(name)
        if len(fits) == 1:
            return fits[0], None
    return resolve_text(field, viz, source, model, params)
