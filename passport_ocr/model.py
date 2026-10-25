"""Statistics learned from train labels: dictionaries and OCR confusion tables."""
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from .config import FIELDS, ROOT, TEXT_FIELDS
from .mrz_zone import line1, lines
from .viz import norm, viz_chars


def diff_segments(truth, read):
    """(expected piece, read piece) for every place where two strings differ."""
    ops = SequenceMatcher(None, truth, read, autojunk=False).get_opcodes()
    return [(truth[i1:i2], read[j1:j2]) for tag, i1, i2, j1, j2 in ops if tag != "equal"]


def deletion_index(words):
    """Maps every word with one character removed to the words it came from."""
    index = defaultdict(set)
    for w in words:
        index[w].add(w)
        for i in range(len(w)):
            index[w[:i] + w[i + 1:]].add(w)
    return index


def build_model(docs, labels):
    """docs are train predictions, labels is {doc_id: fields} for them."""
    orgs = defaultdict(Counter)
    words = {f: Counter() for f in TEXT_FIELDS}
    viz_err = Counter()
    mrz_err = Counter()
    for doc in docs:
        truth = labels[doc["doc_id"]]
        orgs[truth["Issue_organisation_code"]][norm(truth["Issue_organization_ru"])] += 1
        for f in TEXT_FIELDS:
            words[f].update(norm(truth[f]).split())
        for f in FIELDS:
            text, _ = viz_chars(doc.get("viz"), f)
            if text is not None:
                viz_err.update(diff_segments(norm(truth[f]), text))
        if lines(doc.get("mrz")):
            names = [norm(truth[f]) for f in ("Last_name_ru", "First_name_ru", "Middle_name_ru")]
            mrz_err.update(diff_segments(line1(*names), lines(doc["mrz"])[0]["text"]))
    return {
        "orgs": orgs,                # code -> Counter(issuing authority name)
        "words": words,              # field -> Counter(word)
        "word_index": {f: deletion_index(c) for f, c in words.items()},
        "viz_err": viz_err,          # (expected, read) -> count, Cyrillic
        "mrz_err": mrz_err,          # (expected, read) -> count, MRZ alphabet
    }


def load_train():
    with open(ROOT / "data" / "train_predictions.jsonl", encoding="utf-8") as fh:
        docs = [json.loads(line) for line in fh]
    with open(ROOT / "data" / "train_labels.jsonl", encoding="utf-8") as fh:
        labels = {d["doc_id"]: d["fields"] for d in map(json.loads, fh)}
    return docs, labels


def load_model():
    return build_model(*load_train())
