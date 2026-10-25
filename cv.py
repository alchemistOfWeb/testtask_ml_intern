"""5-fold cross-validation on train.

    python cv.py           current PARAMS
    python cv.py grid      + ablations from GRID
    python cv.py sweep     + one step up and down for every threshold (SWEEP)

Dictionaries are built on 4 folds and decisions are scored on the 5th. An error counts
as label noise when only names are wrong and our answer matches both viz and MRZ:
such a document cannot be told apart from a correct one.
"""
import copy
import sys
from collections import Counter

from passport_ocr import FIELDS, NAME_FIELDS, PARAMS, build_model, load_train, process
from passport_ocr.mrz_zone import translit
from passport_ocr.viz import norm, viz_chars


def is_label_noise(doc, wrong, answer):
    if not set(wrong) <= set(NAME_FIELDS):
        return False
    line1 = ((doc.get("mrz") or {}).get("lines") or [{"text": ""}])[0]["text"]
    for f in wrong:
        viz = (viz_chars(doc.get("viz"), f)[0] or "").replace("Ё", "Е")
        if viz != answer[f] or translit(answer[f]) not in line1:
            return False
    return True


def fold_models(docs, labels, folds=5):
    """Fold models do not depend on PARAMS, so a sweep builds them once."""
    return [build_model([d for i, d in enumerate(docs) if i % folds != k], labels) for k in range(folds)]


def run_cv(docs, labels, params, folds=5, models=None):
    stats, by_source = Counter(), Counter()
    models = models or fold_models(docs, labels, folds)
    for i, doc in enumerate(docs):
        answer = process(doc, models[i % folds], params)
        source = doc["source"]
        by_source[(source, "total")] += 1
        if answer.decision != "accept":
            continue
        got = answer.fields.model_dump()
        truth = labels[doc["doc_id"]]
        wrong = [f for f in FIELDS if norm(got[f]) != norm(truth[f])]
        kind = "good" if not wrong else "noise" if is_label_noise(doc, wrong, got) else "bad"
        stats[kind] += 1
        by_source[(source, kind)] += 1
        if kind == "bad":
            stats.update(f"bad:{f}" for f in wrong)
    return stats, by_source


def accepted(stats):
    return stats["good"] + stats["noise"] + stats["bad"]


def fmt(stats, n):
    fields = ", ".join(f"{k[4:]}={v}" for k, v in sorted(stats.items()) if k.startswith("bad:"))
    return (f"принято {accepted(stats)} ({accepted(stats) / n:.1%}), верно {stats['good']}, "
            f"шум разметки {stats['noise']}, ошибки {stats['bad']}" + (f" [{fields}]" if fields else ""))


def tweak(**changes):
    """PARAMS with some values replaced; per-source dicts are updated, not replaced."""
    params = copy.deepcopy(PARAMS)
    for key, value in changes.items():
        if isinstance(value, dict):
            params[key].update(value)
        else:
            params[key] = value
    return params


GRID = {
    "mrz2_align off": tweak(mrz2_align=False),
    "punct_restore off": tweak(punct_restore=False),
    "fix_format off": tweak(fix_format=False),
    "word_fix off": tweak(word_fix=False),
    "dictionary off": tweak(dictionary_word_min_conf=None),
    "known_word off": tweak(known_word_min_conf={"web": None, "mobile": None}),
    "viz_only off": tweak(viz_only_min_conf={"web": None, "mobile": None}),
    "mrz_only off": tweak(mrz_only_name_min_conf=None),
}

SWEEP = {
    "text web 0.92": tweak(text_min_conf={"web": 0.92}),
    "text web 0.94": tweak(text_min_conf={"web": 0.94}),
    "text mobile 0.92": tweak(text_min_conf={"mobile": 0.92}),
    "text mobile 0.94": tweak(text_min_conf={"mobile": 0.94}),
    "text v2 0.998": tweak(text_min_conf={"mobile_v2": 0.998}),
    "text v2 0.9995": tweak(text_min_conf={"mobile_v2": 0.9995}),
    "known web 0.88": tweak(known_word_min_conf={"web": 0.88}),
    "known web 0.92": tweak(known_word_min_conf={"web": 0.92}),
    "known mobile 0.88": tweak(known_word_min_conf={"mobile": 0.88}),
    "known mobile 0.92": tweak(known_word_min_conf={"mobile": 0.92}),
    "known v2 0.995": tweak(known_word_min_conf={"mobile_v2": 0.995}),
    "known_count 2": tweak(known_word_min_count=2),
    "known_count 5": tweak(known_word_min_count=5),
    "dictionary 0.75": tweak(dictionary_word_min_conf=0.75),
    "dictionary 0.85": tweak(dictionary_word_min_conf=0.85),
    "word_fix_conf 0.85": tweak(word_fix_max_conf=0.85),
    "word_fix_conf 0.93": tweak(word_fix_max_conf=0.93),
    "word_fix_count 2": tweak(word_fix_min_count=2),
    "word_fix_count 5": tweak(word_fix_min_count=5),
    "near_ratio 7": tweak(near_ratio=7),
    "near_ratio 15": tweak(near_ratio=15),
    "punct_min 10": tweak(punct_restore_min=10),
    "punct_min 50": tweak(punct_restore_min=50),
    "org_edits 2": tweak(org_fix_max_edits=2),
    "org_edits 5": tweak(org_fix_max_edits=5),
    "org_conf 0.9": tweak(org_fix_max_conf=0.9),
    "org_conf 0.98": tweak(org_fix_max_conf=0.98),
    "name_min_count 1": tweak(name_min_count=1),
    "name_min_count 3": tweak(name_min_count=3),
    "name_ratio 4": tweak(name_ratio=4.0),
    "name_ratio 7": tweak(name_ratio=7.0),
    "name_eps 0.005": tweak(name_eps=0.005),
    "name_eps 0.02": tweak(name_eps=0.02),
    "name_fix 0.95": tweak(name_fix_max_conf=0.95),
    "name_fix 0.995": tweak(name_fix_max_conf=0.995),
    "mrz_only 0.98": tweak(mrz_only_name_min_conf=0.98),
    "mrz_only 0.995": tweak(mrz_only_name_min_conf=0.995),
    "viz_only web 0.985": tweak(viz_only_min_conf={"web": 0.985}),
    "viz_only web 0.995": tweak(viz_only_min_conf={"web": 0.995}),
    "viz_only mobile 0.985": tweak(viz_only_min_conf={"mobile": 0.985}),
    "viz_only mobile 0.995": tweak(viz_only_min_conf={"mobile": 0.995}),
    "viz_only v2 0.999": tweak(viz_only_min_conf={"mobile_v2": 0.999}),
}


def main(modes):
    docs, labels = load_train()
    models = fold_models(docs, labels)
    base, by_source = run_cv(docs, labels, PARAMS, models=models)
    for mode, variants in [("grid", GRID), ("sweep", SWEEP)]:
        if mode not in modes:
            continue
        for name, params in variants.items():
            stats, _ = run_cv(docs, labels, params, models=models)
            print(f"{name:24s} {accepted(stats) - accepted(base):+4d} принятых, "
                  f"{stats['bad'] - base['bad']:+3d} ошибок | " + fmt(stats, len(docs)), flush=True)
    print("PARAMS:", fmt(base, len(docs)))
    for source in ["web", "mobile", "mobile_v2"]:
        s = {k[1]: v for k, v in by_source.items() if k[0] == source}
        print(f"  {source:9s} всего {s.get('total', 0)}, верно {s.get('good', 0)}, "
              f"шум {s.get('noise', 0)}, ошибки {s.get('bad', 0)}")


if __name__ == "__main__":
    main(sys.argv[1:])
