"""Кросс-валидация postprocess.py на train: python cv.py [grid]

Модель (словари, таблицы путаниц) строится на k-1 фолдах, решения считаются на
оставшемся. Ошибка считается «шумом разметки», если неверны только ФИО, а наш ответ
совпадает и с viz, и с MRZ: такой документ по данным модели не отличить от верного.
"""
import copy
import json
import sys
from collections import Counter

import postprocess as pp


def load():
    with open(pp.ROOT / "data" / "train_predictions.jsonl", encoding="utf-8") as fh:
        docs = [json.loads(line) for line in fh]
    with open(pp.ROOT / "data" / "train_labels.jsonl", encoding="utf-8") as fh:
        labels = {d["doc_id"]: d["fields"] for d in map(json.loads, fh)}
    return docs, labels


def is_label_noise(doc, wrong, answer):
    if not set(wrong) <= set(pp.NAME_FIELDS):
        return False
    line1 = ((doc.get("mrz") or {}).get("lines") or [{"text": ""}])[0]["text"]
    for f in wrong:
        viz = (pp.viz_chars(doc.get("viz"), f)[0] or "").replace("Ё", "Е")
        if viz != answer[f] or pp.translit(answer[f]) not in line1:
            return False
    return True


def run_cv(docs, labels, params, folds=5):
    stats = Counter()
    by_source = Counter()
    for k in range(folds):
        train = [d for i, d in enumerate(docs) if i % folds != k]
        model = pp.build_model(train, labels)
        for i, doc in enumerate(docs):
            if i % folds != k:
                continue
            ans = pp.process(doc, model, params)
            src = doc["source"]
            by_source[(src, "total")] += 1
            if ans["decision"] != "accept":
                continue
            g = labels[doc["doc_id"]]
            wrong = [f for f in pp.FIELDS if pp.norm(ans["fields"][f]) != pp.norm(g[f])]
            if not wrong:
                stats["good"] += 1
                by_source[(src, "good")] += 1
            elif is_label_noise(doc, wrong, ans["fields"]):
                stats["noise"] += 1
                by_source[(src, "noise")] += 1
            else:
                stats["bad"] += 1
                by_source[(src, "bad")] += 1
                stats.update(f"bad:{f}" for f in wrong)
    return stats, by_source


def fmt(stats, n):
    acc = stats["good"] + stats["noise"] + stats["bad"]
    fields = ", ".join(f"{k[4:]}={v}" for k, v in sorted(stats.items()) if k.startswith("bad:"))
    return (f"принято {acc} ({acc / n:.1%}), верно {stats['good']}, "
            f"шум разметки {stats['noise']}, ошибки {stats['bad']}" + (f" [{fields}]" if fields else ""))


def variant(**changes):
    params = copy.deepcopy(pp.PARAMS)
    for key, value in changes.items():
        params[key] = value
    return params


def by_source_conf(value_web, value_mobile, value_v2):
    return {"web": value_web, "mobile": value_mobile, "mobile_v2": value_v2}


GRID = {
    # каждый вариант меняет один параметр относительно PARAMS
    "mrz2_align off": variant(mrz2_align=False),
    "punct_restore off": variant(punct_restore=False),
    "punct_restore_min 5": variant(punct_restore_min=5),
    "fix_format off": variant(fix_format=False),
    "text 0.95": variant(text_min_conf=by_source_conf(0.95, 0.95, 0.999)),
    "text 0.90": variant(text_min_conf=by_source_conf(0.9, 0.9, 0.999)),
    "text v2 0.995": variant(text_min_conf=by_source_conf(0.93, 0.93, 0.995)),
    "known_word off": variant(known_word_min_conf=by_source_conf(None, None, None)),
    "known_word 0.85": variant(known_word_min_conf=by_source_conf(0.85, 0.85, None)),
    "word_fix off": variant(word_fix=False),
    "near_ratio 5": variant(near_ratio=5),
    "name_ratio 3": variant(name_ratio=3.0),
    "name_ratio 10": variant(name_ratio=10.0),
    "mrz_only off": variant(mrz_only_name_min_conf=None),
    "viz_only off": variant(viz_only_min_conf=by_source_conf(None, None, None)),
    "viz_only 0.98": variant(viz_only_min_conf=by_source_conf(0.98, 0.98, None)),
    "viz_only 0.995": variant(viz_only_min_conf=by_source_conf(0.995, 0.995, None)),
}


if __name__ == "__main__":
    # python cv.py — текущие PARAMS; python cv.py grid — ещё и варианты из GRID
    folds = 5
    docs, labels = load()
    if "grid" in sys.argv:
        for name, params in GRID.items():
            print(f"{name:28s}", fmt(run_cv(docs, labels, params, folds)[0], len(docs)), flush=True)
    stats, by_source = run_cv(docs, labels, pp.PARAMS, folds)
    print("PARAMS:", fmt(stats, len(docs)))
    for src in ["web", "mobile", "mobile_v2"]:
        s = {k[1]: v for k, v in by_source.items() if k[0] == src}
        print(f"  {src:9s} всего {s.get('total', 0)}, верно {s.get('good', 0)}, "
              f"шум {s.get('noise', 0)}, ошибки {s.get('bad', 0)}")
