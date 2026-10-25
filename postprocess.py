"""Russian passport OCR post-processing.

    python postprocess.py <predictions.jsonl> <answers.jsonl> [--reasons]

--reasons also writes why each rejected document was rejected.
"""
import json
import sys
from collections import Counter

from passport_ocr import load_model, process


def main(in_path, out_path, with_reasons=False):
    model = load_model()
    decisions = Counter()
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for doc in map(json.loads, fin):
            answer = process(doc, model)
            decisions[answer.decision] += 1
            fout.write(answer.to_json(with_reasons) + "\n")
    print(f"accept: {decisions['accept']}, reject: {decisions['reject']}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--reasons"]
    main(args[0], args[1], with_reasons="--reasons" in sys.argv)
