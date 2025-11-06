#!/usr/bin/env python3
"""
evaluate_preds_json.py

Usage:
  python evaluate_preds_json.py --input preds.jsonl [--output fixed.jsonl]

- Reads JSONL with fields: id, statement, pred_raw, gold (gold can be int or str)
- Extracts the FIRST digit (0/1) found in pred_raw
- Compares against gold and prints accuracy summary
- If --output is provided, writes a new JSONL with 'pred' set to 0/1 (or null if not found)
"""

import argparse
import json
import re
from typing import Optional

def first_binary_digit(s: str) -> Optional[int]:
    if not isinstance(s, str):
        s = str(s)
    m = re.search(r"[01]", s)  # first occurrence anywhere
    if m:
        return int(m.group(0))
    return None

def to_int_or_none(x):
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(str(x).strip())
        except Exception:
            return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL file (one record per line)")
    ap.add_argument("--output", default=None, help="Optional output JSONL with 'pred' populated")
    args = ap.parse_args()

    n = 0
    n_gold = 0
    n_correct = 0
    n_unparsed = 0

    out_f = open(args.output, "w", encoding="utf-8") if args.output else None
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                rec = json.loads(line)

                pred_raw = rec.get("pred_raw", "")
                pred = first_binary_digit(pred_raw)
                rec["pred"] = pred  # write/update

                gold = to_int_or_none(rec.get("gold", None))
                if gold is not None:
                    n_gold += 1
                    if pred is None:
                        n_unparsed += 1
                    elif pred == gold:
                        n_correct += 1

                if out_f:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if out_f:
            out_f.close()

    print(f"Processed: {n} lines")
    if n_gold > 0:
        acc = n_correct / n_gold
        print(f"With gold: {n_gold}  |  Accuracy: {acc:.3f}  ({n_correct}/{n_gold})")
        if n_unparsed > 0:
            print(f"Unparsed pred_raw (counted incorrect): {n_unparsed}")
    else:
        print("No gold labels found; accuracy not computed.")
    if args.output:
        print(f"Wrote updated JSONL to: {args.output}")

if __name__ == "__main__":
    main()
