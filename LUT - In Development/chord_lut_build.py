#!/usr/bin/env python3
"""
chord_lut_build.py

Build a CHORD LUT from an existing STRING-AWARE note LUT.

Assumes the input LUT entries look like:
  entry["note"] = "E4_5"   (string-aware label)
  entry["string_idx"] = 5
  entry["takes"][i]["band_template"] = [ ... ]  (same length for all takes)

Chord rule:
- 6 strings (0..5)
- at most 1 note per string
- each string can be muted (no note)

Chord template construction:
- pick ONE take per chosen string-note (default: average takes for that class first)
- chord_template = sum(templates of chosen strings)
- L2 normalize

Output:
- meta includes "type": "chord_lut_from_string_note_lut"
- entries: each chord has:
    {
      "chord_id": "...",
      "notes_by_string": { "0": "E2_0", "1": None, ... },
      "classes": ["E2_0","B2_1",...],   # non-mutes only
      "num_sounding": 4,
      "band_template": [...],
    }

Important knobs to prevent explosion:
- --max_sounding N      limit sounding strings (e.g. 4 or 5)
- --min_sounding N      ignore super sparse (e.g. 2)
- --top_per_string N    only keep top N classes per string (by "strength" proxy)
- --limit_total M       hard cap on total chords generated
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from itertools import product
import numpy as np


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def mean_take_template(entry: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Make a single representative template for a class by averaging all take band_templates.
    """
    takes = entry.get("takes", [])
    vecs = []
    for t in takes:
        bt = t.get("band_template", None)
        if bt is None:
            continue
        vecs.append(np.array(bt, dtype=np.float64))
    if not vecs:
        return None
    m = np.mean(np.stack(vecs, axis=0), axis=0)
    return l2_normalize(m)


def template_strength_proxy(v: np.ndarray) -> float:
    """
    Proxy for "how useful" a class is to include if we need top_per_string pruning.
    With L2-normalized templates, all norms are ~1, so we use L1 as a crude proxy
    for 'broad energy distribution' (you can change this).
    """
    return float(np.sum(np.abs(v)))


def build_string_class_tables(
    note_lut: Dict[str, Any],
    require_band_template: bool = True
) -> Tuple[Dict[int, List[str]], Dict[str, np.ndarray]]:
    """
    Returns:
      classes_by_string: {string_idx: [label,...]}
      template_by_label: {label: template_vector}
    """
    entries = note_lut.get("entries", [])
    classes_by_string: Dict[int, List[str]] = {i: [] for i in range(6)}
    template_by_label: Dict[str, np.ndarray] = {}

    for e in entries:
        label = str(e.get("note", "")).strip()
        if not label:
            continue
        sidx = e.get("string_idx", None)
        if sidx is None:
            # try infer from label suffix _N
            if "_" in label and label.split("_")[-1].isdigit():
                sidx = int(label.split("_")[-1])
            else:
                continue
        sidx = int(sidx)
        if sidx < 0 or sidx > 5:
            continue

        templ = mean_take_template(e)
        if templ is None:
            if require_band_template:
                continue
            else:
                continue

        classes_by_string[sidx].append(label)
        template_by_label[label] = templ

    return classes_by_string, template_by_label


def chord_id_from_notes(notes_by_string: List[Optional[str]]) -> str:
    """
    Create a stable ID like: "S0=E2_0|S1=mut|S2=A3_2|..."
    """
    parts = []
    for i, lab in enumerate(notes_by_string):
        parts.append(f"S{i}={(lab if lab is not None else 'mut')}")
    return "|".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_lut", required=True, help="Input string-note LUT json")
    ap.add_argument("--out", required=True, help="Output chord LUT json")

    ap.add_argument("--min_sounding", type=int, default=1, help="Min sounding strings in a chord")
    ap.add_argument("--max_sounding", type=int, default=6, help="Max sounding strings in a chord")

    ap.add_argument("--top_per_string", type=int, default=0,
                    help="If >0, keep only top N classes per string (prunes explosion)")

    ap.add_argument("--limit_total", type=int, default=0,
                    help="Hard cap on number of chords generated (0 = no cap)")

    ap.add_argument("--mute_token", type=str, default="mut", help="Token used in IDs (cosmetic)")

    args = ap.parse_args()

    note_lut = load_json(args.in_lut)
    classes_by_string, templ_by_label = build_string_class_tables(note_lut)

    # Optional pruning: keep only top N per string by a proxy
    if args.top_per_string and args.top_per_string > 0:
        for s in range(6):
            labs = classes_by_string[s]
            scored = []
            for lab in labs:
                v = templ_by_label[lab]
                scored.append((lab, template_strength_proxy(v)))
            scored.sort(key=lambda x: x[1], reverse=True)
            classes_by_string[s] = [lab for lab, _ in scored[:int(args.top_per_string)]]

    # Build per-string choices including mute
    # Each choice is either None (mute) or a label
    per_string_choices: List[List[Optional[str]]] = []
    for s in range(6):
        choices = [None] + classes_by_string[s]
        per_string_choices.append(choices)

    # Estimate count (rough)
    total_possible = 1
    for ch in per_string_choices:
        total_possible *= len(ch)
    print(f"[info] raw combinations (including mutes): {total_possible}")

    out_entries: List[Dict[str, Any]] = []
    made = 0

    # Enumerate
    for combo in product(*per_string_choices):
        # combo is length 6: Optional[str]
        num_sounding = sum(1 for x in combo if x is not None)

        if num_sounding < int(args.min_sounding) or num_sounding > int(args.max_sounding):
            continue

        # sum templates
        acc = None
        classes = []
        notes_by_string = []
        for sidx, lab in enumerate(combo):
            notes_by_string.append(lab)
            if lab is None:
                continue
            v = templ_by_label.get(lab, None)
            if v is None:
                continue
            classes.append(lab)
            acc = v.copy() if acc is None else (acc + v)

        if acc is None:
            continue

        chord_template = l2_normalize(acc)

        chord_id = chord_id_from_notes(list(notes_by_string))
        entry = {
            "chord_id": chord_id,
            "notes_by_string": {str(i): notes_by_string[i] for i in range(6)},
            "classes": classes,
            "num_sounding": int(num_sounding),
            "band_template": [float(x) for x in chord_template.tolist()],
        }
        out_entries.append(entry)
        made += 1

        if args.limit_total and args.limit_total > 0 and made >= int(args.limit_total):
            print(f"[warn] hit --limit_total {args.limit_total}, stopping early")
            break

        if made % 10000 == 0:
            print(f"[info] generated {made} chords...")

    chord_lut = {
        "meta": {
            "type": "chord_lut_from_string_note_lut",
            "source_lut": str(Path(args.in_lut).name),
            "min_sounding": int(args.min_sounding),
            "max_sounding": int(args.max_sounding),
            "top_per_string": int(args.top_per_string),
            "limit_total": int(args.limit_total),
            "band_template": note_lut.get("meta", {}).get("band_template", {}),
        },
        "entries": out_entries,
    }

    save_json(args.out, chord_lut)
    print(f"[done] wrote chord LUT: {args.out}  entries={len(out_entries)}")


if __name__ == "__main__":
    main()