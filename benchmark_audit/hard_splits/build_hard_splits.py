#!/usr/bin/env python3
# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
"""
Build AVQA-Hard and Music-AVQA-Hard from the single-frame probe outputs.

Rules (as used for the released splits ``gwkrsrch/avqa_hard`` and ``gwkrsrch/music_avqa_hard``):

* AVQA-Hard: keep the items of ``gwkrsrch/avqa_2025`` (AVQA val items whose YouTube clips could
  still be retrieved) that GPT-4o answered incorrectly from the central frame
  (``correct == false`` in ``probe_avqa_single_frame.py`` output).
* Music-AVQA-Hard: keep the items of ``gwkrsrch/music_avqa`` (test split) that (i) the judge
  scored 0 with ``pred == "no"`` and (ii) are not purely visual questions, i.e. the Music-AVQA
  ``type`` field does not contain ``"Visual"`` (audio and audio-visual question types only).

Like the released splits, the output contains only item identifiers and probe outputs; the
questions and answers are joined from the official releases at evaluation time
(``UniMambaMia/official_annotations.py``).

Usage:
    python build_hard_splits.py avqa  --probe avqa_gpt4o_single_frame.jsonl       --output_dir ./hard_splits_out [--push_to_hub user/avqa_hard]
    python build_hard_splits.py music --probe music_avqa_gpt4o_single_frame.jsonl --output_dir ./hard_splits_out [--push_to_hub user/music_avqa_hard]

The script writes a ``datasets`` ``DatasetDict`` with a single ``test`` split to ``--output_dir``
(``save_to_disk``) and optionally pushes it to the Hugging Face Hub.
"""

import argparse
import json
import sys
from pathlib import Path

import datasets

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "UniMambaMia"))
from official_annotations import attach_music_avqa  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _str(value):
    return None if value is None else str(value)


def build_avqa_hard(probe_path, source="gwkrsrch/avqa_2025"):
    probe = {str(r["id"]): r for r in load_jsonl(probe_path)}
    src = datasets.load_dataset(source)["test"]
    keep = []
    for sample in src:
        r = probe.get(str(sample["id"]))
        if r is None:
            continue  # not probed (missing video)
        if not r.get("correct", False):
            keep.append({
                "id": str(sample["id"]),
                "data_source": sample["data_source"],
                "video_duration": _str(r.get("video_duration")),
                "gpt4o_raw": _str(r.get("model_raw")),
                "gpt4o_pred_letter": _str(r.get("pred_letter")),
                "gpt4o_pred_index": _str(None if r.get("pred_index") is None else float(r["pred_index"])),
                "gpt4o_correct": str(bool(r.get("correct", False))),
            })
    print(f"AVQA: {len(src)} items, {len(probe)} probed, {len(keep)} kept as AVQA-Hard")
    return datasets.Dataset.from_list(keep)


def build_music_avqa_hard(probe_path, source="gwkrsrch/music_avqa"):
    probe = {int(r["id"]): r for r in load_jsonl(probe_path)}
    src = attach_music_avqa(datasets.load_dataset(source)["test"])  # adds the official `type` field
    keep = []
    for sample in src:
        r = probe.get(int(sample["source_index"]))
        if r is None:
            continue
        judged_wrong = r.get("judge_score", 0) == 0 and str(r.get("judge_pred", "yes")).lower() == "no"
        audio_related = '"Visual"' not in str(sample.get("type", ""))
        if judged_wrong and audio_related:
            keep.append({
                "source_index": int(sample["source_index"]),
                "question_id": sample["question_id"],
                "gpt4o_pred": r.get("pred", ""),
                "judge_pred": r.get("judge_pred", "no"),
                "judge_score": int(r.get("judge_score", 0)),
            })
    print(f"Music-AVQA: {len(src)} items, {len(probe)} probed, {len(keep)} kept as Music-AVQA-Hard")
    return datasets.Dataset.from_list(keep)


def main():
    p = argparse.ArgumentParser(description="Build the Hard splits from GPT-4o single-frame probe outputs.")
    p.add_argument("benchmark", choices=["avqa", "music"])
    p.add_argument("--probe", required=True, help="JSONL written by probe_*_single_frame.py")
    p.add_argument("--source", default=None, help="Source HF dataset (default: gwkrsrch/avqa_2025 or gwkrsrch/music_avqa)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--push_to_hub", default=None, help="Optional HF dataset id to push to")
    args = p.parse_args()

    if args.benchmark == "avqa":
        ds = build_avqa_hard(args.probe, args.source or "gwkrsrch/avqa_2025")
    else:
        ds = build_music_avqa_hard(args.probe, args.source or "gwkrsrch/music_avqa")

    dd = datasets.DatasetDict({"test": ds})
    dd.save_to_disk(args.output_dir)
    print(f"Saved to {args.output_dir}")
    if args.push_to_hub:
        dd.push_to_hub(args.push_to_hub)
        print(f"Pushed to https://huggingface.co/datasets/{args.push_to_hub}")


if __name__ == "__main__":
    main()
