#!/usr/bin/env python3
# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# Reads the per-sample JSONL logs written by lmms-eval (https://github.com/EvolvingLMMs-Lab/lmms-eval);
# `is_correct` mirrors the metric fields that each lmms-eval task writes.
#
"""
Build the single-frame filter list from GPT-4o probe logs.

The paper's benchmark audit asks whether an item can be answered from a single muted frame.
GPT-4o is run through lmms-eval (model ``gpt4o``, ``max_frames_num=1`` = temporally central
frame) twice, at temperature 0 and temperature 1.0. An item is marked *single-frame answerable*
only if both runs are genuinely correct: the benchmark's own parser marks the answer correct AND
the response is not a refusal.

Refusal detection matters because lmms-eval's multiple-choice parsers extract option letters
aggressively; a refusal such as "I Can't determine ..." can be parsed as "C" and be counted
correct by chance. Without this check, AV-SpeakerBench looked 25% single-frame answerable; with
it, 1.4%.

Output: ``{task_name: [doc_id, ...]}`` where ``doc_id`` is the 0-based index of the sample in
the benchmark's evaluation split (the ``doc_id`` written by lmms-eval).

Usage:
    python benchmark_audit/generate_single_frame_filter.py \
        --log_dir ./eval_logs/gpt4o_single_frame \
        --temp0_timestamps 20260302_142358,20260302_142629 \
        --temp1_timestamps 20260303_170129,20260303_170132,20260303_170135 \
        --output benchmark_audit/single_frame_filter.json
"""

import argparse
import json
import os
import re
from collections import defaultdict


# Patterns that indicate the model refused or couldn't answer the question.
# If any of these match the response text, it is treated as an incorrect answer
# regardless of what the benchmark parser extracted.
REFUSAL_PATTERNS = [
    # Explicit refusal / inability
    r"(?i)\bi\s*(?:can't|cannot|can\s*not)\b",
    r"(?i)\bi'?m\s*(?:unable|not\s*able)\b",
    r"(?i)\bi\s*am\s*unable\b",
    # Apology
    r"(?i)\b(?:sorry|apologize|apologies)\b",
    # Specific inability phrases
    r"(?i)\bunable\s+to\s+(?:determine|identify|see|hear|watch|view|analyze|assess|answer|provide|tell)\b",
    r"(?i)\b(?:cannot|can't)\s+(?:determine|identify|see|hear|watch|view|analyze|assess|answer|provide|tell)\b",
    r"(?i)\bnot\s+(?:possible|able)\s+(?:to|for)\b",
    # Media unavailability
    r"(?i)\b(?:no\s+)?(?:audio|video|sound)\s+(?:is\s+)?(?:available|provided|present)\b",
    r"(?i)\bwithout\s+(?:audio|video|sound|being\s+able)\b",
    # AI identity disclaimers
    r"(?i)\b(?:as\s+an?\s+AI|as\s+a\s+(?:text|language)\s+model)\b",
    # Insufficient information
    r"(?i)\b(?:don't|do\s+not)\s+have\s+(?:enough|sufficient)\s+(?:information|context)\b",
    r"(?i)\b(?:not\s+)?(?:enough|sufficient)\s+(?:information|context|detail)\b",
    # Uncertainty / don't know
    r"(?i)\bi\s*(?:don't|do\s*not)\s*know\b",
    r"(?i)\bi'?m\s*not\s*sure\b",
    r"(?i)\bi\s+am\s+not\s+sure\b",
    # Difficulty expressing
    r"(?i)\b(?:difficult|hard|impossible)\s+to\s+(?:determine|tell|identify|see|hear|know|answer)\b",
    # Image/video doesn't show enough
    r"(?i)\bimage\s+(?:does\s*n[o']?t|alone\s+(?:does\s*n[o']?t|cannot|can't))\b",
    r"(?i)\b(?:doesn't|does\s+not)\s+(?:provide|show|contain|depict)\s+(?:enough|sufficient|a\s+landmark|the)\b",
]

_compiled_refusal = [re.compile(p) for p in REFUSAL_PATTERNS]


def has_refusal(text):
    """Return True if the response text contains refusal language."""
    for pat in _compiled_refusal:
        if pat.search(text):
            return True
    return False


def get_response_text(sample):
    """Extract the model's response text from a sample."""
    if "filtered_resps" in sample:
        resps = sample["filtered_resps"]
        if isinstance(resps, list) and len(resps) > 0:
            return str(resps[0])
        return str(resps)
    return ""


def is_correct(sample):
    """Determine if a sample was answered correctly based on benchmark-specific metrics."""

    # exact_match (boolean) - nextqa_mc_test, avqa_hard, avqa_2025
    if "exact_match" in sample:
        val = sample["exact_match"]
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return val >= 1.0

    # gpt_eval_score.Correctness - activitynetqa, music_avqa, music_avqa_hard
    if "gpt_eval_score" in sample and isinstance(sample["gpt_eval_score"], dict):
        return sample["gpt_eval_score"].get("Correctness") == "yes"

    # videomme_perception_score - videomme
    if "videomme_perception_score" in sample and isinstance(sample["videomme_perception_score"], dict):
        s = sample["videomme_perception_score"]
        return s.get("pred_answer") == s.get("answer")

    # lvb_acc - longvideobench_val_v
    if "lvb_acc" in sample and isinstance(sample["lvb_acc"], dict):
        return sample["lvb_acc"].get("parsed_pred") == sample["lvb_acc"].get("answer")

    # mmmu_acc - video_mmmu_perception, video_mmmu_comprehension, video_mmmu_adaptation
    if "mmmu_acc" in sample and isinstance(sample["mmmu_acc"], dict):
        return sample["mmmu_acc"].get("parsed_pred") == sample["mmmu_acc"].get("answer")

    # exact_match_circular - vnbench_circular
    if "exact_match_circular" in sample and isinstance(sample["exact_match_circular"], dict):
        return sample["exact_match_circular"].get("exact_match") is True

    # worldsense_score - worldsense
    if "worldsense_score" in sample and isinstance(sample["worldsense_score"], dict):
        return sample["worldsense_score"].get("score", 0) >= 1.0

    # av_speakerbench_score - av_speakerbench_audiovisual
    if "av_speakerbench_score" in sample and isinstance(sample["av_speakerbench_score"], dict):
        return sample["av_speakerbench_score"].get("score", 0) >= 1.0

    # avg_accuracy.match_success - tempcompass_multi_choice
    if "avg_accuracy" in sample and isinstance(sample["avg_accuracy"], dict):
        return sample["avg_accuracy"].get("match_success") is True

    # Fallback: compare filtered_resps to target
    if "filtered_resps" in sample and "target" in sample:
        resp = str(sample["filtered_resps"][0]).strip().lower()
        target = str(sample["target"]).strip().lower()
        return resp == target

    return False


def is_genuinely_correct(sample):
    """Return True only if the parser says correct AND the response is not a refusal."""
    if not is_correct(sample):
        return False
    resp_text = get_response_text(sample)
    if has_refusal(resp_text):
        return False
    return True


def extract_task_name_from_filename(filename):
    """Extract task name from JSONL filename like '20260302_142358_samples_videomme.jsonl'."""
    match = re.match(r"\d{8}_\d{6}_samples_(.+)\.jsonl", filename)
    if match:
        return match.group(1)
    return None


def load_correct_doc_ids(log_dir, timestamps):
    """Load all sample files for given timestamps and return
    {task_name: set(correct_doc_ids)}, {task_name: set(all_doc_ids)},
    {task_name: int(refusal_false_positives)}."""
    task_correct = defaultdict(set)
    task_all_doc_ids = defaultdict(set)
    task_refusal_fp = defaultdict(int)  # refusals that parser marked correct

    for ts in timestamps:
        for filename in sorted(os.listdir(log_dir)):
            if not filename.startswith(ts) or not filename.endswith(".jsonl"):
                continue
            task_name = extract_task_name_from_filename(filename)
            if task_name is None:
                continue

            filepath = os.path.join(log_dir, filename)
            with open(filepath) as f:
                for line in f:
                    sample = json.loads(line)
                    doc_id = sample["doc_id"]
                    task_all_doc_ids[task_name].add(doc_id)

                    parser_correct = is_correct(sample)
                    genuinely_correct = is_genuinely_correct(sample)

                    if genuinely_correct:
                        task_correct[task_name].add(doc_id)
                    elif parser_correct:
                        # Parser said correct but response was a refusal
                        task_refusal_fp[task_name] += 1

    return task_correct, task_all_doc_ids, task_refusal_fp


def main():
    parser = argparse.ArgumentParser(
        description="Build the single-frame answerable filter list from GPT-4o probe logs (refusal-aware)"
    )
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing lmms-eval log files")
    parser.add_argument(
        "--temp0_timestamps",
        type=str,
        required=True,
        help="Comma-separated timestamps for temperature=0 runs",
    )
    parser.add_argument(
        "--temp1_timestamps",
        type=str,
        required=True,
        help="Comma-separated timestamps for temperature=1 runs",
    )
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    args = parser.parse_args()

    temp0_ts = [t.strip() for t in args.temp0_timestamps.split(",")]
    temp1_ts = [t.strip() for t in args.temp1_timestamps.split(",")]

    print(f"Loading temp=0 results from timestamps: {temp0_ts}")
    temp0_correct, temp0_all, temp0_rfp = load_correct_doc_ids(args.log_dir, temp0_ts)

    print(f"Loading temp=1 results from timestamps: {temp1_ts}")
    temp1_correct, temp1_all, temp1_rfp = load_correct_doc_ids(args.log_dir, temp1_ts)

    # Find tasks present in both runs
    all_tasks = sorted(set(temp0_correct.keys()) | set(temp1_correct.keys()) |
                       set(temp0_all.keys()) | set(temp1_all.keys()))

    filter_dict = {}
    header = f"{'Task':<40} {'Total':>7} {'T0 Corr':>8} {'T1 Corr':>8} {'Both':>6} {'Filter%':>8} {'T0 RFP':>7} {'T1 RFP':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    total_filtered = 0
    total_samples = 0
    total_t0_rfp = 0
    total_t1_rfp = 0

    for task in all_tasks:
        t0_set = temp0_correct.get(task, set())
        t1_set = temp1_correct.get(task, set())
        both_correct = t0_set & t1_set

        n_total = len(temp0_all.get(task, set()) | temp1_all.get(task, set()))

        if both_correct:
            filter_dict[task] = sorted(both_correct)

        ratio = len(both_correct) / n_total * 100 if n_total > 0 else 0
        t0_rfp = temp0_rfp.get(task, 0)
        t1_rfp = temp1_rfp.get(task, 0)
        print(f"{task:<40} {n_total:>7} {len(t0_set):>8} {len(t1_set):>8} {len(both_correct):>6} {ratio:>7.1f}% {t0_rfp:>7} {t1_rfp:>7}")

        total_filtered += len(both_correct)
        total_samples += n_total
        total_t0_rfp += t0_rfp
        total_t1_rfp += t1_rfp

    print("=" * len(header))
    overall_ratio = total_filtered / total_samples * 100 if total_samples > 0 else 0
    print(f"{'TOTAL':<40} {total_samples:>7} {'':>8} {'':>8} {total_filtered:>6} {overall_ratio:>7.1f}% {total_t0_rfp:>7} {total_t1_rfp:>7}")
    print()
    print(f"RFP = Refusal False Positives: parser said correct, but response was a refusal")
    print(f"  Total T0 refusal-FP removed: {total_t0_rfp}")
    print(f"  Total T1 refusal-FP removed: {total_t1_rfp}")
    print()

    # Save filter JSON
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(filter_dict, f, indent=2)
    print(f"Filter list saved to: {args.output}")
    print(f"Tasks with filters: {len(filter_dict)}")
    for task, doc_ids in sorted(filter_dict.items()):
        print(f"  {task}: {len(doc_ids)} doc_ids to filter")


if __name__ == "__main__":
    main()
