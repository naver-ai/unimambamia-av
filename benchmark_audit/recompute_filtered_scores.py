#!/usr/bin/env python3
# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# The per-benchmark metric functions reproduce the aggregation of the corresponding lmms-eval tasks
# (https://github.com/EvolvingLMMs-Lab/lmms-eval); the open-ended matching in `_eval_open` and
# `_normalize_str` follows MMMU (https://github.com/MMMU-Benchmark/MMMU, Apache-2.0) by way of the
# lmms-eval VideoMMMU task.
#
"""
Recompute benchmark scores on the single-frame filtered subsets from existing lmms-eval logs.

Reads the per-sample JSONL files written by ``lmms_eval --log_samples``, drops the doc_ids listed
in the filter (see ``single_frame_filter.json``) and re-aggregates every benchmark metric. No
inference is needed, so the "filtered" numbers of the paper (Table 1 bottom, Table 3 right of each
"/") can be reproduced from the same logs that produce the unfiltered numbers.

Usage:
    # one model directory (lmms-eval writes <output_path>/<model_args-derived dir>/<ts>_samples_<task>.jsonl)
    python benchmark_audit/recompute_filtered_scores.py \
        --filter_json benchmark_audit/single_frame_filter.json \
        --log_dir ./eval_logs \
        --model_dirs <model_dir> --show_original

    # several models selected by a regex on the directory name
    python benchmark_audit/recompute_filtered_scores.py \
        --filter_json benchmark_audit/single_frame_filter.json \
        --log_dir ./eval_logs --pattern "UniMambaMia|Qwen" --show_original --output_csv filtered_scores.csv

    # seed groups (mean +- std) via a JSON config {short_name: dir_name, "_seed_groups": {group: [short_name, ...]}}
    python benchmark_audit/recompute_filtered_scores.py --filter_json ... --log_dir ... --model_config models.json
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict


# ============================================================
# Per-benchmark metric computation functions
# Each function takes a list of sample dicts and returns
# a dict of {metric_name: value}.
# ============================================================

def compute_exact_match(samples, task_name):
    """For avqa_2025, avqa_hard, nextqa_mc_test."""
    correct = 0
    for s in samples:
        em = s["exact_match"]
        if isinstance(em, bool) and em:
            correct += 1
        elif isinstance(em, (int, float)) and em >= 1.0:
            correct += 1
    n = len(samples)
    return {"exact_match": correct / n if n else 0}


def compute_gpt_eval(samples, task_name):
    """For activitynetqa, music_avqa, music_avqa_hard."""
    score_sum = 0
    acc_count = 0
    n = len(samples)
    for s in samples:
        ge = s["gpt_eval_score"]
        score_sum += ge["score"]
        if ge.get("Correctness") == "yes":
            acc_count += 1
    return {
        "gpt_eval_score": score_sum / n if n else 0,
        "gpt_eval_accuracy": acc_count / n * 100 if n else 0,
    }


def compute_videomme(samples, task_name):
    """For videomme."""
    correct = 0
    for s in samples:
        vps = s["videomme_perception_score"]
        if vps["pred_answer"] == vps["answer"]:
            correct += 1
    n = len(samples)
    return {"videomme_perception_score": correct / n * 100 if n else 0}


def compute_lvb_acc(samples, task_name):
    """For longvideobench_val_v."""
    correct = 0
    for s in samples:
        la = s["lvb_acc"]
        if la["parsed_pred"] == la["answer"]:
            correct += 1
    n = len(samples)
    return {"lvb_acc": correct / n if n else 0}


def compute_mmmu_acc(samples, task_name):
    """For video_mmmu_perception, video_mmmu_comprehension, video_mmmu_adaptation."""
    correct = 0
    for s in samples:
        ma = s["mmmu_acc"]
        pred = ma["parsed_pred"]
        answer = ma["answer"]
        qt = ma.get("question_type", "multiple-choice")
        if qt == "multiple-choice" or qt == "perception":
            if pred == answer:
                correct += 1
        elif qt == "open":
            # Fuzzy matching for open questions (replicating eval_open logic)
            if _eval_open(answer, pred):
                correct += 1
        elif qt == "None" or qt is None:
            if pred == answer:
                correct += 1
        else:
            if pred == answer:
                correct += 1
    n = len(samples)
    return {"mmmu_acc": correct / n if n else 0}


def _normalize_str(input_str):
    """Normalize string for open-ended MMMU comparison."""
    try:
        return [float(input_str)]
    except ValueError:
        return [input_str.strip().lower()]


def _eval_open(gold_i, pred_i):
    """Evaluate an open question (from video_mmmu utils)."""
    if isinstance(gold_i, list):
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(_normalize_str(answer))
    else:
        norm_answers = _normalize_str(gold_i)

    if isinstance(pred_i, list):
        for pred in pred_i:
            if isinstance(pred, str):
                for norm_ans in norm_answers:
                    if isinstance(norm_ans, str) and norm_ans in pred.lower():
                        return True
            else:
                if pred in norm_answers:
                    return True
    else:
        pred_str = str(pred_i).strip().lower()
        for norm_ans in norm_answers:
            if isinstance(norm_ans, str) and norm_ans in pred_str:
                return True
            elif not isinstance(norm_ans, str):
                try:
                    if float(pred_str) == norm_ans:
                        return True
                except (ValueError, TypeError):
                    pass
    return False


def compute_vnbench(samples, task_name):
    """For vnbench_circular."""
    # exact_match_independent: per-sample accuracy
    indep_correct = 0
    # exact_match_circular: per-group (video_task_id) all-correct
    groups = {}
    for s in samples:
        emc = s["exact_match_circular"]
        if emc.get("exact_match") is True:
            indep_correct += 1
        vid = emc["video_task_id"]
        if vid not in groups:
            groups[vid] = True
        if emc.get("exact_match") is not True:
            groups[vid] = False

    n = len(samples)
    circ_correct = sum(1 for v in groups.values() if v)
    n_groups = len(groups)
    return {
        "exact_match_independent": indep_correct / n if n else 0,
        "exact_match_circular": circ_correct / n_groups * 100 if n_groups else 0,
    }


def compute_worldsense(samples, task_name):
    """For worldsense."""
    score_sum = 0
    for s in samples:
        score_sum += s["worldsense_score"]["score"]
    n = len(samples)
    return {"worldsense_score": score_sum / n * 100 if n else 0}


def compute_av_speakerbench(samples, task_name):
    """For av_speakerbench_audiovisual."""
    score_sum = 0
    for s in samples:
        score_sum += s["av_speakerbench_score"]["score"]
    n = len(samples)
    return {"av_speakerbench_score": score_sum / n * 100 if n else 0}


def compute_tempcompass(samples, task_name):
    """For tempcompass_multi_choice."""
    # avg_accuracy = count(rating==1) / total * 100
    correct = 0
    for s in samples:
        if s["avg_accuracy"]["rating"] == 1:
            correct += 1
    n = len(samples)
    return {"avg_accuracy": correct / n * 100 if n else 0}


# Map task names to their compute functions
TASK_COMPUTE_FN = {
    "avqa_2025": compute_exact_match,
    "avqa_hard": compute_exact_match,
    "nextqa_mc_test": compute_exact_match,
    "activitynetqa": compute_gpt_eval,
    "music_avqa": compute_gpt_eval,
    "music_avqa_hard": compute_gpt_eval,
    "videomme": compute_videomme,
    "longvideobench_val_v": compute_lvb_acc,
    "video_mmmu_perception": compute_mmmu_acc,
    "video_mmmu_comprehension": compute_mmmu_acc,
    "video_mmmu_adaptation": compute_mmmu_acc,
    "vnbench_circular": compute_vnbench,
    "worldsense": compute_worldsense,
    "av_speakerbench_audiovisual": compute_av_speakerbench,
    "tempcompass_multi_choice": compute_tempcompass,
}

# Primary metric to report per task (for the CSV summary)
TASK_PRIMARY_METRIC = {
    "activitynetqa": "gpt_eval_accuracy",
    "av_speakerbench_audiovisual": "av_speakerbench_score",
    "avqa_2025": "exact_match",
    "avqa_hard": "exact_match",
    "longvideobench_val_v": "lvb_acc",
    "music_avqa": "gpt_eval_accuracy",
    "music_avqa_hard": "gpt_eval_accuracy",
    "nextqa_mc_test": "exact_match",
    "tempcompass_multi_choice": "avg_accuracy",
    "video_mmmu_perception": "mmmu_acc",
    "video_mmmu_comprehension": "mmmu_acc",
    "video_mmmu_adaptation": "mmmu_acc",
    "videomme": "videomme_perception_score",
    "vnbench_circular": "exact_match_circular",
    "worldsense": "worldsense_score",
}

# Canonical display order for benchmarks
BENCH_ORDER = [
    "activitynetqa",
    "av_speakerbench_audiovisual",
    "avqa_2025",
    "avqa_hard",
    "longvideobench_val_v",
    "music_avqa",
    "music_avqa_hard",
    "nextqa_mc_test",
    "tempcompass_multi_choice",
    "video_mmmu",
    "video_mmmu_perception",
    "video_mmmu_comprehension",
    "video_mmmu_adaptation",
    "videomme",
    "vnbench_circular",
    "worldsense",
]


def extract_task_name_from_filename(filename):
    """Extract task name from JSONL filename like '20260301_012014_samples_videomme.jsonl'."""
    match = re.match(r"\d{8}_\d{6}_samples_(.+)\.jsonl", filename)
    if match:
        return match.group(1)
    return None


def load_samples(log_dir, model_dir):
    """Load all sample JSONL files for a model directory.
    Returns {task_name: [sample_dicts]}.
    """
    dirpath = os.path.join(log_dir, model_dir)
    task_samples = {}

    jsonl_files = sorted(glob.glob(os.path.join(dirpath, "*_samples_*.jsonl")))
    if not jsonl_files:
        return task_samples

    # Group files by task name, then pick the best one per task
    task_files = {}  # task_name -> list of (filepath, filename)
    for fpath in jsonl_files:
        fname = os.path.basename(fpath)
        task_name = extract_task_name_from_filename(fname)
        if task_name is None:
            continue
        if task_name not in TASK_COMPUTE_FN:
            continue
        task_files.setdefault(task_name, []).append((fpath, fname))

    for task_name, files in task_files.items():
        # Files are already sorted by timestamp (YYYYMMDD_HHMMSS prefix).
        # Always use the latest (last) file for each task.
        fpath, fname = files[-1]
        samples = []
        try:
            with open(fpath) as f:
                for line in f:
                    samples.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  WARNING: Partial/corrupt JSONL {fname} ({len(samples)} lines read before error: {e})")
            # If latest file is corrupt, fall back to the previous complete file
            if len(files) > 1:
                for prev_fpath, prev_fname in reversed(files[:-1]):
                    prev_samples = []
                    try:
                        with open(prev_fpath) as f:
                            for line in f:
                                prev_samples.append(json.loads(line))
                        print(f"  -> Falling back to {prev_fname} ({len(prev_samples)} samples)")
                        samples = prev_samples
                        break
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
        if samples:
            task_samples[task_name] = samples

    return task_samples


def apply_filter(task_samples, filter_dict):
    """Remove filtered doc_ids from samples.
    Returns {task_name: filtered_samples}, filter_stats.
    """
    filtered_samples = {}
    filter_stats = {}

    for task_name, samples in task_samples.items():
        if task_name in filter_dict:
            filtered_ids = filter_dict[task_name]
            original_count = len(samples)
            remaining = [s for s in samples if s["doc_id"] not in filtered_ids]
            removed = original_count - len(remaining)
            filtered_samples[task_name] = remaining
            filter_stats[task_name] = {
                "original": original_count,
                "filtered": removed,
                "remaining": len(remaining),
                "filter_pct": round(removed / original_count * 100, 1) if original_count else 0,
            }
        else:
            filtered_samples[task_name] = samples
            filter_stats[task_name] = {
                "original": len(samples),
                "filtered": 0,
                "remaining": len(samples),
                "filter_pct": 0,
            }

    return filtered_samples, filter_stats


def compute_scores(task_samples):
    """Compute aggregate scores for each task.
    Returns {task_name: {metric_name: value}}.
    """
    scores = {}
    for task_name, samples in task_samples.items():
        if task_name in TASK_COMPUTE_FN:
            scores[task_name] = TASK_COMPUTE_FN[task_name](samples, task_name)

    # Compute video_mmmu group score (weighted average by size)
    mmmu_subtasks = ["video_mmmu_perception", "video_mmmu_comprehension", "video_mmmu_adaptation"]
    subtask_scores = []
    subtask_sizes = []
    for sub in mmmu_subtasks:
        if sub in scores:
            subtask_scores.append(scores[sub]["mmmu_acc"])
            subtask_sizes.append(len(task_samples[sub]))

    if subtask_scores:
        total_size = sum(subtask_sizes)
        if total_size > 0:
            weighted_avg = sum(s * n for s, n in zip(subtask_scores, subtask_sizes)) / total_size
        else:
            weighted_avg = sum(subtask_scores) / len(subtask_scores)
        scores["video_mmmu"] = {"mmmu_acc": weighted_avg}

    return scores


def get_primary_score(scores, task_name):
    """Get the primary metric value for display."""
    if task_name not in scores:
        return None
    if task_name == "video_mmmu":
        return scores[task_name].get("mmmu_acc")
    metric = TASK_PRIMARY_METRIC.get(task_name)
    if metric and metric in scores[task_name]:
        return scores[task_name][metric]
    return None


def format_score(score, task_name):
    """Format score for display."""
    if score is None:
        return "-"
    # Metrics that are already percentages
    pct_metrics = {
        "activitynetqa", "av_speakerbench_audiovisual", "music_avqa",
        "music_avqa_hard", "tempcompass_multi_choice", "videomme",
        "vnbench_circular", "worldsense",
    }
    if task_name in pct_metrics:
        return f"{score:.2f}"
    else:
        # Fraction-based metrics (exact_match, lvb_acc, mmmu_acc) -- display as percentage
        return f"{score * 100:.2f}"


def short_model_name(model_dir):
    """Create a shorter display name from the long directory name."""
    name = model_dir
    # Remove common prefixes
    name = re.sub(r"^checkpoints__", "", name)
    name = re.sub(r"^(further_)?video_llava_", "", name)
    # Remove common middle parts
    name = re.sub(r"newnewdiet_siglip2_16_384_qwen2_7b_vp_flat_video_frame_fft_video_avepool2_", "", name)
    name = re.sub(r"_fps1_0_mf32_L\d+_\d+n_ddn_square_", "_", name)
    name = re.sub(r"_fps1_0_mf64_L\d+_\d+n_ddn_square_", "_", name)
    # Clean up
    name = re.sub(r"__+", "_", name)
    name = name.strip("_")
    return name


def main():
    parser = argparse.ArgumentParser(description="Recompute filtered scores from existing JSONL logs")
    parser.add_argument("--filter_json", type=str, required=True,
                        help="Path to the single-frame filter JSON (benchmark_audit/single_frame_filter.json)")
    parser.add_argument("--log_dir", type=str, default="./eval_logs",
                        help="Directory containing model log subdirectories")
    parser.add_argument("--model_dirs", type=str, default=None,
                        help="Comma-separated list of model directory names")
    parser.add_argument("--model_config", type=str, default=None,
                        help="JSON file mapping short_name -> dir_name (ignores keys starting with _)")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Regex pattern to match model directory names")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV file path")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Output JSON file path (detailed results)")
    parser.add_argument("--show_original", action="store_true",
                        help="Also compute and show original (unfiltered) scores")
    args = parser.parse_args()

    # Load filter
    with open(args.filter_json) as f:
        raw_filter = json.load(f)
    filter_dict = {task: set(doc_ids) for task, doc_ids in raw_filter.items()}
    total_filtered = sum(len(v) for v in filter_dict.values())
    print(f"Loaded filter: {total_filtered} doc_ids across {len(filter_dict)} tasks")

    # Determine model directories
    name_to_dir = {}  # short_name -> dir_name (for display)
    if args.model_config:
        with open(args.model_config) as f:
            config = json.load(f)
        model_dirs = []
        for key, dirname in config.items():
            if key.startswith("_"):
                continue
            dirpath = os.path.join(args.log_dir, dirname)
            if not os.path.isdir(dirpath):
                print(f"WARNING: Directory not found: {dirname}")
                continue
            model_dirs.append(dirname)
            name_to_dir[dirname] = key
    elif args.model_dirs:
        model_dirs = [d.strip() for d in args.model_dirs.split(",")]
    elif args.pattern:
        all_dirs = sorted(os.listdir(args.log_dir))
        model_dirs = [d for d in all_dirs if re.search(args.pattern, d)]
    else:
        print("Error: Provide --model_dirs, --model_config, or --pattern")
        sys.exit(1)

    print(f"Processing {len(model_dirs)} model directories...\n")

    # Helper to get display name
    def display_name(mdir):
        return name_to_dir.get(mdir, short_model_name(mdir))

    # Process each model
    all_results = {}  # model_dir -> {task -> {metric: value}}
    all_original = {}  # model_dir -> {task -> {metric: value}} (unfiltered)
    all_filter_stats = {}
    all_sample_counts = {}  # model_dir -> {task -> n_samples}

    for i, mdir in enumerate(model_dirs):
        print(f"[{i+1}/{len(model_dirs)}] {display_name(mdir)}")
        task_samples = load_samples(args.log_dir, mdir)

        if not task_samples:
            print(f"  -> No sample files found, skipping")
            continue

        # Compute original scores
        if args.show_original:
            original_scores = compute_scores(task_samples)
            all_original[mdir] = original_scores

        # Apply filter and compute filtered scores
        filtered_samples, fstats = apply_filter(task_samples, filter_dict)
        filtered_scores = compute_scores(filtered_samples)

        all_results[mdir] = filtered_scores
        all_filter_stats[mdir] = fstats
        all_sample_counts[mdir] = {t: len(s) for t, s in filtered_samples.items()}

        # Print per-model summary
        for task in BENCH_ORDER:
            fscore = get_primary_score(filtered_scores, task)
            if fscore is not None:
                fstr = format_score(fscore, task)
                if args.show_original and mdir in all_original:
                    oscore = get_primary_score(original_scores, task)
                    ostr = format_score(oscore, task) if oscore is not None else "-"
                    stats = fstats.get(task, {})
                    print(f"  {task:<35} orig={ostr:>8}  filt={fstr:>8}  (removed {stats.get('filtered',0)}/{stats.get('original',0)})")
                else:
                    stats = fstats.get(task, {})
                    print(f"  {task:<35} {fstr:>8}  (removed {stats.get('filtered',0)}/{stats.get('original',0)})")
        print()

    # Build CSV output
    if args.output_csv:
        _write_csv(args.output_csv, model_dirs, all_results, all_original if args.show_original else None, display_name)
        print(f"CSV saved to: {args.output_csv}")

    # Build JSON output
    if args.output_json:
        _write_json(args.output_json, model_dirs, all_results, all_filter_stats, all_sample_counts,
                     all_original if args.show_original else None, display_name)
        print(f"JSON saved to: {args.output_json}")

    # Compute seed-group mean±std if config has _seed_groups
    seed_groups = {}
    if args.model_config:
        with open(args.model_config) as f:
            cfg = json.load(f)
        if "_seed_groups" in cfg:
            seed_groups = cfg["_seed_groups"]

    group_stats = {}  # group_name -> {task -> {"mean": float, "std": float, "values": [...]}}
    if seed_groups:
        # Map short_name -> dir_name for lookup
        short_to_dir = {}
        for k, v in config.items():
            if not k.startswith("_"):
                short_to_dir[k] = v

        for gname, members in seed_groups.items():
            group_stats[gname] = {}
            member_dirs = [short_to_dir[m] for m in members if m in short_to_dir]
            for task in BENCH_ORDER:
                values = []
                for mdir in member_dirs:
                    if mdir in all_results:
                        s = get_primary_score(all_results[mdir], task)
                        if s is not None:
                            # Normalize to percentage
                            pct_metrics = {
                                "activitynetqa", "av_speakerbench_audiovisual", "music_avqa",
                                "music_avqa_hard", "tempcompass_multi_choice", "videomme",
                                "vnbench_circular", "worldsense",
                            }
                            val = s if task in pct_metrics else s * 100
                            values.append(val)
                if values:
                    mean = sum(values) / len(values)
                    if len(values) > 1:
                        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
                    else:
                        std = 0.0
                    group_stats[gname][task] = {"mean": mean, "std": std, "values": values}

        # Also compute group stats for original scores
        group_stats_orig = {}
        if args.show_original and all_original:
            for gname, members in seed_groups.items():
                group_stats_orig[gname] = {}
                member_dirs = [short_to_dir[m] for m in members if m in short_to_dir]
                for task in BENCH_ORDER:
                    values = []
                    for mdir in member_dirs:
                        if mdir in all_original:
                            s = get_primary_score(all_original[mdir], task)
                            if s is not None:
                                pct_metrics = {
                                    "activitynetqa", "av_speakerbench_audiovisual", "music_avqa",
                                    "music_avqa_hard", "tempcompass_multi_choice", "videomme",
                                    "vnbench_circular", "worldsense",
                                }
                                val = s if task in pct_metrics else s * 100
                                values.append(val)
                    if values:
                        mean = sum(values) / len(values)
                        if len(values) > 1:
                            std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
                        else:
                            std = 0.0
                        group_stats_orig[gname][task] = {"mean": mean, "std": std, "values": values}
        else:
            group_stats_orig = None

        # Write group summary CSV
        if args.output_csv:
            group_csv_path = args.output_csv.replace(".csv", "_group_mean_std.csv")
            _write_group_csv(group_csv_path, seed_groups, group_stats, group_stats_orig)
            print(f"Group mean±std CSV saved to: {group_csv_path}")

        # Append standalone models to group_stats for unified display
        standalone_models = {}
        all_member_shorts = set()
        for members in seed_groups.values():
            all_member_shorts.update(members)
        for k, v in config.items():
            if k.startswith("_"):
                continue
            if k not in all_member_shorts:
                # Standalone model
                standalone_models[k] = v

    # Print unified table
    print("\n" + "=" * 120)
    print("UNIFIED FILTERED SCORES TABLE")
    print("=" * 120)
    _print_table(model_dirs, all_results, all_original if args.show_original else None, display_name)

    # Print mean±std table
    if group_stats:
        print("\n" + "=" * 120)
        print("SEED-GROUP MEAN±STD TABLE (filtered)")
        print("=" * 120)
        _print_group_table(seed_groups, group_stats, standalone_models if seed_groups else {},
                           all_results, display_name)

        if group_stats_orig:
            print("\n" + "=" * 120)
            print("SEED-GROUP MEAN±STD TABLE (original)")
            print("=" * 120)
            _print_group_table(seed_groups, group_stats_orig, standalone_models if seed_groups else {},
                               all_original, display_name)


def _write_csv(path, model_dirs, all_results, all_original=None, display_name=None):
    """Write results to CSV."""
    if display_name is None:
        display_name = short_model_name
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    headers = ["model"]
    for task in BENCH_ORDER:
        if all_original:
            headers.append(f"{task}_orig")
        headers.append(f"{task}_filt")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for mdir in model_dirs:
            if mdir not in all_results:
                continue
            row = [display_name(mdir)]
            for task in BENCH_ORDER:
                if all_original and mdir in all_original:
                    oscore = get_primary_score(all_original[mdir], task)
                    row.append(format_score(oscore, task))
                elif all_original:
                    row.append("-")
                fscore = get_primary_score(all_results[mdir], task)
                row.append(format_score(fscore, task))
            writer.writerow(row)


def _write_json(path, model_dirs, all_results, all_filter_stats, all_sample_counts, all_original=None, display_name=None):
    """Write detailed results to JSON."""
    if display_name is None:
        display_name = short_model_name
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    output = {}
    for mdir in model_dirs:
        if mdir not in all_results:
            continue
        entry = {
            "short_name": display_name(mdir),
            "filtered_scores": {},
            "filter_stats": all_filter_stats.get(mdir, {}),
            "sample_counts": all_sample_counts.get(mdir, {}),
        }
        if all_original and mdir in all_original:
            entry["original_scores"] = {}
            for task in BENCH_ORDER:
                oscore = get_primary_score(all_original[mdir], task)
                if oscore is not None:
                    entry["original_scores"][task] = round(oscore, 5)

        for task in BENCH_ORDER:
            fscore = get_primary_score(all_results[mdir], task)
            if fscore is not None:
                entry["filtered_scores"][task] = round(fscore, 5)

        output[mdir] = entry

    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def _print_table(model_dirs, all_results, all_original=None, display_name=None):
    """Print a readable table."""
    if display_name is None:
        display_name = short_model_name
    # Determine which tasks have data
    tasks_with_data = []
    for task in BENCH_ORDER:
        if any(get_primary_score(all_results.get(m, {}), task) is not None for m in model_dirs):
            tasks_with_data.append(task)

    # Short task names for display
    short_task = {
        "activitynetqa": "ActNet",
        "av_speakerbench_audiovisual": "AVSpkr",
        "avqa_2025": "AVQA",
        "avqa_hard": "AVQA-H",
        "longvideobench_val_v": "LVB",
        "music_avqa": "MAVQA",
        "music_avqa_hard": "MAVQA-H",
        "nextqa_mc_test": "NeXT",
        "tempcompass_multi_choice": "TempC",
        "video_mmmu": "VMMMU",
        "video_mmmu_perception": "VMMMU-P",
        "video_mmmu_comprehension": "VMMMU-C",
        "video_mmmu_adaptation": "VMMMU-A",
        "videomme": "VidMME",
        "vnbench_circular": "VNB",
        "worldsense": "WS",
    }

    col_w = 8
    name_w = 30
    header = f"{'Model':<{name_w}}"
    for t in tasks_with_data:
        header += f" {short_task.get(t, t):>{col_w}}"
    print(header)
    print("-" * len(header))

    for mdir in model_dirs:
        if mdir not in all_results:
            continue
        row = f"{display_name(mdir):<{name_w}}"
        for t in tasks_with_data:
            fscore = get_primary_score(all_results[mdir], t)
            row += f" {format_score(fscore, t):>{col_w}}"
        print(row)

    if all_original:
        print("\n(Original unfiltered scores)")
        print("-" * len(header))
        for mdir in model_dirs:
            if mdir not in all_original:
                continue
            row = f"{display_name(mdir):<{name_w}}"
            for t in tasks_with_data:
                oscore = get_primary_score(all_original[mdir], t)
                row += f" {format_score(oscore, t):>{col_w}}"
            print(row)


def _write_group_csv(path, seed_groups, group_stats, group_stats_orig=None):
    """Write group mean±std summary to CSV."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    headers = ["model"]
    for task in BENCH_ORDER:
        if group_stats_orig:
            headers.append(f"{task}_orig")
        headers.append(f"{task}_filt")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for gname in seed_groups:
            if gname not in group_stats:
                continue
            row = [gname]
            for task in BENCH_ORDER:
                if group_stats_orig and gname in group_stats_orig and task in group_stats_orig[gname]:
                    gs = group_stats_orig[gname][task]
                    row.append(f"{gs['mean']:.2f}±{gs['std']:.2f}")
                elif group_stats_orig:
                    row.append("-")
                if task in group_stats[gname]:
                    gs = group_stats[gname][task]
                    row.append(f"{gs['mean']:.2f}±{gs['std']:.2f}")
                else:
                    row.append("-")
            writer.writerow(row)


def _print_group_table(seed_groups, group_stats, standalone_models, all_results, display_name):
    """Print mean±std table for seed groups + standalone models."""
    # Determine which tasks have data
    tasks_with_data = []
    for task in BENCH_ORDER:
        has_data = any(task in group_stats.get(g, {}) for g in seed_groups)
        if not has_data:
            has_data = any(
                get_primary_score(all_results.get(d, {}), task) is not None
                for d in standalone_models.values()
            )
        if has_data:
            tasks_with_data.append(task)

    short_task = {
        "activitynetqa": "ActNet",
        "av_speakerbench_audiovisual": "AVSpkr",
        "avqa_2025": "AVQA",
        "avqa_hard": "AVQA-H",
        "longvideobench_val_v": "LVB",
        "music_avqa": "MAVQA",
        "music_avqa_hard": "MAVQA-H",
        "nextqa_mc_test": "NeXT",
        "tempcompass_multi_choice": "TempC",
        "video_mmmu": "VMMMU",
        "video_mmmu_perception": "VMMMU-P",
        "video_mmmu_comprehension": "VMMMU-C",
        "video_mmmu_adaptation": "VMMMU-A",
        "videomme": "VidMME",
        "vnbench_circular": "VNB",
        "worldsense": "WS",
    }

    col_w = 13
    name_w = 25
    header = f"{'Model':<{name_w}}"
    for t in tasks_with_data:
        header += f" {short_task.get(t, t):>{col_w}}"
    print(header)
    print("-" * len(header))

    # Print seed group rows
    for gname in seed_groups:
        if gname not in group_stats:
            continue
        row = f"{gname:<{name_w}}"
        for t in tasks_with_data:
            if t in group_stats[gname]:
                gs = group_stats[gname][t]
                cell = f"{gs['mean']:.2f}±{gs['std']:.2f}"
            else:
                cell = "-"
            row += f" {cell:>{col_w}}"
        print(row)

    # Print standalone model rows
    if standalone_models:
        print("-" * len(header))
        for sname, sdir in standalone_models.items():
            if sdir not in all_results:
                continue
            row = f"{sname:<{name_w}}"
            for t in tasks_with_data:
                score = get_primary_score(all_results[sdir], t)
                if score is not None:
                    pct_metrics = {
                        "activitynetqa", "av_speakerbench_audiovisual", "music_avqa",
                        "music_avqa_hard", "tempcompass_multi_choice", "videomme",
                        "vnbench_circular", "worldsense",
                    }
                    val = score if t in pct_metrics else score * 100
                    cell = f"{val:.2f}"
                else:
                    cell = "-"
                row += f" {cell:>{col_w}}"
            print(row)


if __name__ == "__main__":
    main()
