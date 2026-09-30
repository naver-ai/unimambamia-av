#!/usr/bin/env python3
# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# `parse_verbose_response` follows the multiple-choice parser of MMMU
# (https://github.com/MMMU-Benchmark/MMMU, Apache-2.0), which the AVQA tasks in this repository also
# use, so that the probe and the benchmark parse answers the same way.
"""
GPT-4o single-frame probe for AVQA (multiple choice), used to build AVQA-Hard.

For every item the temporally central frame of the clip is extracted with ffmpeg (longer side
capped at 384 px, no upscaling) and sent to GPT-4o together with the question and the options.
No audio and no other frames are provided. Items that GPT-4o answers correctly from this single
muted frame are considered visual shortcuts; AVQA-Hard keeps the items it gets wrong
(see ``build_hard_splits.py``).

Input: either ``--dataset gwkrsrch/avqa_2025`` (item ids; questions and options are joined from the
official AVQA release by ``UniMambaMia/official_annotations.py``) or ``--val-jsonl``, a JSONL file in
LLaVA conversation format, one item per line::

    {"id": "1144", "video": "abcd1234.mp4",
     "conversations": [{"from": "human", "value": "<image>\\nQuestion ...\\nOptions:\\nA. ...\\nB. ..."},
                       {"from": "gpt", "value": "A. ..."}],
     "answer": 0,                                  # 0-based index of the correct option
     "raw_annotated_multi_choice": ["...", "..."]}  # option texts

Output: JSONL with ``id, video, video_duration, gt_answer_index, gt_answer_letter, pred_letter,
pred_index, correct, model_raw`` per item. Re-running resumes from the existing output.

Usage:
    export OPENAI_API_KEY=...
    python probe_avqa_single_frame.py --dataset gwkrsrch/avqa_2025 --video-root $HF_HOME/avqa_cropped_videos_loadable \\
        --output avqa_gpt4o_single_frame.jsonl --model gpt-4o-2024-08-06
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

LETTERS = "ABCDEFGH"
LETTER_REGEX = re.compile(r"\b([A-H])\b", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description="Single-frame MCQA probe with GPT-4o (AVQA).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--val-jsonl", type=Path, help="AVQA items in LLaVA conversation format (JSONL).")
    src.add_argument("--dataset", type=str, help="Released AVQA split, e.g. gwkrsrch/avqa_2025 (test split); the official annotations are joined on the fly.")
    p.add_argument("--video-root", type=Path, required=True, help="Directory containing the video files.")
    p.add_argument("--output", type=Path, required=True, help="Output predictions JSONL.")
    p.add_argument("--frame-dir", type=Path, default=None, help="Cache directory for extracted frames (default: <video-root>/central_frames).")
    p.add_argument("--duration-cache", type=Path, default=None, help="JSON file caching video durations.")
    p.add_argument("--model", type=str, default="gpt-4o-2024-08-06", help="OpenAI vision model.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--sleep", type=float, default=2.5, help="Base sleep (seconds) between retries.")
    p.add_argument("--max-samples", type=int, default=None, help="Limit the number of items (smoke test).")
    p.add_argument("--no-resume", action="store_false", dest="resume", help="Do not skip ids already in --output.")
    p.set_defaults(resume=True)
    return p.parse_args()


class FrameExtractor:
    """Extracts the central frame of a video with ffmpeg (longer side <= 384 px, no upscaling)."""

    SCALE_FILTER = "scale='if(gt(iw,ih), if(gt(iw,384),384, iw), -1)':'if(gt(iw,ih), -1, if(gt(ih,384),384, ih))'"

    def __init__(self, frame_dir: Path, duration_cache_path: Optional[Path] = None):
        self.frame_dir = frame_dir
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.duration_cache_path = duration_cache_path
        self.duration_cache: Dict[str, float] = {}
        if duration_cache_path and duration_cache_path.exists():
            try:
                self.duration_cache = json.load(open(duration_cache_path))
            except Exception:
                self.duration_cache = {}

    def _save_duration_cache(self):
        if self.duration_cache_path:
            try:
                json.dump(self.duration_cache, open(self.duration_cache_path, "w"))
            except Exception:
                pass

    def get_duration(self, video_path: Path) -> Optional[float]:
        key = str(video_path)
        if key in self.duration_cache:
            return self.duration_cache[key]
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", key]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return None
        try:
            dur = float(r.stdout.strip())
        except ValueError:
            return None
        self.duration_cache[key] = dur
        self._save_duration_cache()
        return dur

    def extract_central_frame(self, video_path: Path) -> Path:
        out_path = self.frame_dir / f"{video_path.stem}.jpg"
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        duration = self.get_duration(video_path)
        if not duration or duration <= 0:
            raise RuntimeError(f"Cannot obtain duration for {video_path}")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{duration / 2.0}", "-i", str(video_path), "-frames:v", "1", "-vf", self.SCALE_FILTER, "-q:v", "2", str(out_path)]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0 or not out_path.exists():
            raise RuntimeError(f"ffmpeg failed for {video_path}: {r.stderr.decode(errors='ignore')}")
        return out_path


def build_prompt(entry: Dict[str, Any]) -> str:
    """Question + options as given to the Video-LLMs, plus an instruction to answer with the letter only."""
    return entry["conversations"][0]["value"].replace("<image>\n", "") + "\nReturn ONLY the single letter (A-H) of the correct option."


def b64_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_letter(raw_text: str) -> Optional[str]:
    """First pass: a bare option letter (optionally wrapped in a small JSON object)."""
    if not raw_text:
        return None
    txt = raw_text.strip()
    if txt.startswith("{"):
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                for k in ("value", "answer", "choice", "prediction"):
                    if k in obj:
                        txt = str(obj[k]).strip()
                        break
        except Exception:
            pass
    m = LETTER_REGEX.search(txt)
    return m.group(1).upper() if m else None


def parse_verbose_response(response: str, choices: List[str]) -> Optional[str]:
    """Second pass for verbose answers: (A) / "A " / "A." patterns, then option-text matching (MMMU-style)."""
    if not response:
        return None
    resp = f" {response.strip()} "
    for ch in [",", ".", "!", "?", ";", ":", "'"]:
        resp = resp.strip(ch)
    letters = LETTERS[: len(choices)]

    candidates, bracket_style, index_ans = [], False, True
    for ch in letters:
        if f"({ch})" in resp:
            candidates.append(ch)
            bracket_style = True
    if not candidates:
        candidates = [ch for ch in letters if f"{ch} " in resp]
    if not candidates:
        candidates = [ch for ch in letters if f"{ch}." in resp]
    if not candidates and len(resp.split()) > 5:
        for ch in letters:
            text = str(choices[letters.index(ch)])
            if text and text.lower() in resp.lower():
                candidates.append(ch)
                index_ans = False
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # several candidates: take the last mention
    best, best_pos = None, -1
    for can in candidates:
        if index_ans:
            pos = resp.rfind(f"({can})") if bracket_style else resp.rfind(f" {can} ")
        else:
            pos = resp.lower().rfind(str(choices[letters.index(can)]).lower())
        if pos > best_pos:
            best, best_pos = can, pos
    return best


def load_released_split(dataset_id, attach_name, split="test"):
    """Load one of the released identifier-only splits and join the official annotations onto it."""
    import datasets

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "UniMambaMia"))
    import official_annotations

    return getattr(official_annotations, attach_name)(datasets.load_dataset(dataset_id)[split])


def main():
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if OpenAI is None or not api_key:
        sys.exit("The openai package and the OPENAI_API_KEY environment variable are required.")
    client = OpenAI(api_key=api_key)

    entries: List[Dict[str, Any]] = []
    if args.val_jsonl is not None:
        with open(args.val_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        source = str(args.val_jsonl)
    else:
        for row in load_released_split(args.dataset, "attach_avqa"):
            options = [row[k].strip() for k in "ABCD" if row.get(k) is not None]
            entries.append({
                "id": str(row["id"]),
                "video": row["video"],
                "conversations": [{"from": "human", "value": "<image>\n" + row["question"] + "\nOptions:\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))}],
                "answer": LETTERS.index(str(row["answer"]).strip().upper()),
                "raw_annotated_multi_choice": options,
            })
        source = args.dataset
    if args.max_samples:
        entries = entries[: args.max_samples]
    print(f"Loaded {len(entries)} items from {source}")

    extractor = FrameExtractor(args.frame_dir or args.video_root / "central_frames", args.duration_cache)

    done = set()
    if args.resume and args.output.exists():
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line).get("id"))
        print(f"Resuming: {len(done)} items already predicted")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.output, "a", encoding="utf-8") as wf:
        for entry in tqdm(entries, desc="GPT-4o single-frame probe"):
            entry_id = entry.get("id")
            if entry_id in done or not entry.get("video"):
                continue
            vpath = args.video_root / entry["video"]
            if not vpath.exists():
                continue
            try:
                frame_path = extractor.extract_central_frame(vpath)
                img_b64 = b64_image(frame_path)
            except Exception as ex:
                print(f"[warn] skip id={entry_id}: {ex}")
                continue

            raw_text, last_err = "", None
            for attempt in range(args.max_retries):
                try:
                    completion = client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": [
                            {"type": "text", "text": build_prompt(entry)},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        ]}],
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    raw_text = (completion.choices[0].message.content or "").strip()
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(args.sleep * (attempt + 1))
            else:
                print(f"[error] id={entry_id}: {last_err}")

            choices = entry.get("raw_annotated_multi_choice") or []
            pred_letter = extract_letter(raw_text) or parse_verbose_response(raw_text, choices)
            letter_space = LETTERS[: len(choices)] if choices else LETTERS
            pred_index = letter_space.index(pred_letter) if pred_letter in letter_space else None
            gt_idx = entry.get("answer")
            record = {
                "id": entry_id,
                "video": entry["video"],
                "video_duration": extractor.duration_cache.get(str(vpath)),
                "gt_answer_index": gt_idx,
                "gt_answer_letter": LETTERS[gt_idx] if isinstance(gt_idx, int) and 0 <= gt_idx < len(LETTERS) else None,
                "pred_letter": pred_letter,
                "pred_index": pred_index,
                "correct": bool(pred_index is not None and isinstance(gt_idx, int) and pred_index == gt_idx),
                "model_raw": raw_text,
            }
            wf.write(json.dumps(record, ensure_ascii=False) + "\n")
            wf.flush()
            written += 1
    print(f"Done. {written} new predictions written to {args.output}")


if __name__ == "__main__":
    main()
