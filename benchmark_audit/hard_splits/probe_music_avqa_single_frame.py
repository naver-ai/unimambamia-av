#!/usr/bin/env python3
# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# The judge prompts are the ActivityNet-QA judge prompts of lmms-eval
# (https://github.com/EvolvingLMMs-Lab/lmms-eval, Apache-2.0), which the Music-AVQA task uses, so
# that the probe and the benchmark are judged identically.
"""
GPT-4o single-frame probe for Music-AVQA (open-ended), used to build Music-AVQA-Hard.

Music-AVQA answers are short free-form phrases, so the probe has two steps per item:
  1. GPT-4o answers the question from the temporally central frame only (longer side <= 384 px,
     no audio, no other frames), with the same post-prompt the Video-LLMs receive
     ("Answer the question using a single word or phrase.").
  2. A text-only judge (gpt-4o-mini) compares the prediction with the ground truth and returns
     ``{'pred': 'yes'|'no', 'score': 0-5}``, using the ActivityNet-QA judge prompt from lmms-eval.

Music-AVQA-Hard keeps the audio-related items the probe gets wrong (see ``build_hard_splits.py``).

Input: the ``test`` split of ``gwkrsrch/music_avqa`` (item positions ``source_index`` in the official
``avqa-test.json``; questions and answers are joined from the official Music-AVQA release by
``UniMambaMia/official_annotations.py``) and a directory with the video files (the lmms-eval cache
dir works).
Output: JSONL with ``id`` (= ``source_index``), ``video, pred, gt, judge_pred, judge_score`` per
item; re-running resumes.

Usage:
    export OPENAI_API_KEY=...
    python probe_music_avqa_single_frame.py --video-root $HF_HOME/music_avqa --output music_avqa_gpt4o_single_frame.jsonl
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

POST_PROMPT = "\nAnswer the question using a single word or phrase."

JUDGE_SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
    "------"
    "##INSTRUCTIONS: "
    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
    "- Consider synonyms or paraphrases as valid matches.\n"
    "- Evaluate the correctness of the prediction compared to the answer."
)
JUDGE_USER_PROMPT = (
    "Please evaluate the following video-based question-answer pair:\n\n"
    "Question: {question}\n"
    "Correct Answer: {answer}\n"
    "Predicted Answer: {pred}\n\n"
    "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
    "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
    "For example, your response should look like this: {{'pred': 'yes', 'score': 4}}."
)


def parse_args():
    p = argparse.ArgumentParser(description="Single-frame open-ended probe with GPT-4o + judge (Music-AVQA).")
    p.add_argument("--dataset", type=str, default="gwkrsrch/music_avqa", help="Released Music-AVQA split (HF dataset id).")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--video-root", type=Path, required=True, help="Directory containing the Music-AVQA video files.")
    p.add_argument("--output", type=Path, required=True, help="Output predictions JSONL.")
    p.add_argument("--frame-dir", type=Path, default=None, help="Cache directory for extracted frames (default: <video-root>/central_frames).")
    p.add_argument("--duration-cache", type=Path, default=None)
    p.add_argument("--model-vqa", type=str, default="gpt-4o-2024-08-06", help="Vision model answering the question.")
    p.add_argument("--model-judge", type=str, default="gpt-4o-mini", help="Text-only judge model.")
    p.add_argument("--temperature", type=float, default=0.0, help="Temperature of the vision model (the judge always uses 0).")
    p.add_argument("--max-tokens-vqa", type=int, default=32)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--sleep", type=float, default=2.0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--no-resume", action="store_false", dest="resume")
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

    def get_duration(self, video_path: Path) -> Optional[float]:
        key = str(video_path)
        if key in self.duration_cache:
            return self.duration_cache[key]
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", key]
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            dur = float(r.stdout.strip())
        except ValueError:
            return None
        self.duration_cache[key] = dur
        if self.duration_cache_path:
            try:
                json.dump(self.duration_cache, open(self.duration_cache_path, "w"))
            except Exception:
                pass
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
        if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg failed for {video_path}: {r.stderr.decode(errors='ignore')}")
        return out_path


def b64_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def with_retries(fn, max_retries: int, base_sleep: float):
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:  # network / rate limit
            last_err = e
            time.sleep(base_sleep * (attempt + 1))
    raise RuntimeError(f"API call failed after {max_retries} retries: {last_err}")


def call_vqa(client, model: str, question: str, img_b64: str, temperature: float, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def parse_judge(text: str) -> Dict[str, Any]:
    text = " ".join((text or "").split())
    obj: Any = {}
    for loader in (ast.literal_eval, json.loads):
        try:
            obj = loader(text)
            break
        except Exception:
            continue
    if not isinstance(obj, dict):
        obj = {}
    pred = str(obj.get("pred", "")).strip().lower()
    try:
        score = int(round(float(obj.get("score", 0))))
    except Exception:
        score = 0
    return {"pred": pred if pred in ("yes", "no") else "no", "score": max(0, min(5, score))}


def call_judge(client, model: str, question: str, answer: str, pred: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_USER_PROMPT.format(question=question, answer=answer, pred=pred)},
        ],
        temperature=0.0,
        max_tokens=16,
    )
    return parse_judge(resp.choices[0].message.content)


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

    entries = list(load_released_split(args.dataset, "attach_music_avqa", args.split))
    if args.max_samples:
        entries = entries[: args.max_samples]
    print(f"Loaded {len(entries)} items from {args.dataset}[{args.split}]")

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
        for entry in tqdm(entries, desc="GPT-4o single-frame probe + judge"):
            entry_id = entry["source_index"]
            if entry_id in done or not entry.get("video") or not entry.get("question"):
                continue
            vpath = Path(entry["video"])
            if not vpath.is_absolute():
                vpath = args.video_root / vpath
            if not vpath.exists():
                continue
            try:
                img_b64 = b64_image(extractor.extract_central_frame(vpath))
            except Exception as e:
                print(f"[warn] skip id={entry_id}: {e}")
                continue

            try:
                pred = with_retries(lambda: call_vqa(client, args.model_vqa, entry["question"].strip() + POST_PROMPT, img_b64, args.temperature, args.max_tokens_vqa), args.max_retries, args.sleep)
            except Exception as e:
                print(f"[warn] VQA failed (id={entry_id}): {e}")
                pred = ""
            try:
                judge = with_retries(lambda: call_judge(client, args.model_judge, entry["question"], entry["answer"], pred), args.max_retries, args.sleep)
            except Exception as e:
                print(f"[warn] judge failed (id={entry_id}): {e}")
                judge = {"pred": "no", "score": 0}

            record = {"id": entry_id, "video": entry["video"], "pred": pred, "gt": entry["answer"], "judge_pred": judge["pred"], "judge_score": judge["score"]}
            wf.write(json.dumps(record, ensure_ascii=False) + "\n")
            wf.flush()
            written += 1
    print(f"Done. {written} new predictions written to {args.output}")


if __name__ == "__main__":
    main()
