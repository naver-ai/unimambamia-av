# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license

"""Attach the official AVQA / Music-AVQA annotations to the released evaluation splits.

The Hugging Face datasets released with this project (``gwkrsrch/avqa_2025``, ``gwkrsrch/avqa_hard``,
``gwkrsrch/music_avqa``, ``gwkrsrch/music_avqa_hard``) contain only item identifiers and the values we
produced (single-frame probe outputs, measured clip durations, split membership). The questions,
options and answers are not redistributed: this module downloads them from the official repositories
of the two benchmarks, checks their SHA-256, and joins them onto the released rows.

* AVQA (Yang et al., ACM MM 2022), https://mn.cs.tsinghua.edu.cn/avqa/ :
  ``val_qa.json`` of https://github.com/AlyssaYoung/AVQA, joined on the official ``id``.
* Music-AVQA (Li et al., CVPR 2022), https://gewu-lab.github.io/MUSIC-AVQA/ :
  ``data/json/avqa-test.json`` of https://github.com/GeWu-Lab/MUSIC-AVQA, joined on the position of
  the item in that file (``source_index``), because the file repeats some ``question_id`` values.

Row order is preserved, so the ``doc_id`` values of ``benchmark_audit/single_frame_filter.json`` keep
pointing at the same items.

The files are cached under ``$HF_HOME/unimambamia_av/official``. On machines without internet access,
download them once and point ``UNIMAMBAMIA_AVQA_ANNOTATIONS`` / ``UNIMAMBAMIA_MUSIC_AVQA_ANNOTATIONS``
at the local copies.
"""

import ast
import hashlib
import json
import os
import re
import tempfile
import urllib.request

import datasets

AVQA_SOURCE = {
    "name": "AVQA val_qa.json",
    "url": "https://raw.githubusercontent.com/AlyssaYoung/AVQA/5f5b87157e5265e85d88ae22a9bee03046f45fb6/data/annotation/val_qa.json",
    "sha256": "304acac38a254b4dc64b2e07b5202854ff22604d968585777f026010fc0be5bc",
    "env": "UNIMAMBAMIA_AVQA_ANNOTATIONS",
    "filename": "avqa_val_qa.json",
}

MUSIC_AVQA_SOURCE = {
    "name": "Music-AVQA avqa-test.json",
    "url": "https://raw.githubusercontent.com/GeWu-Lab/MUSIC-AVQA/f72f0e8a9bb897826aa0dda934fdece339d4e62f/data/json/avqa-test.json",
    "sha256": "fa79cbd8e7f0bf043e1761271b216f130832dd419a171bce9d9b561017af6cc2",
    "env": "UNIMAMBAMIA_MUSIC_AVQA_ANNOTATIONS",
    "filename": "music_avqa_test.json",
}

_OPTION_LETTERS = "ABCD"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_official_annotations(source):
    """Return the parsed official annotation file, downloading and verifying it if needed."""
    path = os.environ.get(source["env"])
    if not path:
        cache_dir = os.path.join(os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface")), "unimambamia_av", "official")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, source["filename"])
        if not os.path.exists(path) or _sha256(path) != source["sha256"]:
            # Several evaluation processes may start at once: download to a private file, then rename.
            fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".part")
            os.close(fd)
            try:
                urllib.request.urlretrieve(source["url"], tmp)
                os.chmod(tmp, 0o644)  # the cache may be shared between users
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

    if _sha256(path) != source["sha256"]:
        raise ValueError(
            f"{source['name']} at {path} does not match the expected SHA-256 {source['sha256']}. "
            f"Download it from {source['url']} (or set {source['env']} to an exact copy)."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extend(dataset, columns):
    for name, values in columns.items():
        if name in dataset.column_names:
            dataset = dataset.remove_columns(name)
        dataset = dataset.add_column(name, values)
    return dataset


def attach_avqa(dataset: datasets.Dataset) -> datasets.Dataset:
    """lmms-eval ``process_docs`` for avqa_2025 / avqa_hard: add question, options and answer by ``id``."""
    official = {str(item["id"]): item for item in fetch_official_annotations(AVQA_SOURCE)}
    rows = [official[str(i)] for i in dataset["id"]]
    return _extend(
        dataset,
        {
            "video": [r["video_name"] + ".mp4" for r in rows],
            "video_id": [str(r["video_id"]) for r in rows],
            "question": [r["question_text"] for r in rows],
            "options": [str(r["multi_choice"]) for r in rows],
            **{letter: [r["multi_choice"][k] for r in rows] for k, letter in enumerate(_OPTION_LETTERS)},
            "answer": [_OPTION_LETTERS[r["answer"]] for r in rows],
            "gt_answer_index": [str(r["answer"]) for r in rows],
            "gt_answer_letter": [_OPTION_LETTERS[r["answer"]] for r in rows],
            "question_relation": [r["question_relation"] for r in rows],
            "question_type": [r["question_type"] for r in rows],
        },
    )


def _fill_template(template, values):
    """Music-AVQA questions are templates such as "Is the <Object> ..."; fill the slots in order."""
    slots = iter(ast.literal_eval(values))
    return re.sub(r"<[^<>]+>", lambda m: next(slots, m.group(0)), template)


def attach_music_avqa(dataset: datasets.Dataset) -> datasets.Dataset:
    """lmms-eval ``process_docs`` for music_avqa / music_avqa_hard: add the item at ``source_index``."""
    official = fetch_official_annotations(MUSIC_AVQA_SOURCE)
    rows = [official[i] for i in dataset["source_index"]]
    return _extend(
        dataset,
        {
            "question_id": [str(r["question_id"]) for r in rows],
            "type": [r["type"] for r in rows],
            "templ_values": [r["templ_values"] for r in rows],
            "question_deleted": [str(r["question_deleted"]) for r in rows],
            "video": [r["video_id"] + ".mp4" for r in rows],
            "video_name": [r["video_id"] for r in rows],
            "question_content_raw": [r["question_content"] for r in rows],
            "question": [_fill_template(r["question_content"], r["templ_values"]) for r in rows],
            "answer": [r["anser"] for r in rows],  # the field is spelled "anser" in the official file
        },
    )
