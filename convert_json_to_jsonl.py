# MambaMia
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license

"""
Convert JSON to JSONL format for video training data.

Usage:
    python convert_json_to_jsonl.py input.json
    python convert_json_to_jsonl.py input.json output.jsonl
"""
import json
import sys
import os

def convert(json_path, jsonl_path=None):
    if jsonl_path is None:
        jsonl_path = os.path.splitext(json_path)[0] + ".jsonl"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Converted {len(data)} samples: {json_path} -> {jsonl_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_json_to_jsonl.py input.json [output.jsonl]")
        sys.exit(1)

    json_path = sys.argv[1]
    jsonl_path = sys.argv[2] if len(sys.argv) > 2 else None
    convert(json_path, jsonl_path)
