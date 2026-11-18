#!/usr/bin/env python3
"""Compare the TypeScript binding against the Python CPU runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UTTERANCES = [
    "打开车窗，同时把空调调到24度",
    "读取消息，同时回电给张工",
    "放音乐，把导航打开",
    "把座椅加热打开",
    "导航到最近充电站",
]


def python_result(utterance: str) -> dict:
    script = (
        "from inference import VoyagOnnxPredictor; "
        "import json; "
        "p=VoyagOnnxPredictor(providers=['CPUExecutionProvider']); "
        "print(json.dumps(p.predict(%r), ensure_ascii=False))" % utterance
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def ts_result(utterance: str) -> dict:
    proc = subprocess.run(
        ["node", "bindings/typescript/verify_single.js", utterance],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(proc.stdout.strip())


def comparable(value: dict) -> dict:
    value = dict(value)
    value["relations"] = [
        {
            "source_id": r["source_id"],
            "target_id": r["target_id"],
            "type": r["type"],
        }
        for r in value.get("relations", [])
    ]
    return value


def main() -> int:
    failed = False
    for utterance in UTTERANCES:
        py = python_result(utterance)
        ts = ts_result(utterance)
        ok = comparable(py) == comparable(ts)
        if ok:
            py_conf = {(r["source_id"], r["target_id"]): r["confidence"] for r in py["relations"]}
            ts_conf = {(r["source_id"], r["target_id"]): r["confidence"] for r in ts["relations"]}
            for key in py_conf:
                if abs(py_conf[key] - ts_conf[key]) > 1e-3:
                    ok = False
        print(("PASS" if ok else "FAIL") + ": " + utterance)
        if not ok:
            print("  python:", json.dumps(py, ensure_ascii=False))
            print("  ts:    ", json.dumps(ts, ensure_ascii=False))
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
