#!/usr/bin/env python3
"""Runnable local inference demo for the Voyag V8 ONNX release.

The predictor auto-selects CUDA when a GPU runtime is available and falls back
to CPU otherwise; the effective device is printed at startup. Timing shown for
each call: the first call includes CUDA kernel warm-up, later calls are
steady-state.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import VoyagOnnxPredictor


def main() -> None:
    t0 = time.perf_counter()
    predictor = VoyagOnnxPredictor()
    print("device: %s  (load %.1fs)" % (predictor.device, time.perf_counter() - t0))
    print()
    examples = [
        "打开车窗，同时把空调调到24度",
        "读取消息，同时回电给张工",
        "放音乐，把导航打开",
    ]
    for utterance in examples:
        t0 = time.perf_counter()
        result = predictor.predict(utterance)
        print("utterance: %s" % utterance)
        print("elapsed_ms: %.1f" % ((time.perf_counter() - t0) * 1000.0))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    main()
