"""Local ONNX inference for Voyag V8.

This module is the runnable public runtime. It depends only on onnxruntime,
numpy, and tokenizers; it does not import PyTorch or any private architecture
source.
"""

from __future__ import annotations

import json
import os
import site as _site
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _register_nvidia_dll_dirs() -> None:
    """Put pip-installed NVIDIA CUDA wheels on the DLL search path (Windows).

    onnxruntime loads its CUDA provider through dlopen, which resolves
    dependencies via PATH; os.add_dll_directory alone is not enough. Register
    the nvidia/*/bin dirs of the pip wheels in-process so GPU inference works
    without the user editing PATH manually. Safe no-op on other platforms.
    """
    if sys.platform != "win32":
        return
    candidates: List[Path] = []
    for site_dir in _site.getsitepackages():
        candidates.append(Path(site_dir))
    user_site = _site.getusersitepackages()
    if user_site:
        candidates.append(Path(user_site))
    current_path = os.environ.get("PATH", "")
    added: List[str] = []
    for site_dir in candidates:
        try:
            matches = site_dir.glob("nvidia/*/bin")
        except OSError:
            continue
        for match in matches:
            directory = str(match)
            if directory not in current_path:
                added.append(directory)
    if not added:
        return
    os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + current_path
    for directory in added:
        try:
            os.add_dll_directory(directory)
        except OSError:
            pass


_register_nvidia_dll_dirs()

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent

MODEL_NAME = "voyag-v8-onnx"

PROVIDER_ALIASES = {
    "cuda": "CUDAExecutionProvider",
    "gpu": "CUDAExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def resolve_providers(requested: Optional[List[str]] = None) -> List[str]:
    """Return the execution provider list for the session.

    Resolution order: explicit argument > VOYAG_PROVIDER env var > auto
    (CUDA when available, otherwise CPU). CUDA always keeps CPU as fallback.
    """
    if requested:
        return list(requested)
    env_value = os.environ.get("VOYAG_PROVIDER", "").strip().lower()
    if env_value:
        alias = PROVIDER_ALIASES.get(env_value, env_value)
        available = ort.get_available_providers()
        if alias not in available:
            raise ValueError(
                "VOYAG_PROVIDER=%s requests provider %r which is not available; "
                "installed providers: %s" % (env_value, alias, available)
            )
        return [alias, "CPUExecutionProvider"] if alias != "CPUExecutionProvider" else [alias]
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

CANONICAL_STATE_FIELDS = [
    "utterance",
    "speed_kmh",
    "acc",
    "window_state",
    "hvac_state",
    "hvac_temperature_c",
    "media_state",
    "seat_heat",
    "light_state",
    "phone_call",
    "nav_active",
    "battery_soc",
    "gps_valid",
    "dtc",
    "warning",
]

NEUTRAL_STATE: Dict[str, Any] = {
    "utterance": "",
    "speed_kmh": 0,
    "acc": "off",
    "window_state": "closed",
    "hvac_state": "off",
    "hvac_temperature_c": 22,
    "media_state": "idle",
    "seat_heat": "off",
    "light_state": "off",
    "phone_call": "idle",
    "nav_active": False,
    "battery_soc": 0.70,
    "gps_valid": True,
    "dtc": [],
    "warning": [],
}


def canonicalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return state in the fixed field order expected by the model."""
    if "utterance" not in state:
        raise ValueError("state must contain 'utterance'")
    out: Dict[str, Any] = {}
    for key in CANONICAL_STATE_FIELDS:
        if key in state:
            out[key] = state[key]
    for key, value in state.items():
        if key not in out:
            out[key] = value
    return out


class VoyagOnnxPredictor:
    """Load the ONNX graph, tokenizer, and public schema once, then predict."""

    def __init__(
        self,
        onnx_path: Optional[Path | str] = None,
        schema_path: Optional[Path | str] = None,
        tokenizer_path: Optional[Path | str] = None,
        special_tokens_path: Optional[Path | str] = None,
        providers: Optional[List[str]] = None,
    ) -> None:
        self.onnx_path = Path(onnx_path or os.environ.get("VOYAG_ONNX") or ROOT / "onnx" / "voyag-v8.onnx")
        self.schema_path = Path(schema_path or ROOT / "schema" / "voyag_schema.json")
        self.tokenizer_path = Path(tokenizer_path or ROOT / "tokenizer" / "tokenizer.json")
        self.special_tokens_path = Path(
            special_tokens_path or ROOT / "tokenizer" / "special_tokens_map.json"
        )

        self.schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        self.slots = self.schema["slots"]
        self.relation_labels = self.schema["relation_labels"]
        self.max_len = int(self.schema["model_input"]["max_len"])
        self.head_max_len = int(self.schema["model_input"]["head_max_len"])

        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        special = json.loads(self.special_tokens_path.read_text(encoding="utf-8"))
        self.pad_id = self._special_id(special["pad_token"]["content"])
        self.cls_id = self._special_id(special["cls_token"]["content"])
        self.sep_id = self._special_id(special["sep_token"]["content"])
        self.mask_id = self._special_id(special["mask_token"]["content"])
        self.mask_token = special["mask_token"]["content"]

        if len(self.slots) != self.schema["model_input"]["num_slots"]:
            raise ValueError("public schema slot count does not match the ONNX graph")
        self.slot_ids = list(self.slots)
        self.pairs = [
            (i, j) for i in range(len(self.slot_ids)) for j in range(i + 1, len(self.slot_ids))
        ]

        # Slot question heads are identical for every request; encode them once.
        questions = self._questions()
        self._head_cache = [self._build_head(question) for question in questions]

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = int(os.environ.get("VOYAG_THREADS", "0"))
        sess_options.inter_op_num_threads = 1
        resolved = resolve_providers(providers)
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_options,
            providers=resolved,
        )
        active_providers = self.session.get_providers()
        if "CUDAExecutionProvider" in resolved and "CUDAExecutionProvider" not in active_providers:
            print(
                "warning: CUDAExecutionProvider requested but not active "
                "(active: %s); running on CPU" % active_providers
            )

    @property
    def device(self) -> str:
        """Name of the execution provider actually serving the session."""
        providers = self.session.get_providers()
        if not providers:
            return "unknown"
        first = providers[0]
        return {
            "CUDAExecutionProvider": "cuda",
            "CPUExecutionProvider": "cpu",
        }.get(first, first.replace("ExecutionProvider", "").lower())

    def _special_id(self, content: str) -> int:
        token_id = self.tokenizer.token_to_id(content)
        if token_id is None:
            raise ValueError(f"tokenizer is missing required special token: {content}")
        return int(token_id)

    def _questions(self) -> List[Dict[str, Any]]:
        questions = []
        for qid in self.slot_ids:
            slot = self.slots[qid]
            questions.append(
                {
                    "t": slot.get("type", "choice"),
                    "ins": slot["instructions"],
                    "crit": slot["criteria"],
                }
            )
        return questions

    def _encode(self, text: str, truncate: int | None = None) -> List[int]:
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        ids = encoding.ids
        return ids[:truncate] if truncate is not None else ids

    def _render_options(self, question: Dict[str, Any]) -> List[str]:
        options = []
        for label, description in question["crit"].items():
            rendered = str(label) if description in (None, "") else f"{label}: {description}"
            options.append(rendered)
        return options

    def _build_head(self, question: Dict[str, Any]) -> tuple[List[int], List[int]]:
        options = self._render_options(question)
        instruction = str(question["ins"]).replace(self.mask_token, " ")
        head_ids = self._encode(f"{question['t']} question: {instruction}")

        option_ids = []
        for option in options:
            tokens = self._encode(" " + option.replace(self.mask_token, " "), truncate=48)
            option_ids.append([self.mask_id] + tokens)

        option_budget = self.head_max_len - sum(len(tokens) for tokens in option_ids)
        if option_budget < 16:
            per_option = max(4, (self.head_max_len - 16) // max(1, len(option_ids)))
            option_ids = [tokens[:per_option] for tokens in option_ids]
            option_budget = self.head_max_len - sum(len(tokens) for tokens in option_ids)

        head_ids = head_ids[: max(8, option_budget)]
        ids = [self.cls_id] + head_ids + [self.sep_id]
        markers = []
        for tokens in option_ids:
            markers.append(len(ids))
            ids.extend(tokens)
        ids.append(self.sep_id)
        return ids, markers

    def _state_ids(self, state: Dict[str, Any]) -> List[int]:
        serialized = json.dumps(canonicalize_state(state), ensure_ascii=False)
        return self._encode(serialized.replace(self.mask_token, " "))

    def _build_inputs(self, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        state_ids = self._state_ids(state)
        rows = []
        for head_ids, head_markers in self._head_cache:
            head_len = len(head_ids)
            room = max(0, self.max_len - head_len - 1)
            state_part = state_ids[:room]
            ids = head_ids + state_part + [self.sep_id]
            ids = ids[: self.max_len]
            markers = [marker for marker in head_markers if marker < self.max_len]
            rows.append((ids, markers))

        max_markers = max(len(markers) for _, markers in rows)
        input_ids = np.full((len(rows), self.max_len), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(rows), self.max_len), dtype=np.int64)
        marker_pos = np.zeros((len(rows), max_markers), dtype=np.int64)
        marker_mask = np.zeros((len(rows), max_markers), dtype=np.bool_)
        qtype = np.zeros((len(rows),), dtype=np.int64)

        for row_idx, (ids, markers) in enumerate(rows):
            length = len(ids)
            input_ids[row_idx, :length] = ids
            attention_mask[row_idx, :length] = 1
            marker_pos[row_idx, : len(markers)] = markers
            marker_mask[row_idx, : len(markers)] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "marker_pos": marker_pos,
            "marker_mask": marker_mask,
            "qtype": qtype,
        }

    def predict_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state = canonicalize_state(dict(state))
        inputs = self._build_inputs(state)
        intent_probabilities, _, relation_probabilities, _ = self.session.run(None, inputs)

        active = {}
        intents: List[List[str]] = []
        for slot_index, qid in enumerate(self.slot_ids):
            option_index = int(np.argmax(intent_probabilities[slot_index]))
            option_labels = list(self.slots[qid]["criteria"])
            label = option_labels[option_index]
            if label == "none":
                continue
            active[qid] = label
            intents.append([qid, label])

        relations = []
        mutex_violations = []
        for pair_index, (left_index, right_index) in enumerate(self.pairs):
            left_qid = self.slot_ids[left_index]
            right_qid = self.slot_ids[right_index]
            if left_qid not in active or right_qid not in active:
                continue
            relation_index = int(np.argmax(relation_probabilities[pair_index]))
            relation_type = self.relation_labels[relation_index]
            relation = {
                "source_id": left_qid,
                "target_id": right_qid,
                "type": relation_type,
                "confidence": round(float(relation_probabilities[pair_index][relation_index]), 4),
            }
            relations.append(relation)
            if relation_type == "mutually_exclusive":
                mutex_violations.append(
                    {
                        "source_id": left_qid,
                        "target_id": right_qid,
                        "type": "mutually_exclusive",
                    }
                )

        if mutex_violations:
            execution_hint = "clarify"
        elif any(relation["type"] == "sequential" for relation in relations):
            execution_hint = "sequential"
        elif any(relation["type"] == "parallel" for relation in relations):
            execution_hint = "parallel"
        elif len(intents) > 1:
            execution_hint = "unknown"
        else:
            execution_hint = "single"

        return {
            "utterance": state.get("utterance", ""),
            "model": MODEL_NAME,
            "intents": intents,
            "relations": relations,
            "execution_hint": execution_hint,
            "mutex_violations": mutex_violations,
            "clarification": "检测到互斥意图，请确认先执行哪一个。" if mutex_violations else None,
        }

    def predict(self, utterance: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        merged = dict(NEUTRAL_STATE)
        if state is not None:
            merged.update(state)
        merged["utterance"] = utterance
        return self.predict_state(merged)

    def predict_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outputs = []
        for request in requests:
            utterance = request.get("utterance")
            state = request.get("state")
            if utterance is not None:
                outputs.append(self.predict(str(utterance), state))
            elif state is not None:
                outputs.append(self.predict_state(state))
            else:
                raise ValueError("request must contain 'utterance' or 'state'")
        return outputs
