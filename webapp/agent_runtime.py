from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


CURRENT_DIR = Path(__file__).resolve().parent
# Support both layouts:
# 1) repo-root/webapp/ with repo-root/src/
# 2) standalone webapp folder next to a separate Pensieve project
PROJECT_ROOT = CURRENT_DIR.parent if (CURRENT_DIR.parent / "src").exists() else (
    CURRENT_DIR if (CURRENT_DIR / "frontend").exists() else CURRENT_DIR.parent
)
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import ppo2
except Exception:  # pragma: no cover
    ppo2 = None


PENSIEVE_VIDEO_BITRATE_KBPS = [300, 750, 1200, 1850, 2850, 4300]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
S_INFO = 6
S_LEN = 8
A_DIM = 6
SAFE_STEP_BUFFER_RESERVE = 5.0
SAFE_STEP_MIN_SAFETY_BUDGET = 2.0
SAFE_STEP_MAX_UPSTEP = 1

QUALITY_LADDER = [
    {"label": "360p", "bitrate_mbps": 0.30, "resolution": "640x360", "score": 0.7, "kbps": 300},
    {"label": "480p", "bitrate_mbps": 0.75, "resolution": "854x480", "score": 1.1, "kbps": 750},
    {"label": "720p", "bitrate_mbps": 1.20, "resolution": "1280x720", "score": 1.8, "kbps": 1200},
    {"label": "900p", "bitrate_mbps": 1.85, "resolution": "1600x900", "score": 2.6, "kbps": 1850},
    {"label": "1080p", "bitrate_mbps": 2.85, "resolution": "1920x1080", "score": 3.5, "kbps": 2850},
    {"label": "4K", "bitrate_mbps": 4.30, "resolution": "3840x2160", "score": 4.6, "kbps": 4300},
]

MODEL_EXTENSIONS = {".pth", ".pt"}
EPOCH_RE = re.compile(r"ep_(\d+)", re.IGNORECASE)


def clamp_quality(index: int) -> int:
    return max(0, min(index, len(QUALITY_LADDER) - 1))


def select_safe_quality(
    action_prob: np.ndarray,
    current_idx: int,
    buffer_size: float,
    next_chunk_sizes_mb: np.ndarray,
    throughput_mb_per_s: float,
) -> tuple[int, int, int]:
    raw_target_idx = int(np.argmax(action_prob))
    budget = max(buffer_size - SAFE_STEP_BUFFER_RESERVE, SAFE_STEP_MIN_SAFETY_BUDGET)

    feasible = [
        idx
        for idx, chunk_size_mb in enumerate(next_chunk_sizes_mb)
        if (chunk_size_mb / throughput_mb_per_s) <= budget
    ]
    safe_cap_idx = feasible[-1] if feasible else 0

    chosen_idx = min(raw_target_idx, safe_cap_idx)
    chosen_idx = min(chosen_idx, current_idx + SAFE_STEP_MAX_UPSTEP)
    return clamp_quality(chosen_idx), raw_target_idx, safe_cap_idx


def _scan_roots() -> list[Path]:
    roots: list[Path] = []
    base_candidates = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "ppo",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "pretrain",
    ]
    roots.extend(base_candidates)

    parent_dir = PROJECT_ROOT.parent
    if parent_dir.exists():
        for sibling in parent_dir.iterdir():
            if not sibling.is_dir() or sibling.resolve() == PROJECT_ROOT.resolve():
                continue
            if sibling.name.lower().startswith("pensieve"):
                roots.extend(
                    [
                        sibling / "src",
                        sibling / "ppo",
                        sibling / "models",
                        sibling / "pretrain",
                    ]
                )

    unique_roots: list[Path] = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_roots.append(root)
    return unique_roots


def serialize_model_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _iter_training_models() -> list[Path]:
    candidates: list[Path] = []
    for base_dir in _scan_roots():
        if not base_dir.exists():
            continue
        candidates.extend(
            path
            for path in base_dir.rglob("*")
            if path.is_file() and path.name.startswith("nn_model") and path.suffix in MODEL_EXTENSIONS
        )
    unique_candidates = {path.resolve(): path for path in candidates}
    return list(unique_candidates.values())


def _model_epoch(path: Path) -> int:
    match = EPOCH_RE.search(path.name)
    return int(match.group(1)) if match else -1


def _sort_key(path: Path) -> tuple[int, int, float]:
    path_str = str(path).lower()
    training_priority = 1 if "\\pretrain\\" in path_str or "/pretrain/" in path_str else 2
    return (training_priority, _model_epoch(path), path.stat().st_mtime)


def discover_model_paths(limit: int = 40) -> list[str]:
    candidates = _iter_training_models()
    candidates.sort(key=_sort_key, reverse=True)
    return [serialize_model_path(path) for path in candidates[:limit]]


def discover_last_trained_model() -> str | None:
    candidates = _iter_training_models()
    if not candidates:
        return None
    candidates.sort(key=_sort_key, reverse=True)
    return serialize_model_path(candidates[0])


def resolve_model_path(model_path: str | None, default_path: Path) -> Path:
    if not model_path:
        return default_path
    candidate = Path(model_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if candidate.exists():
        return candidate.resolve()

    candidate_name = candidate.name
    candidate_epoch = _model_epoch(candidate)
    alternative_names = {candidate_name}
    if candidate.suffix.lower() == ".pt":
        alternative_names.add(candidate.with_suffix(".pth").name)
    elif candidate.suffix.lower() == ".pth":
        alternative_names.add(candidate.with_suffix(".pt").name)

    all_candidates = _iter_training_models()
    name_matches = [path for path in all_candidates if path.name in alternative_names]
    if name_matches:
        name_matches.sort(key=_sort_key, reverse=True)
        return name_matches[0].resolve()

    if candidate_epoch >= 0:
        epoch_matches = [path for path in all_candidates if _model_epoch(path) == candidate_epoch]
        if epoch_matches:
            epoch_matches.sort(key=_sort_key, reverse=True)
            return epoch_matches[0].resolve()

    return candidate.resolve()


def checkpoint_summary(model_path: Path) -> tuple[bool, str]:
    if not model_path.exists():
        return False, "Model file not found."
    if torch is None:
        return False, "PyTorch is unavailable."
    try:
        payload = torch.load(model_path, map_location="cpu")
        if isinstance(payload, dict):
            return True, f"Loaded dict checkpoint with {len(payload.keys())} keys."
        return True, f"Loaded checkpoint object of type {type(payload).__name__}."
    except Exception as exc:  # pragma: no cover
        return False, f"Model load failed: {exc}"


class SimulatedRLAgent:
    controller_type = "simulated"
    controller_label = "Simulated PPO Rules"

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model_loaded, self.model_summary = checkpoint_summary(model_path)

    def recommend(
        self,
        network_speed: float,
        buffer_size: float,
        current_quality: str | None,
        session_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if network_speed > 5:
            idx = 4
        elif network_speed >= 2:
            idx = 2
        else:
            idx = 1

        if buffer_size < 3:
            idx -= 1
        elif buffer_size > 10:
            idx += 1

        current_idx = next(
            (i for i, quality in enumerate(QUALITY_LADDER) if quality["label"] == current_quality),
            idx,
        )
        idx = clamp_quality(idx)
        chosen = QUALITY_LADDER[idx]
        return {
            "quality": chosen["label"],
            "bitrate_mbps": chosen["bitrate_mbps"],
            "resolution": chosen["resolution"],
            "reason": self._explain(network_speed, buffer_size),
            "controller_type": self.controller_type,
            "controller_label": self.controller_label,
        }

    @staticmethod
    def _explain(network_speed: float, buffer_size: float) -> str:
        parts = []
        if network_speed > 5:
            parts.append("الشبكة قوية لذا تم الميل إلى جودة أعلى")
        elif network_speed >= 2:
            parts.append("الشبكة متوسطة لذا تم اختيار جودة متوازنة")
        else:
            parts.append("الشبكة ضعيفة لذا تم خفض الجودة")

        if buffer_size < 3:
            parts.append("المخزن المؤقت منخفض وتم إعطاء الأولوية للاستقرار")
        elif buffer_size > 10:
            parts.append("المخزن المؤقت مرتفع لذا يمكن رفع الجودة")
        return "، ".join(parts)


class RealPensieveAgent:
    controller_type = "real"
    controller_label = "Real Pensieve PPO"

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.available = torch is not None and ppo2 is not None and model_path.exists()
        self.model_loaded = False
        self.model_summary = "Pensieve runtime unavailable."
        self.network = None
        if self.available:
            try:
                self.network = ppo2.Network(state_dim=[S_INFO, S_LEN], action_dim=A_DIM, learning_rate=0.0001)
                self.network.load_model(str(model_path))
                self.model_loaded = True
                self.model_summary = "Pensieve PPO model loaded successfully with safe-step inference."
            except Exception as exc:  # pragma: no cover
                self.model_summary = f"Pensieve load failed: {exc}"

    def initial_session_state(self) -> dict[str, Any]:
        return {
            "state": np.zeros((S_INFO, S_LEN), dtype=np.float32),
            "chunk_index": 0,
            "last_action_idx": 1,
        }

    def recommend(
        self,
        network_speed: float,
        buffer_size: float,
        current_quality: str | None,
        session_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.model_loaded or self.network is None:
            raise RuntimeError(self.model_summary)

        if session_state is None:
            session_state = self.initial_session_state()

        state = np.array(session_state["state"], copy=True)
        state = np.roll(state, -1, axis=1)
        current_idx = next(
            (idx for idx, quality in enumerate(QUALITY_LADDER) if quality["label"] == current_quality),
            session_state.get("last_action_idx", 1),
        )

        throughput_mb_per_s = max(network_speed / 8.0, 0.01)
        current_bitrate = QUALITY_LADDER[current_idx]["bitrate_mbps"]
        chunk_size_mb = current_bitrate * 4.0 / 8.0
        estimated_download_time = max(chunk_size_mb / throughput_mb_per_s, 0.01)
        next_chunk_sizes_mb = np.array([quality["bitrate_mbps"] * 4.0 / 8.0 for quality in QUALITY_LADDER], dtype=np.float32)
        remaining = max(0, 48 - int(session_state.get("chunk_index", 0)))

        state[0, -1] = PENSIEVE_VIDEO_BITRATE_KBPS[current_idx] / float(np.max(PENSIEVE_VIDEO_BITRATE_KBPS))
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = throughput_mb_per_s
        state[3, -1] = estimated_download_time / BUFFER_NORM_FACTOR
        state[4, :A_DIM] = next_chunk_sizes_mb
        state[5, -1] = min(remaining, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP

        action_prob = self.network.predict(np.reshape(state, (1, S_INFO, S_LEN)))
        chosen_idx, raw_target_idx, safe_cap_idx = select_safe_quality(
            action_prob=action_prob,
            current_idx=current_idx,
            buffer_size=buffer_size,
            next_chunk_sizes_mb=next_chunk_sizes_mb,
            throughput_mb_per_s=throughput_mb_per_s,
        )
        session_state["state"] = state
        session_state["chunk_index"] = int(session_state.get("chunk_index", 0)) + 1
        session_state["last_action_idx"] = chosen_idx

        chosen = QUALITY_LADDER[chosen_idx]
        raw_target = QUALITY_LADDER[raw_target_idx]["label"]
        safe_cap = QUALITY_LADDER[safe_cap_idx]["label"]
        return {
            "quality": chosen["label"],
            "bitrate_mbps": chosen["bitrate_mbps"],
            "resolution": chosen["resolution"],
            "reason": f"قرار PPO مباشر من النموذج. احتمالية الإجراء الأعلى: {action_prob[chosen_idx]:.3f}",
            "controller_type": self.controller_type,
            "controller_label": self.controller_label,
            "action_probabilities": [round(float(item), 4) for item in action_prob.tolist()],
            "raw_target_quality": raw_target,
            "safe_cap_quality": safe_cap,
        }


class AgentRegistry:
    def __init__(self, default_model_path: Path) -> None:
        self.default_model_path = default_model_path
        self._cache: dict[tuple[str, str], Any] = {}

    def get_agent(self, controller_type: str, model_path: str | None) -> Any:
        resolved = resolve_model_path(model_path, self.default_model_path)
        cache_key = (controller_type, str(resolved))
        if cache_key in self._cache:
            return self._cache[cache_key]

        if controller_type == "real":
            agent = RealPensieveAgent(resolved)
        else:
            agent = SimulatedRLAgent(resolved)

        self._cache[cache_key] = agent
        return agent

    def bootstrap_payload(self) -> dict[str, Any]:
        default_agent = self.get_agent("simulated", None)
        return {
            "default_model_path": str(self.default_model_path.relative_to(PROJECT_ROOT)),
            "available_models": discover_model_paths(),
            "last_trained_model": discover_last_trained_model(),
            "available_controllers": [
                {"value": "simulated", "label": "Simulated PPO Rules"},
                {"value": "real", "label": "Real Pensieve PPO"},
            ],
            "agent": {
                "type": default_agent.controller_label,
                "model_path": str(default_agent.model_path.relative_to(PROJECT_ROOT)) if default_agent.model_path.exists() else None,
                "model_loaded": default_agent.model_loaded,
                "model_summary": default_agent.model_summary,
            },
        }

    def model_catalog_payload(self) -> dict[str, Any]:
        return {
            "default_model_path": str(self.default_model_path.relative_to(PROJECT_ROOT)),
            "available_models": discover_model_paths(),
            "last_trained_model": discover_last_trained_model(),
        }


@dataclass
class SessionMetrics:
    session_id: str
    match_id: str
    mode: str = "ai"
    controller_type: str = "simulated"
    controller_label: str = "Simulated PPO Rules"
    model_path: str = ""
    current_quality: str = "720p"
    decisions: list[dict[str, Any]] = field(default_factory=list)
    total_rebuffer_seconds: float = 0.0
    rebuffer_events: int = 0
    runtime_state: dict[str, Any] = field(default_factory=dict, repr=False)

    def log_decision(
        self,
        network_speed: float,
        buffer_size: float,
        applied_quality: str,
        ai_quality: str,
        rebuffer_seconds: float,
        qoe: float,
    ) -> None:
        if rebuffer_seconds > 0:
            self.total_rebuffer_seconds += rebuffer_seconds
            self.rebuffer_events += 1

        self.current_quality = applied_quality
        self.decisions.append(
            {
                "network_speed": network_speed,
                "buffer_size": buffer_size,
                "applied_quality": applied_quality,
                "ai_quality": ai_quality,
                "rebuffer_seconds": rebuffer_seconds,
                "qoe": qoe,
            }
        )

    def summary(self) -> dict[str, Any]:
        if not self.decisions:
            return {
                "session_id": self.session_id,
                "avg_quality": "N/A",
                "avg_bitrate_mbps": 0,
                "avg_qoe": 0,
                "final_qoe": 0,
                "total_rebuffer_seconds": 0,
                "rebuffer_events": 0,
                "decision_count": 0,
                "controller_type": self.controller_type,
                "controller_label": self.controller_label,
                "model_path": self.model_path,
            }

        quality_order = {item["label"]: idx for idx, item in enumerate(QUALITY_LADDER)}
        avg_quality_index = round(
            np.mean([quality_order[item["applied_quality"]] for item in self.decisions])
        )
        avg_bitrate = round(
            float(
                np.mean(
                    [next(q["bitrate_mbps"] for q in QUALITY_LADDER if q["label"] == item["applied_quality"]) for item in self.decisions]
                )
            ),
            2,
        )
        avg_qoe = round(float(np.mean([item["qoe"] for item in self.decisions])), 3)
        final_qoe = round(float(np.sum([item["qoe"] for item in self.decisions])), 3)

        return {
            "session_id": self.session_id,
            "avg_quality": QUALITY_LADDER[avg_quality_index]["label"],
            "avg_bitrate_mbps": avg_bitrate,
            "avg_qoe": avg_qoe,
            "final_qoe": final_qoe,
            "total_rebuffer_seconds": round(self.total_rebuffer_seconds, 2),
            "rebuffer_events": self.rebuffer_events,
            "decision_count": len(self.decisions),
            "controller_type": self.controller_type,
            "controller_label": self.controller_label,
            "model_path": self.model_path,
        }
