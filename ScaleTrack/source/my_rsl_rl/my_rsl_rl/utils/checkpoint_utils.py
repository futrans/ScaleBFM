from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def aggregate_evaluation_metrics(
    metrics: dict[str, torch.Tensor],
    tracking_failures: torch.Tensor,
    motion_names: list[str],
    source_by_motion: dict[str, str],
    mode_ids: torch.Tensor | None,
    mode_names: list[str],
) -> dict[str, object]:
    """Aggregate validation without hiding source or control-mode regressions."""
    failures = tracking_failures.detach().cpu().bool()
    metric_values = {
        key: value.detach().cpu()
        for key, value in metrics.items()
        if key != "motion_ids" and not key.endswith("_max")
    }
    if any(len(value) != len(motion_names) for value in metric_values.values()):
        raise ValueError("evaluation metrics and motion names have different lengths")
    if len(failures) != len(motion_names):
        raise ValueError("tracking failures and motion names have different lengths")

    sources = [source_by_motion.get(name, "unknown") for name in motion_names]
    if mode_ids is None:
        modes = ["unassigned"] * len(motion_names)
    else:
        raw_mode_ids = mode_ids.detach().cpu().tolist()
        if len(raw_mode_ids) != len(motion_names):
            raise ValueError("evaluation mode IDs and motion names have different lengths")
        modes = [mode_names[index] for index in raw_mode_ids]

    def aggregate(indices: list[int]) -> dict[str, float | int]:
        index = torch.tensor(indices, dtype=torch.long)
        group_failures = failures[index]
        passed = ~group_failures
        result: dict[str, float | int] = {
            "motions": len(indices),
            "passed": int(passed.sum().item()),
            "success_rate": float(passed.float().mean().item()),
        }
        for key, values in metric_values.items():
            selected = values[index]
            result[key] = float(selected.mean().item())
            result[f"{key}_success"] = float(selected[passed].mean().item()) if passed.any() else 0.0
        return result

    by_source = {
        source: aggregate([index for index, value in enumerate(sources) if value == source])
        for source in sorted(set(sources))
    }
    by_mode = {
        mode: aggregate([index for index, value in enumerate(modes) if value == mode])
        for mode in mode_names
        if mode in modes
    }
    if "unassigned" in modes:
        by_mode["unassigned"] = aggregate([index for index, value in enumerate(modes) if value == "unassigned"])
    by_source_mode = {
        source: {
            mode: aggregate(
                [
                    index
                    for index, (source_value, mode_value) in enumerate(zip(sources, modes, strict=True))
                    if source_value == source and mode_value == mode
                ]
            )
            for mode in sorted({mode for source_value, mode in zip(sources, modes, strict=True) if source_value == source})
        }
        for source in sorted(set(sources))
    }
    return {
        "overall": aggregate(list(range(len(motion_names)))),
        "by_source": by_source,
        "by_mode": by_mode,
        "by_source_mode": by_source_mode,
    }


def validation_is_better(candidate: dict[str, float], best: dict[str, float] | None) -> bool:
    """Compare validation metrics using the frozen ScaleBFM checkpoint ordering."""
    if best is None:
        return True
    candidate_key = (
        float(candidate["success_rate"]),
        -float(candidate["error_body_pos_g_success"]),
    )
    best_key = (
        float(best["success_rate"]),
        -float(best["error_body_pos_g_success"]),
    )
    return candidate_key > best_key


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_copy(source: str | Path, destination: str | Path, *, immutable: bool = False) -> Path:
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and destination_path.exists():
        raise FileExistsError(f"immutable checkpoint already exists: {destination_path}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source_path, temporary)
        if immutable:
            os.link(temporary, destination_path)
            temporary.unlink()
        else:
            os.replace(temporary, destination_path)
    finally:
        temporary.unlink(missing_ok=True)
    return destination_path


def write_immutable_json(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except FileExistsError:
        if destination.read_text(encoding="utf-8") != serialized:
            raise
    return destination


def atomic_json_save(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(serialized)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def resume_config_sha256(config: dict[str, Any]) -> str:
    """Hash training semantics while allowing resume location and target length to change."""
    operational_fields = {"load_checkpoint", "load_run", "max_iterations", "resume"}
    frozen = {key: value for key, value in config.items() if key not in operational_fields}
    return canonical_sha256(frozen)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])
