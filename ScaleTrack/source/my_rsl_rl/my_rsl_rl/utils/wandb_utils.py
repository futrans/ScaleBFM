# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.") from None


class WandbSummaryWriter(SummaryWriter):
    """Write TensorBoard events locally and mirror scalar metrics to W&B."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        super().__init__(log_dir=log_dir, flush_secs=flush_secs)

        project = cfg.get("wandb_project") or os.getenv("WANDB_PROJECT") or "maskmotion-scalebfm"
        entity = cfg.get("wandb_entity") or os.getenv("WANDB_ENTITY") or os.getenv("WANDB_USERNAME")
        run_name = cfg.get("wandb_run_name") or os.path.basename(os.path.normpath(log_dir))
        mode = cfg.get("wandb_mode") or os.getenv("WANDB_MODE") or "online"
        run_id = cfg.get("wandb_run_id") or os.getenv("WANDB_RUN_ID")

        init_kwargs = {
            "project": project,
            "entity": entity,
            "name": run_name,
            "group": cfg.get("wandb_group"),
            "tags": tuple(cfg.get("wandb_tags") or ()),
            "mode": mode,
            "dir": log_dir,
        }
        if run_id:
            init_kwargs.update({"id": run_id, "resume": "allow"})

        self._wandb_run = wandb.init(**init_kwargs)
        self._closed = False
        self._wandb_run.config.update({"log_dir": log_dir})

    def store_config(self, env_cfg: dict | object, runner_cfg: dict, alg_cfg: dict, policy_cfg: dict) -> None:
        self._wandb_run.config.update({"runner_cfg": runner_cfg})
        self._wandb_run.config.update({"policy_cfg": policy_cfg})
        self._wandb_run.config.update({"alg_cfg": alg_cfg})
        try:
            self._wandb_run.config.update({"env_cfg": env_cfg.to_dict()})
        except Exception:
            self._wandb_run.config.update({"env_cfg": asdict(env_cfg)})

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        self._wandb_run.log({tag: scalar_value}, step=global_step)

    def stop(self) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            super().close()
        finally:
            self._wandb_run.finish()

    def log_config(self, env_cfg: dict | object, runner_cfg: dict, alg_cfg: dict, policy_cfg: dict) -> None:
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path: str, iter: int) -> None:
        self._wandb_run.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        self._wandb_run.save(path, base_path=os.path.dirname(path))
