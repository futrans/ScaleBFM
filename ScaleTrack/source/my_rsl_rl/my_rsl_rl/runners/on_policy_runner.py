# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import os
import math
import signal
import statistics
import time
import torch
import warnings
from collections import deque
from datetime import datetime, timezone
from tensordict import TensorDict
from rich.progress import track
from copy import deepcopy

import my_rsl_rl
from my_rsl_rl.algorithms import PPO
from my_rsl_rl.env import VecEnv
from my_rsl_rl.modules import (
    ActorCritic, 
    ActorCriticHumanoidTransformer,
)
from my_rsl_rl.utils import resolve_obs_groups, store_code_state
from my_rsl_rl.utils.checkpoint_utils import (
    aggregate_evaluation_metrics,
    atomic_copy,
    atomic_json_save,
    atomic_torch_save,
    canonical_sha256,
    capture_rng_state,
    restore_rng_state,
    resume_config_sha256,
    sha256_file,
    validation_is_better,
    write_immutable_json,
)


class OnPolicyRunner:
    """On-policy runner for training and evaluation of actor-critic methods."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.original_cfg = deepcopy(train_cfg)
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Check if multi-GPU is enabled
        self._configure_multi_gpu()

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()
        default_sets = ["critic"]
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)

        # Create the algorithm
        self.alg = self._construct_algorithm(obs)

        # Decide whether to disable logging
        # Note: We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [my_rsl_rl.__file__]
        self.best_validation_metrics = None
        self.best_checkpoint_update = None
        self.latest_validation_report = None
        self.evaluation_breakdowns = {}
        self.source_by_split = self._load_provenance_sources()
        self.last_checkpoint_update = None
        self.stop_requested = False
        self.stop_signal = None
        self.permanent_checkpoint_interval = int(self.cfg.get("permanent_checkpoint_interval", self.save_interval))

        self.eval_during_training = self.cfg["eval_during_training"]
        self.eval_interval = self.cfg["eval_interval"]
        self.eval_metric_keys = self.cfg.get('eval_metric_keys', [])
        self.eval_max_steps = self.cfg.get('eval_max_steps', None)
        self.success_metric_dict = self.cfg.get('success_metric_dict', {})
        self.command_name = self.cfg.get('command_name', 'motion')
        self.success_discount_coef = self.cfg.get("success_discount_coef", 0.999)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Initialize writer
        self._prepare_logging_writer()

        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        start_tot_time = self.tot_time
        previous_signal_handlers = self._install_stop_signal_handlers()
        try:
            for it in range(start_iter + 1, tot_iter + 1):
                eval_dict = None
                validation_dict = None

                start = time.time()
                # Rollout
                with torch.inference_mode():
                    for _ in range(self.num_steps_per_env):
                        # Sample actions
                        actions = self.alg.act(obs)
                        # Step the environment
                        obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                        # Move to device
                        obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                        # Process the step
                        self.alg.process_env_step(obs, rewards, dones, extras)
                        # Book keeping
                        if self.log_dir is not None:
                            if "episode" in extras:
                                ep_infos.append(extras["episode"])
                            elif "log" in extras:
                                ep_infos.append(extras["log"])

                            cur_reward_sum += rewards
                            # Update episode length
                            cur_episode_length += 1
                            # Clear data for completed episodes
                            new_ids = (dones > 0).nonzero(as_tuple=False)
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                    stop = time.time()
                    collection_time = stop - start
                    start = stop

                    # Compute returns
                    self.alg.compute_returns(obs)

                # Update policy
                loss_dict = self.alg.update()

                stop = time.time()
                learn_time = stop - start
                self.current_learning_iteration = it

                validation_due = self.eval_during_training and it % self.eval_interval == 0
                if validation_due:
                    with torch.inference_mode():
                        eval_dict = self.evaluate_policy(split="train")
                        validation_dict = self.evaluate_policy(split="validation")
                        command = self.env.unwrapped.command_manager.get_term(self.command_name)
                        command.resample_motions()
                        command.randomize_next_resampling = True
                        obs, _ = self.env.reset()

                if self.log_dir is not None and not self.disable_logs:
                    self.log(locals())

                if validation_due:
                    self._save_validation_checkpoints(validation_dict)

                # Clear episode infos
                ep_infos.clear()
                # Save code state
                if it == start_iter + 1 and not self.disable_logs:
                    git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                    if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                        for path in git_file_paths:
                            self.writer.save_file(path)

                if self._stop_requested_across_ranks():
                    self._save_terminal_checkpoint("stopped")
                    break
            else:
                self._save_terminal_checkpoint("completed")
        finally:
            self._restore_stop_signal_handlers(previous_signal_handlers)

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:

        if locs.get('eval_dict') is not None:
            for key, value in locs['eval_dict'].items():
                self.writer.add_scalar(f"Eval/{key}", value, locs['it'])
        
        if locs.get('validation_dict') is not None:
            for key, value in locs['validation_dict'].items():
                self.writer.add_scalar(f"Validation/{key}", value, locs['it'])

        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # Log episode information
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # Handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # Log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # Log losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])

        self.writer.add_scalar("Loss/actor_learning_rate", self.alg.actor_learning_rate, locs["it"])
        self.writer.add_scalar("Loss/critic_learning_rate", self.alg.critic_learning_rate, locs["it"])

        # Log noise std
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # Log performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # Log training
        if len(locs["rewbuffer"]) > 0:
            # Everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            )
            # Print losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
            # Print rewards
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(locs["rewbuffer"]):.2f}\n"""
            # Print episode information
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(locs["lenbuffer"]):.2f}\n"""
        else:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""

        log_string += ep_string
        completed_in_run = locs["it"] - locs["start_iter"]
        remaining_in_run = locs["tot_iter"] - locs["it"]
        elapsed_in_run = self.tot_time - locs["start_tot_time"]
        eta_seconds = elapsed_in_run / completed_in_run * remaining_in_run
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} {
                time.strftime(
                    "%H:%M:%S",
                    time.gmtime(
                        eta_seconds
                    ),
                )
            }\n"""
        )
        print(log_string)

    def _install_stop_signal_handlers(self) -> dict[int, object]:
        previous_handlers = {}

        def request_stop(signum, _frame) -> None:
            self.stop_requested = True
            self.stop_signal = signal.Signals(signum).name
            print(f"[ScaleBFM checkpoint] {self.stop_signal} received; stopping after the current PPO update.")

        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, request_stop)
            except ValueError:
                return {}
        return previous_handlers

    @staticmethod
    def _restore_stop_signal_handlers(previous_handlers: dict[int, object]) -> None:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    def _stop_requested_across_ranks(self) -> bool:
        if not self.is_distributed:
            return self.stop_requested
        requested = torch.tensor(int(self.stop_requested), device=self.device)
        torch.distributed.all_reduce(requested, op=torch.distributed.ReduceOp.MAX)
        self.stop_requested = bool(requested.item())
        return self.stop_requested

    def _gather_rng_states(self) -> dict[str, dict]:
        local_state = capture_rng_state()
        if not self.is_distributed:
            return {"0": local_state}
        gathered = [None] * self.gpu_world_size
        torch.distributed.all_gather_object(gathered, local_state)
        return {str(rank): state for rank, state in enumerate(gathered)}

    def _local_four_arm_sampler_state(self) -> dict | None:
        command = self.env.unwrapped.command_manager.get_term(self.command_name)
        runtime = getattr(command, "four_arm", None)
        return runtime.state_dict() if runtime is not None else None

    def _gather_four_arm_sampler_states(self) -> dict[str, dict] | None:
        local_state = self._local_four_arm_sampler_state()
        if local_state is None:
            return None
        if not self.is_distributed:
            return {"0": local_state}
        gathered = [None] * self.gpu_world_size
        torch.distributed.all_gather_object(gathered, local_state)
        return {str(rank): state for rank, state in enumerate(gathered)}

    def _build_checkpoint_payload(
        self,
        rng_states: dict[str, dict],
        four_arm_sampler_states: dict[str, dict] | None,
        infos: dict | None = None,
    ) -> dict:
        command = self.env.unwrapped.command_manager.get_term(self.command_name)
        return {
            "schema_version": "scalebfm.scaletrack.checkpoint.v1",
            "model_state_dict": self.alg.policy.state_dict(),
            "actor_optimizer_state_dict": self.alg.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.alg.critic_optimizer.state_dict(),
            "algorithm_state": {
                "actor_learning_rate": self.alg.actor_learning_rate,
                "critic_learning_rate": self.alg.critic_learning_rate,
            },
            "completed_updates": self.current_learning_iteration,
            "iter": self.current_learning_iteration,
            "total_timesteps": self.tot_timesteps,
            "total_time": self.tot_time,
            "adaptive_motion_sampling_prob": command.motion_sampling_prob.detach().cpu(),
            "rng_states": rng_states,
            "four_arm_sampler_states": four_arm_sampler_states,
            "best_validation_metrics": self.best_validation_metrics,
            "best_checkpoint_update": self.best_checkpoint_update,
            "latest_validation_report": self.latest_validation_report,
            "last_checkpoint_update": self.last_checkpoint_update,
            "runner_config_sha256": canonical_sha256(self.original_cfg),
            "resume_config_sha256": resume_config_sha256(self.original_cfg),
            "run_metadata": self.original_cfg.get("run_metadata", {}),
            "infos": infos,
        }

    def _save_last_checkpoint(self, infos: dict | None = None) -> str | None:
        self.last_checkpoint_update = self.current_learning_iteration
        rng_states = self._gather_rng_states()
        four_arm_sampler_states = self._gather_four_arm_sampler_states()
        last_path = os.path.join(self.log_dir, "last.pt")
        if self.gpu_global_rank == 0:
            atomic_torch_save(
                self._build_checkpoint_payload(rng_states, four_arm_sampler_states, infos),
                last_path,
            )
        if self.is_distributed:
            torch.distributed.barrier()
        return last_path if self.gpu_global_rank == 0 else None

    def _save_validation_checkpoints(self, validation_metrics: dict[str, float]) -> None:
        report_path = os.path.join(
            self.log_dir,
            "validation",
            f"update_{self.current_learning_iteration:08d}.json",
        )
        became_best = False
        required = {"success_rate", "error_body_pos_g_success"}
        missing = sorted(required.difference(validation_metrics)) if self.gpu_global_rank == 0 else None
        if self.is_distributed:
            missing_payload = [missing]
            torch.distributed.broadcast_object_list(missing_payload, src=0)
            missing = missing_payload[0]
        if missing:
            raise ValueError(f"validation metrics missing checkpoint selection fields: {missing}")

        if self.gpu_global_rank == 0:
            became_best = validation_is_better(validation_metrics, self.best_validation_metrics)
            if became_best:
                self.best_validation_metrics = dict(validation_metrics)
                self.best_checkpoint_update = self.current_learning_iteration
            self.latest_validation_report = report_path

        last_path = self._save_last_checkpoint(
            {
                "validation_metrics": validation_metrics if self.gpu_global_rank == 0 else None,
                "validation_breakdown": self.evaluation_breakdowns.get("validation")
                if self.gpu_global_rank == 0
                else None,
                "validation_report": report_path,
            }
        )

        if self.gpu_global_rank != 0:
            return

        best_path = None
        if became_best:
            best_path = str(atomic_copy(last_path, os.path.join(self.log_dir, "best.pt")))

        archive_path = None
        if self.current_learning_iteration % self.permanent_checkpoint_interval == 0:
            archive_path = str(
                atomic_copy(
                    last_path,
                    os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"),
                    immutable=True,
                )
            )

        report = {
            "schema_version": "scalebfm.scaletrack.validation.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": os.path.basename(os.path.normpath(self.log_dir)),
            "completed_updates": self.current_learning_iteration,
            "total_timesteps": self.tot_timesteps,
            "metrics": validation_metrics,
            "stratified_metrics": self.evaluation_breakdowns.get("validation"),
            "selection_priority": ["success_rate:max", "error_body_pos_g_success:min"],
            "became_best": became_best,
            "best_checkpoint_update": self.best_checkpoint_update,
            "last_checkpoint_path": os.path.abspath(last_path),
            "last_checkpoint_sha256": sha256_file(last_path),
            "best_checkpoint_path": os.path.abspath(best_path) if best_path else None,
            "archive_checkpoint_path": os.path.abspath(archive_path) if archive_path else None,
            "runner_config_sha256": canonical_sha256(self.original_cfg),
            "run_metadata": self.original_cfg.get("run_metadata", {}),
        }
        write_immutable_json(report, report_path)
        print(
            f"[ScaleBFM checkpoint] update={self.current_learning_iteration} "
            f"last={last_path} best={became_best} archive={archive_path or '-'}"
        )

    def _save_terminal_checkpoint(self, reason: str) -> None:
        if self.log_dir is None:
            return
        if self.last_checkpoint_update != self.current_learning_iteration:
            last_path = self._save_last_checkpoint({"terminal_reason": reason})
        else:
            last_path = os.path.join(self.log_dir, "last.pt") if self.gpu_global_rank == 0 else None
            if self.is_distributed:
                torch.distributed.barrier()

        if self.gpu_global_rank != 0:
            return

        archive_path = os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")
        if reason == "completed" and os.path.exists(archive_path):
            terminal_path = archive_path
        else:
            terminal_path = os.path.join(
                self.log_dir,
                "checkpoints",
                f"{reason}_{self.current_learning_iteration:08d}.pt",
            )
            atomic_copy(last_path, terminal_path, immutable=True)

        if reason != "stopped":
            return

        record = {
            "schema_version": "scalebfm.scaletrack.stop_record.v1",
            "stopped_at": datetime.now(timezone.utc).isoformat(),
            "run_id": os.path.basename(os.path.normpath(self.log_dir)),
            "completed_updates": self.current_learning_iteration,
            "total_timesteps": self.tot_timesteps,
            "checkpoint_path": os.path.abspath(terminal_path),
            "checkpoint_sha256": sha256_file(terminal_path),
            "validation_report": self.latest_validation_report,
            "signal": self.stop_signal,
            "note": os.getenv("SCALEBFM_STOP_NOTE", ""),
            "runner_config_sha256": canonical_sha256(self.original_cfg),
            "run_metadata": self.original_cfg.get("run_metadata", {}),
        }
        immutable_record = os.path.join(
            self.log_dir,
            "checkpoints",
            f"stop_record_{self.current_learning_iteration:08d}.json",
        )
        write_immutable_json(record, immutable_record)
        atomic_json_save(record, os.path.join(self.log_dir, "stop_record.json"))
        print(f"[ScaleBFM checkpoint] stop record: {immutable_record}")

    def save(self, path: str, infos: dict | None = None) -> None:
        rng_states = {str(self.gpu_global_rank): capture_rng_state()}
        local_sampler_state = self._local_four_arm_sampler_state()
        four_arm_sampler_states = (
            {str(self.gpu_global_rank): local_sampler_state}
            if local_sampler_state is not None
            else None
        )
        atomic_torch_save(
            self._build_checkpoint_payload(rng_states, four_arm_sampler_states, infos),
            path,
        )

        # Upload model to external logging service
        # if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
        #     self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location or self.device)
        self._validate_resume_metadata(loaded_dict.get("run_metadata", {}))
        self._validate_resume_config(loaded_dict)
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.alg.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            algorithm_state = loaded_dict.get("algorithm_state", {})
            self.alg.actor_learning_rate = algorithm_state.get(
                "actor_learning_rate", self.alg.actor_optimizer.param_groups[0]["lr"]
            )
            self.alg.critic_learning_rate = algorithm_state.get(
                "critic_learning_rate", self.alg.critic_optimizer.param_groups[0]["lr"]
            )

        if "completed_updates" in loaded_dict:
            self.current_learning_iteration = int(loaded_dict["completed_updates"])
        else:
            self.current_learning_iteration = int(loaded_dict["iter"]) + 1
        self.tot_timesteps = int(loaded_dict.get("total_timesteps", 0))
        self.tot_time = float(loaded_dict.get("total_time", 0.0))
        self.best_validation_metrics = loaded_dict.get("best_validation_metrics")
        self.best_checkpoint_update = loaded_dict.get("best_checkpoint_update")
        self.latest_validation_report = loaded_dict.get("latest_validation_report")
        self.last_checkpoint_update = loaded_dict.get("last_checkpoint_update")

        command = self.env.unwrapped.command_manager.get_term(self.command_name)
        sampling_prob = loaded_dict.get("adaptive_motion_sampling_prob")
        if sampling_prob is not None:
            if command.motion_sampling_prob.shape != sampling_prob.shape:
                raise ValueError(
                    "checkpoint adaptive motion sampling shape does not match the current training manifest: "
                    f"{tuple(sampling_prob.shape)} != {tuple(command.motion_sampling_prob.shape)}"
                )
            command.motion_sampling_prob.copy_(sampling_prob)

        runtime = getattr(command, "four_arm", None)
        if runtime is not None:
            sampler_state = loaded_dict.get("four_arm_sampler_states", {}).get(str(self.gpu_global_rank))
            if sampler_state is None:
                raise ValueError("four-arm resume checkpoint is missing the local sampler state")
            runtime.load_state_dict(sampler_state)

        rank_state = loaded_dict.get("rng_states", {}).get(str(self.gpu_global_rank))
        if rank_state is not None:
            restore_rng_state(rank_state)
        return loaded_dict.get("infos")

    def load_shared_init(self, path: str, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location or self.device)
        shared_init = loaded_dict.get("four_arm_shared_init")
        if not isinstance(shared_init, dict):
            raise ValueError("shared-init checkpoint is missing four_arm_shared_init metadata")
        if shared_init.get("schema_version") != "maskmotion.m5.scalebfm_four_arm_shared_init.v1":
            raise ValueError("shared-init checkpoint has an unsupported four-arm schema")
        parity = shared_init.get("projection_parity", {})
        parity_atol = float(shared_init.get("projection_parity_atol", 1e-5))
        if not parity or any(
            float(item.get("max_abs_error", float("inf"))) > parity_atol
            for item in parity.values()
        ):
            raise ValueError("shared-init checkpoint did not preserve zero-input projection parity")
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
        self.alg.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
        algorithm_state = loaded_dict.get("algorithm_state", {})
        self.alg.actor_learning_rate = algorithm_state.get(
            "actor_learning_rate", self.alg.actor_optimizer.param_groups[0]["lr"]
        )
        self.alg.critic_learning_rate = algorithm_state.get(
            "critic_learning_rate", self.alg.critic_optimizer.param_groups[0]["lr"]
        )

        sampling_prob = loaded_dict.get("adaptive_motion_sampling_prob")
        if sampling_prob is not None:
            command = self.env.unwrapped.command_manager.get_term(self.command_name)
            if command.motion_sampling_prob.shape != sampling_prob.shape:
                raise ValueError(
                    "shared-init adaptive motion sampling shape does not match the training manifest: "
                    f"{tuple(sampling_prob.shape)} != {tuple(command.motion_sampling_prob.shape)}"
                )
            command.motion_sampling_prob.copy_(sampling_prob)

        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.best_validation_metrics = None
        self.best_checkpoint_update = None
        self.latest_validation_report = None
        self.last_checkpoint_update = None
        return loaded_dict.get("infos")

    def _validate_resume_metadata(self, saved_metadata: dict) -> None:
        current_metadata = self.original_cfg.get("run_metadata", {})
        if not saved_metadata or not current_metadata:
            return
        for field in ("distributed_world_size", "num_envs_per_rank"):
            if saved_metadata.get(field) != current_metadata.get(field):
                raise ValueError(
                    f"resume changes frozen resource field {field}: "
                    f"{saved_metadata.get(field)!r} != {current_metadata.get(field)!r}"
                )
        for split in ("train", "validation"):
            saved_manifest = saved_metadata.get("manifests", {}).get(split, {})
            current_manifest = current_metadata.get("manifests", {}).get(split, {})
            if saved_manifest.get("sha256") != current_manifest.get("sha256"):
                raise ValueError(f"resume changes frozen {split} manifest")
            if saved_manifest.get("provenance_sha256") != current_manifest.get("provenance_sha256"):
                raise ValueError(f"resume changes frozen {split} provenance")
        saved_four_arm = saved_metadata.get("four_arm", {})
        current_four_arm = current_metadata.get("four_arm", {})
        if saved_four_arm or current_four_arm.get("variant") not in {None, "off"}:
            for field in ("variant", "runtime", "schedule_sha256", "geometry_sha256"):
                if saved_four_arm.get(field) != current_four_arm.get(field):
                    raise ValueError(f"resume changes frozen four-arm field {field}")
            for field in ("shared_init_path", "shared_init_sha256"):
                if field in saved_four_arm and field not in current_four_arm:
                    current_four_arm[field] = saved_four_arm[field]

    def _validate_resume_config(self, loaded_dict: dict) -> None:
        saved_sha256 = loaded_dict.get("resume_config_sha256")
        if saved_sha256 is None:
            return
        current_sha256 = resume_config_sha256(self.original_cfg)
        if saved_sha256 == current_sha256:
            return

        config_with_saved_lineage = deepcopy(self.original_cfg)
        config_with_saved_lineage["run_metadata"] = loaded_dict.get("run_metadata", {})
        if saved_sha256 != resume_config_sha256(config_with_saved_lineage):
            raise ValueError("resume changes frozen training configuration")

        saved_commits = loaded_dict.get("run_metadata", {}).get("git_commits", {})
        current_commits = self.original_cfg.get("run_metadata", {}).get("git_commits", {})
        if saved_commits != current_commits:
            print(f"[ScaleBFM checkpoint] code lineage changed on resume: {saved_commits} -> {current_commits}")

    def _load_provenance_sources(self) -> dict[str, dict[str, str]]:
        source_by_split = {}
        manifests = self.original_cfg.get("run_metadata", {}).get("manifests", {})
        for split, manifest in manifests.items():
            provenance_path = manifest.get("provenance_path")
            if not provenance_path:
                source_by_split[split] = {}
                continue
            with open(provenance_path, encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            mapping = {str(row["clip_id"]): str(row["source"]) for row in rows}
            if len(mapping) != len(rows):
                raise ValueError(f"duplicate clip IDs in {split} provenance")
            source_by_split[split] = mapping
        return source_by_split

    def reset_environment_after_resume(self) -> None:
        command = self.env.unwrapped.command_manager.get_term(self.command_name)
        command.randomize_next_resampling = True
        command.resample_motions()
        self.env.reset()

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self) -> None:
        self.alg.policy.train()

    def eval_mode(self) -> None:
        self.alg.policy.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)

    def close(self) -> None:
        """Flush local events and finish the configured external logger."""
        if self.writer is None:
            return
        self.writer.flush()
        self.writer.close()
        self.writer = None

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        """Construct the actor-critic algorithm."""

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Initialize the policy
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic | ActorCriticHumanoidTransformer = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # Initialize the storage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg

    def _prepare_logging_writer(self) -> None:
        """Prepare the logging writers."""
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune or Tensorboard summary writer, default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from my_rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from my_rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

    def evaluate_policy(self, split="train"):
        if split not in {"train", "validation"}:
            raise ValueError(f"unsupported evaluation split: {split}")

        command = self.env.unwrapped.command_manager.get_term(self.command_name)
        is_train = split == "train"
        if split == "validation" and not command.has_validation_set:
            return {}

        self.eval_mode()

        self._set_env_is_evaluating(split)

        num_motions = command.num_motion
        motion_lengths = command.time_totals
        motion_range = torch.argsort(motion_lengths)
        
        if self.is_distributed:
            world_size = self.gpu_world_size
            rank = self.gpu_global_rank
            motion_range = motion_range[rank::world_size] # weishuai: This is to ensure balanced spread of motions of all lengths; Otherwise, some motions may be too long and torch.distributed.barriar may get timed out.
        num_motions_to_eval = len(motion_range)

        assert num_motions_to_eval > 0, f"There should be at least one motion to evaluate!"

        metrics = {}
        metrics["motion_ids"] = motion_range
        for k in self.eval_metric_keys:
            metrics[k] = torch.zeros(num_motions_to_eval)
            metrics[f"{k}_max"] = torch.zeros(num_motions_to_eval)
            # metrics[f"{k}_min"] = torch.zeros(num_motions)

        motion_map = []
        for i in range(0, len(motion_range), self.env.num_envs):
            batch_idx = torch.arange(i, min(i+self.env.num_envs, len(motion_range)))
            batch_motion_ids = motion_range[i: i+self.env.num_envs]
            motion_map.append((batch_idx, batch_motion_ids))
        num_iterations = len(motion_map)

        policy = self.get_inference_policy(device=self.device)

        for iter in track(
            range(num_iterations), total=num_iterations, description="Evaluting..."
        ):
            local_idx, motion_ids = motion_map[iter]
            num_motions_this_iter = len(motion_ids)
            
            command.motion_ids[:num_motions_this_iter] = motion_ids
            command.motion_ids[num_motions_this_iter:] = motion_ids[0]
            elapsed_time = torch.zeros_like(motion_ids)

            motion_lengths = command.time_totals[motion_ids]
            if self.eval_max_steps:
                motion_lengths = torch.clamp(motion_lengths, max=self.eval_max_steps*torch.ones_like(motion_lengths))
            max_length = motion_lengths.max().item()
            
            obs, extras = self.env.reset()

            for _ in track(range(max_length), total=max_length, transient=True):
                actions = policy(obs.to(self.device))
                obs, _, _, _ = self.env.step(actions.to(self.env.device))

                elapsed_time += 1

                clip_done = (elapsed_time >= motion_lengths).cpu()
                clip_not_done = torch.logical_not(clip_done)

                env_metric_dict = command.metrics
                for k in self.eval_metric_keys:
                    if k not in env_metric_dict:
                        raise ValueError(f"key {k} not found in command manager!")
                    value = env_metric_dict[k].cpu()

                    metric = value[:num_motions_this_iter]
                    metrics[k][local_idx[clip_not_done]] += metric[clip_not_done]
                    metrics[f"{k}_max"][local_idx[clip_not_done]] = torch.maximum(
                        metrics[f"{k}_max"][local_idx[clip_not_done]],
                        metric[clip_not_done]
                    )
                    # metrics[f"{k}_min"][motion_ids[clip_not_done]] = torch.minimum(
                    #     metrics[f"{k}_min"][motion_ids[clip_not_done]],
                    #     metric[clip_not_done]
                    # )
                    
        motion_lengths = command.time_totals[motion_range]
        if self.eval_max_steps:
            motion_lengths = torch.clamp(motion_lengths, max=self.eval_max_steps*torch.ones_like(motion_lengths))
        for k in self.eval_metric_keys:
            metrics[k] =  metrics[k] / motion_lengths

        if self.success_metric_dict:
            tracking_failures = torch.zeros(num_motions_to_eval, dtype=torch.bool)
            for k in self.success_metric_dict:
                if k in metrics:
                    tracking_failures = torch.logical_or(tracking_failures, metrics[f"{k}_max"] > self.success_metric_dict[k])
            tracking_failures = tracking_failures.float()
            failed_motions_index = torch.nonzero(tracking_failures).flatten().tolist()
            failed_motions_id = motion_range[failed_motions_index]
            motion_names = command.motion_names
            
            if self.is_distributed:
                fail_save_path = os.path.join(
                    self.log_dir,
                    f"failed_{split}_motions_rank_{self.gpu_global_rank}.txt",
                )
            else:
                fail_save_path = os.path.join(self.log_dir, f"failed_{split}_motions.txt")
            
            with open(fail_save_path, 'w') as f:
                for index in failed_motions_id:
                    f.write(f"{motion_names[index]}\n")

        if self.is_distributed:
            metric_path = os.path.join(self.log_dir, f"{split}_{self.gpu_global_rank}_metrics.pt")
            with open(metric_path, "wb") as f:
                torch.save(metrics, f)
            
            torch.distributed.barrier()

            metric_dict = {}

            if self.gpu_global_rank == 0:
                gathered_metrics = {k: torch.zeros(command.num_motion) for k in metrics}

                for rank in range(torch.distributed.get_world_size()):
                    rank_metric_path = os.path.join(self.log_dir, f"{split}_{rank}_metrics.pt")
                    with open(rank_metric_path, 'rb') as f:
                        other_metrics = torch.load(f, map_location="cpu")

                    for k in gathered_metrics:
                        if k == "motion_ids":
                            gathered_metrics[k][other_metrics["motion_ids"]] += 1
                        else:
                            gathered_metrics[k][other_metrics["motion_ids"]] = other_metrics[k]

                    os.unlink(rank_metric_path)

                metrics = gathered_metrics
                assert torch.all(metrics["motion_ids"] == 1.0).item(), (
                    f"{split} motions must be evaluated exactly once across all ranks"
                )

                if self.success_metric_dict:
                    example_key = list(self.success_metric_dict.keys())[0]
                    tracking_failures = torch.zeros_like(metrics[example_key], dtype=torch.bool)
                    for k in self.success_metric_dict:
                        if k in metrics:
                            tracking_failures = torch.logical_or(tracking_failures, metrics[f"{k}_max"] > self.success_metric_dict[k])
                    tracking_failures = tracking_failures.float()
                    metric_dict["success_rate"] = 1.0 - tracking_failures.detach().mean().item()

                    for k in self.eval_metric_keys:
                        mask = (tracking_failures == 0)
                        if mask.any():
                            result = metrics[k][mask].detach().mean().item()
                        else:
                            result = 0.0
                        metric_dict[f"{k}_success"] = result

                for k in self.eval_metric_keys:
                    metric_dict[k] = metrics[k].detach().mean().item()
            
            if is_train and self.success_metric_dict and getattr(command, "four_arm", None) is None:
                if self.gpu_global_rank == 0:
                    failed_idx = (tracking_failures == 1)
                    success_discount = math.pow(self.success_discount_coef, self.eval_interval)
                    new_sampling_prob = command.motion_sampling_prob.clone()
                    new_sampling_prob[failed_idx] /= success_discount
                    new_sampling_prob[~failed_idx] *= success_discount
                    new_sampling_prob.clamp_(min=0.03, max=1.0)
                    new_sampling_prob_cuda = new_sampling_prob.to(self.device)
                else:
                    new_sampling_prob_cuda = torch.zeros(
                        command.num_motion_train,
                        dtype=torch.float,
                        device=self.device,
                    )
                
                torch.distributed.broadcast(new_sampling_prob_cuda, src=0)

                new_sampling_prob = new_sampling_prob_cuda.detach().cpu()
                command.motion_sampling_prob[:] = new_sampling_prob

            torch.distributed.barrier()

        else:
            metric_dict = {}
            if self.success_metric_dict:
                tracking_failures = torch.zeros(num_motions, dtype=torch.bool)
                for k in self.success_metric_dict:
                    if k in metrics:
                        tracking_failures = torch.logical_or(tracking_failures, metrics[f"{k}_max"] > self.success_metric_dict[k])
                tracking_failures = tracking_failures.float()
                metric_dict["success_rate"] = 1.0 - tracking_failures.detach().mean().item()
                for k in self.eval_metric_keys:
                    mask = (tracking_failures == 0)
                    if mask.any():
                        result = metrics[k][mask].detach().mean().item()
                    else:
                        result = 0.0
                    metric_dict[f"{k}_success"] = result
                if is_train and getattr(command, "four_arm", None) is None:
                    failed_idx = (tracking_failures == 1)
                    success_discount = math.pow(self.success_discount_coef, self.eval_interval)
                    new_sampling_prob = command.motion_sampling_prob.clone()
                    new_sampling_prob[failed_idx] /= success_discount
                    new_sampling_prob[~failed_idx] *= success_discount
                    new_sampling_prob.clamp_(min=0.03, max=1.0)
                    command.motion_sampling_prob[:] = new_sampling_prob
            for k in self.eval_metric_keys:
                metric_dict[k] = metrics[k].detach().mean().item()

        if self.gpu_global_rank == 0:
            evaluated_motion_ids = torch.arange(command.num_motion) if self.is_distributed else motion_range
            motion_names = [command.motion_names[index] for index in evaluated_motion_ids.tolist()]
            mode_ids = None
            mode_names = []
            if command.cfg.mode_candidates:
                mode_ids = command.evaluation_mode_ids[evaluated_motion_ids]
                mode_names = list(command.cfg.mode_candidates)
            source_by_motion = self.source_by_split.get(split, {})
            if source_by_motion:
                missing_sources = sorted(set(motion_names).difference(source_by_motion))
                if missing_sources:
                    raise ValueError(
                        f"{split} provenance does not cover {len(missing_sources)} evaluated motions; "
                        f"first missing motion: {missing_sources[0]}"
                    )
            self.evaluation_breakdowns[split] = aggregate_evaluation_metrics(
                metrics,
                tracking_failures,
                motion_names,
                source_by_motion,
                mode_ids,
                mode_names,
            )
    
        self._set_env_no_evaluating(split)

        self.train_mode()

        return metric_dict
        
    def _set_env_is_evaluating(self, split="train"):
        command = self.env.unwrapped.command_manager.get_term(self.command_name)
        command.is_evaluating = True
        command.switch_motion_set(split)

        # Disable reset
        for key in self.env.unwrapped.termination_manager.active_terms:
            term_cfg = self.env.unwrapped.termination_manager.get_term_cfg(key)
            if hasattr(term_cfg, "params") and 'disable_flag' in term_cfg.params:
                term_cfg.params['disable_flag'] = True
            self.env.unwrapped.termination_manager.set_term_cfg(key, term_cfg)

        for key in self.env.unwrapped.event_manager.active_terms:
            if key in ['interval', 'reset']:
                for term_name in self.env.unwrapped.event_manager.active_terms[key]:
                    term_cfg = self.env.unwrapped.event_manager.get_term_cfg(term_name)
                    if hasattr(term_cfg, "params") and 'disable_flag' in term_cfg.params:
                        term_cfg.params['disable_flag'] = True
                    self.env.unwrapped.event_manager.set_term_cfg(term_name, term_cfg)
        
        self.obs_noise_cfg = {}

        for group_name in self.env.unwrapped.observation_manager.active_terms:
            obs_group_names = self.env.unwrapped.observation_manager._group_obs_term_names[group_name]
            obs_group_terms = self.env.unwrapped.observation_manager._group_obs_term_cfgs[group_name]
            self.obs_noise_cfg[group_name] = {}

            for i in range(len(obs_group_names)):
                name, term = obs_group_names[i], obs_group_terms[i]
                if hasattr(term, 'noise') and term.noise:
                    self.obs_noise_cfg[group_name][name] = deepcopy(term.noise)
                    self.env.unwrapped.observation_manager._group_obs_term_cfgs[group_name][i].noise = None
        
    def _set_env_no_evaluating(self, split="train"):
        command = self.env.unwrapped.command_manager.get_term(self.command_name)
        command.is_evaluating = False
        command.switch_motion_set("train")

        for key in self.env.unwrapped.termination_manager.active_terms:
            term_cfg = self.env.unwrapped.termination_manager.get_term_cfg(key)
            if hasattr(term_cfg, "params") and 'disable_flag' in term_cfg.params:
                term_cfg.params['disable_flag'] = False
            self.env.unwrapped.termination_manager.set_term_cfg(key, term_cfg)
        
        for key in self.env.unwrapped.event_manager.active_terms:
            if key in ['interval', 'reset']:
                for term_name in self.env.unwrapped.event_manager.active_terms[key]:
                    term_cfg = self.env.unwrapped.event_manager.get_term_cfg(term_name)
                    if hasattr(term_cfg, "params") and 'disable_flag' in term_cfg.params:
                        term_cfg.params['disable_flag'] = False
                    self.env.unwrapped.event_manager.set_term_cfg(term_name, term_cfg)

        for group_name in self.env.unwrapped.observation_manager.active_terms:
            obs_group_names = self.env.unwrapped.observation_manager._group_obs_term_names[group_name]
            # obs_group_terms = self.env.unwrapped.observation_manager._group_obs_term_cfgs[group_name]

            for i in range(len(obs_group_names)):
                name = obs_group_names[i]
                if name in self.obs_noise_cfg[group_name]:
                    self.env.unwrapped.observation_manager._group_obs_term_cfgs[group_name][i].noise = self.obs_noise_cfg[group_name][name]
        self.obs_noise_cfg = {}
