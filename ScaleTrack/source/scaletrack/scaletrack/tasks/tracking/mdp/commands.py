from __future__ import annotations
import yaml
import os
import torch
import pickle
import numpy as np
import time
from tqdm import tqdm
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING
from pathlib import Path
from easydict import EasyDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing.shared_memory import SharedMemory

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from scaletrack.tasks.tracking.four_arm import FourArmRuntime, robot_top_from_geometry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def load_motion_data_worker(args):
    idx, name, motion_path, body_indexes = args
    
    try:
        data = np.load(motion_path)
        if data["fps"] != 50:
            raise ValueError(f"Loaded motion should be 50FPS. An error occurs when loading motion: {name}")
        if data["joint_pos"].shape[0] == 0:
            raise ValueError(f"Loaded motion should not be zero duration. An error occurs when loading motion: {name}")
        
        return {
            "idx": idx,
            "name": name,
            "joint_pos": data["joint_pos"],
            "joint_vel": data["joint_vel"],
            "body_pos_w": data["body_pos_w"][:, body_indexes],
            "body_quat_w": data["body_quat_w"][:, body_indexes],
            "body_lin_vel_w": data["body_lin_vel_w"][:, body_indexes],
            "body_ang_vel_w": data["body_ang_vel_w"][:, body_indexes],
            "time_step": data["joint_pos"].shape[0]
        }
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None

def load_motions_np(motions: dict, body_indexes: Sequence[int]):

    motion_count = len(motions)
    motion_items = [(i, name, path, body_indexes) for i, (name, path) in enumerate(motions.items())]
    load_workers = min(int(os.getenv("SCALETRACK_MOTION_LOAD_WORKERS", "16")), motion_count)
    print(
        f"[ScaleTrack motion loader] phase=loading motions={motion_count} "
        f"workers={load_workers}",
        flush=True,
    )

    names = [None] * motion_count # weishuai: Pre-allocate to align index;
    joint_pos = [None] * motion_count
    joint_vel = [None] * motion_count
    body_pos_w = [None] * motion_count
    body_quat_w = [None] * motion_count
    body_lin_vel_w = [None] * motion_count
    body_ang_vel_w = [None] * motion_count
    time_step_total = [None] * motion_count
            
    motion_iterator = iter(motion_items)
    with ProcessPoolExecutor(max_workers=load_workers) as executor:
        pending = set()
        for _ in range(min(motion_count, load_workers * 4)):
            pending.add(executor.submit(load_motion_data_worker, next(motion_iterator)))
        with tqdm(total=motion_count, desc="Loading motions...") as progress:
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    result = future.result()
                    if result is None:
                        raise ValueError(
                            "Defective motions should be filtered from the dataset to ensure index alignment "
                            "under distributed training and parallel loading setting!"
                        )
                    idx = result["idx"]
                    names[idx] = result["name"]
                    joint_pos[idx] = result["joint_pos"]
                    joint_vel[idx] = result["joint_vel"]
                    body_pos_w[idx] = result["body_pos_w"]
                    body_quat_w[idx] = result["body_quat_w"]
                    body_lin_vel_w[idx] = result["body_lin_vel_w"]
                    body_ang_vel_w[idx] = result["body_ang_vel_w"]
                    time_step_total[idx] = result["time_step"]
                    progress.update(1)
                    try:
                        item = next(motion_iterator)
                    except StopIteration:
                        continue
                    pending.add(executor.submit(load_motion_data_worker, item))

    def concatenate_chunks(label, chunks):
        started_at = time.monotonic()
        print(f"[ScaleTrack motion loader] phase=concatenating field={label}", flush=True)
        concatenated = np.concatenate(chunks)
        chunks.clear()
        print(
            f"[ScaleTrack motion loader] phase=concatenated field={label} "
            f"shape={concatenated.shape} elapsed={time.monotonic() - started_at:.1f}s",
            flush=True,
        )
        return concatenated

    try:
        joint_pos_np = concatenate_chunks("joint_pos", joint_pos)
        joint_vel_np = concatenate_chunks("joint_vel", joint_vel)
        body_pos_w_np = concatenate_chunks("body_pos_w", body_pos_w)
        body_quat_w_np = concatenate_chunks("body_quat_w", body_quat_w)
        body_lin_vel_w_np = concatenate_chunks("body_lin_vel_w", body_lin_vel_w)
        body_ang_vel_w_np = concatenate_chunks("body_ang_vel_w", body_ang_vel_w)
        time_step_total_np = np.array(time_step_total)
    except Exception as e:
        print(f"Gathering operation failed possibly due to contaminated motion data. There is still None placeholder in the array!")
        raise e

    del joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, time_step_total

    return names, joint_pos_np, joint_vel_np, body_pos_w_np, body_quat_w_np, body_lin_vel_w_np, body_ang_vel_w_np, time_step_total_np


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0")) if self.is_distributed else 0
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        motion_file = self.cfg.motion_file
        print(f"Loading motions from {motion_file}")
        motion_file = Path(motion_file)
        
        if motion_file.suffix == ".yaml":
            with open(motion_file, 'r') as f:
                motions = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
            
            self.num_motion_train = len(motions)
            self.motion_names_train, self.cat_joint_pos_train, self.cat_joint_vel_train, \
                self.cat_body_pos_w_train, self.cat_body_quat_w_train, \
                    self.cat_body_lin_vel_w_train, self.cat_body_ang_vel_w_train, \
                        self.time_totals_train = self.load_motions(motions, self.body_indexes.cpu().numpy(), set_name="train")
            self.time_offsets_train = torch.zeros(self.num_motion_train, dtype=torch.long)
            if self.num_motion_train > 1:
                self.time_offsets_train[1:] = torch.cumsum(self.time_totals_train[:-1], dim=0)
        else:
            raise NotImplementedError
        
        validation_motion_file = self.cfg.validation_motion_file
        if validation_motion_file:
            self.has_validation_set = True
            print(f"Loading validation set motions from {validation_motion_file}")
            validation_motion_file = Path(validation_motion_file)
            
            if validation_motion_file.suffix == ".yaml":
                with open(validation_motion_file, 'r') as f:
                    validation_motions = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
                
                self.num_motion_validation = len(validation_motions)
                self.motion_names_validation, self.cat_joint_pos_validation, self.cat_joint_vel_validation, \
                    self.cat_body_pos_w_validation, self.cat_body_quat_w_validation, \
                        self.cat_body_lin_vel_w_validation, self.cat_body_ang_vel_w_validation, \
                            self.time_totals_validation = self.load_motions(
                                validation_motions,
                                self.body_indexes.cpu().numpy(),
                                set_name="validation",
                            )
                self.time_offsets_validation = torch.zeros(self.num_motion_validation, dtype=torch.long)
                if self.num_motion_validation > 1:
                    self.time_offsets_validation[1:] = torch.cumsum(self.time_totals_validation[:-1], dim=0)
            else:
                raise NotImplementedError
        else:
            self.has_validation_set = False
        
        self.cat_joint_pos = self.cat_joint_pos_train
        self.cat_joint_vel = self.cat_joint_vel_train
        self.cat_body_pos_w = self.cat_body_pos_w_train
        self.cat_body_quat_w = self.cat_body_quat_w_train
        self.cat_body_lin_vel_w = self.cat_body_lin_vel_w_train
        self.cat_body_ang_vel_w = self.cat_body_ang_vel_w_train
        self.time_totals = self.time_totals_train
        self.time_offsets = self.time_offsets_train      
        self.motion_names = self.motion_names_train
        self.num_motion = self.num_motion_train  

        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long)
        self.motion_sampling_prob = torch.ones(self.num_motion_train, dtype=torch.float)

        self.metrics["error_anchor_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos_g"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs,device=self.device)
        self.metrics['error_body_pos_relative'] = torch.zeros(self.num_envs,device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot_relative"] = torch.zeros(self.num_envs,device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["four_arm_conditioned"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["four_arm_condition_visible"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["four_arm_constraint_active"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["four_arm_condition_abs_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["four_arm_upper_bound_violation"] = torch.zeros(self.num_envs, device=self.device)

        self.is_evaluating = False
        self.randomize_next_resampling = True

        if self.cfg.mode_candidates:
            self._mode_table = torch.zeros((len(self.cfg.mode_candidates), len(self.cfg.body_names)), device=self.device, dtype=torch.float, requires_grad=False)
            for mode_idx, (mode_name, link_names) in enumerate(self.cfg.mode_candidates.items()):
                print(f"Mode Index: {mode_idx}; Mode Name: {mode_name}; Activated Links: {link_names}")
                for name in link_names:
                    assert name in self.cfg.body_names, f"Link {name} in Mode {mode_name} does not exist in the list of link names given."
                    self._mode_table[mode_idx, self.cfg.body_names.index(name)] = 1

            self.evaluation_mode_ids_train = self._balanced_evaluation_mode_ids(self.motion_names_train)
            if self.has_validation_set:
                self.evaluation_mode_ids_validation = self._balanced_evaluation_mode_ids(
                    self.motion_names_validation
                )
            self.evaluation_mode_ids = self.evaluation_mode_ids_train
        
            sampled_mode_ids = torch.randint(0, self._mode_table.shape[0], (self.num_envs,), device=self.device, dtype=torch.long)
            self._mode = self._mode_table[sampled_mode_ids].clone() # (num_envs, num_links)
        else:
            self._mode = torch.bernoulli(
                torch.ones(
                    self.num_envs, len(self.cfg.body_names),
                    dtype=torch.float32, device=self.device, requires_grad=False
                ) * 0.5
            )

        assert self.cfg.rand_timestep_range[1] <= self.cfg.ref_frame_buffer_size
        self._rand_timestep = torch.randint(
            low=self.cfg.rand_timestep_range[0], high=self.cfg.rand_timestep_range[1], size=(self.num_envs, 1), requires_grad=False
        )

        self._future_manual_cache: dict[tuple[int, ...], dict[str, torch.Tensor]] = {}
        self._four_arm_robot_top_cache: torch.Tensor | None = None

        self.four_arm: FourArmRuntime | None = None
        if self.cfg.four_arm_variant != "off":
            self.four_arm = FourArmRuntime(
                variant=self.cfg.four_arm_variant,
                schedule_path=self.cfg.four_arm_schedule_file,
                geometry_path=self.cfg.four_arm_geometry_file,
                motion_names=self.motion_names_train,
                robot_body_names=self.robot.body_names,
                num_envs=self.num_envs,
                device=self.device,
                conditioned_fraction=self.cfg.four_arm_conditioned_fraction,
                bones_seed_max_fraction=self.cfg.four_arm_bones_seed_max_fraction,
                sampler_seed=self.cfg.four_arm_sampler_seed,
                sampler_rank=self.cfg.four_arm_sampler_rank,
            )

        self.resample_motions()

    def resample_motions(self):
        self.motion_ids[:] = torch.multinomial(self.motion_sampling_prob, num_samples=self.num_envs, replacement=True)
        if self.four_arm is not None and not self.is_evaluating:
            self.four_arm.sample_conditioned(torch.arange(self.num_envs), self.motion_ids)
        
        sampling_probabilities = self.motion_sampling_prob / self.motion_sampling_prob.sum().clamp_min(1e-12)

        k = min(50, sampling_probabilities.shape[0])
        topk = torch.topk(sampling_probabilities, k, largest=True)
        print(f"Top {k} weights:")
        print("Indices:", topk.indices)
        print("Values:", topk.values)
        
    def _global_time_index(self):
        mids = self.motion_ids
        ts = torch.clamp(self.time_steps, min=torch.zeros_like(self.time_steps), max=self.time_totals[mids]-1)
        return self.time_offsets.gather(0, mids) + ts
    
    def _global_time_indices(self, frame_offsets: torch.Tensor) -> torch.Tensor:
        mids = self.motion_ids  # (num_envs,)
        ts = self.time_steps.unsqueeze(-1) + frame_offsets  # (num_envs, num_frames)
        motion_lengths = self.time_totals[mids].unsqueeze(-1)  # (num_envs, 1)
        ts = torch.clamp(ts, min=torch.zeros_like(motion_lengths), max=motion_lengths - 1)  # (num_envs, num_frames)
        time_offsets = self.time_offsets.gather(0, mids).unsqueeze(-1)  # (num_envs, 1)
        return time_offsets + ts  # (num_envs, num_frames)
    
    def _future_frame_offsets(self) -> torch.Tensor:
        frame_offset = torch.arange(self.cfg.ref_frame_buffer_size, dtype=torch.long).unsqueeze(0).repeat(self.num_envs, 1)
        frame_offset = torch.cat([frame_offset, self._rand_timestep], dim=-1)
        return frame_offset

    def _gather_cat_by_global_indices(self, cat_tensor: torch.Tensor, global_indices: torch.Tensor) -> torch.Tensor:
        flat = global_indices.reshape(-1)
        gathered = cat_tensor.index_select(0, flat)
        return gathered.view(*global_indices.shape, *cat_tensor.shape[1:])

    def _future_frame_offsets_manual(self, future_idx: Sequence[int]) -> torch.Tensor:
        idx = torch.as_tensor(future_idx, dtype=torch.long)  # CPU
        idx_expand = idx.unsqueeze(0).expand(self.num_envs, -1)
        if torch.any(idx < 0):
            rand_expand = self._rand_timestep.expand(-1, idx.shape[0])
            return torch.where(idx.unsqueeze(0) < 0, rand_expand, idx_expand)
        return idx_expand

    def _global_time_indices_manual(self, future_idx: Sequence[int]) -> torch.Tensor:
        frame_offsets = self._future_frame_offsets_manual(future_idx)
        return self._global_time_indices(frame_offsets)

    def body_pos_w_future_manual(self, future_idx: Sequence[int]) -> torch.Tensor:
        key = tuple(future_idx)
        cache = self._future_manual_cache.setdefault(key, {})
        if "body_pos_w_future" not in cache:
            global_indices = cache.get("global_indices")
            if global_indices is None:
                global_indices = self._global_time_indices_manual(future_idx)
                cache["global_indices"] = global_indices
                cache["frame_offsets"] = self._future_frame_offsets_manual(future_idx)
            body_pos_all = self._gather_cat_by_global_indices(self.cat_body_pos_w, global_indices).to(self.device)
            body_pos_all = body_pos_all + self._env.scene.env_origins[:, None, None, :]
            cache["body_pos_w_future"] = self._apply_four_arm_pelvis_target(
                body_pos_all,
                self.time_steps.unsqueeze(-1) + cache["frame_offsets"],
            )
        return cache["body_pos_w_future"]

    def body_quat_w_future_manual(self, future_idx: Sequence[int]) -> torch.Tensor:
        key = tuple(future_idx)
        cache = self._future_manual_cache.setdefault(key, {})
        if "body_quat_w_future" not in cache:
            global_indices = cache.get("global_indices")
            if global_indices is None:
                global_indices = self._global_time_indices_manual(future_idx)
                cache["global_indices"] = global_indices
                cache["frame_offsets"] = self._future_frame_offsets_manual(future_idx)
            body_quat_all = self._gather_cat_by_global_indices(self.cat_body_quat_w, global_indices).to(self.device)
            cache["body_quat_w_future"] = body_quat_all
        return cache["body_quat_w_future"]

    def body_lin_vel_w_future_manual(self, future_idx: Sequence[int]) -> torch.Tensor:
        key = tuple(future_idx)
        cache = self._future_manual_cache.setdefault(key, {})
        if "body_lin_vel_w_future" not in cache:
            global_indices = cache.get("global_indices")
            if global_indices is None:
                global_indices = self._global_time_indices_manual(future_idx)
                cache["global_indices"] = global_indices
                cache["frame_offsets"] = self._future_frame_offsets_manual(future_idx)
            body_lin_vel_all = self._gather_cat_by_global_indices(self.cat_body_lin_vel_w, global_indices).to(self.device)
            cache["body_lin_vel_w_future"] = body_lin_vel_all
        return cache["body_lin_vel_w_future"]

    def body_ang_vel_w_future_manual(self, future_idx: Sequence[int]) -> torch.Tensor:
        key = tuple(future_idx)
        cache = self._future_manual_cache.setdefault(key, {})
        if "body_ang_vel_w_future" not in cache:
            global_indices = cache.get("global_indices")
            if global_indices is None:
                global_indices = self._global_time_indices_manual(future_idx)
                cache["global_indices"] = global_indices
                cache["frame_offsets"] = self._future_frame_offsets_manual(future_idx)
            body_ang_vel_all = self._gather_cat_by_global_indices(self.cat_body_ang_vel_w, global_indices).to(self.device)
            cache["body_ang_vel_w_future"] = body_ang_vel_all
        return cache["body_ang_vel_w_future"]

    def time_offsets_future_manual(self, future_idx: Sequence[int]) -> torch.Tensor:
        key = tuple(future_idx)
        cache = self._future_manual_cache.setdefault(key, {})
        if "time_offsets_future" not in cache:
            frame_offsets = cache.get("frame_offsets")
            if frame_offsets is None:
                frame_offsets = self._future_frame_offsets_manual(future_idx)
                cache["frame_offsets"] = frame_offsets
            cache["time_offsets_future"] = frame_offsets.unsqueeze(-1).to(self.device)
        return cache["time_offsets_future"]

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return self.joint_pos

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.cat_joint_pos[self._global_time_index()].to(self.device) # weishuai: Only move to GPU when needed; The number of motions may be too large

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.cat_joint_vel[self._global_time_index()].to(self.device)

    @property
    def body_pos_w(self) -> torch.Tensor:
        body_pos = self.cat_body_pos_w[self._global_time_index()].to(self.device) + self._env.scene.env_origins[:, None, :]
        return self._apply_four_arm_pelvis_target(body_pos, self.time_steps)

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.cat_body_quat_w[self._global_time_index()].to(self.device)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.cat_body_lin_vel_w[self._global_time_index()].to(self.device)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.cat_body_ang_vel_w[self._global_time_index()].to(self.device)

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]

    @property
    def joint_pos_future(self) -> torch.Tensor:
        frame_offsets = self._future_frame_offsets()
        global_indices = self._global_time_indices(frame_offsets)  # (num_envs, num_frames)
        joint_pos_all = self.cat_joint_pos[global_indices].to(self.device)  # (num_envs, num_frames, num_joints)
        return joint_pos_all
    
    @property
    def joint_vel_future(self) -> torch.Tensor:
        frame_offsets = self._future_frame_offsets()
        global_indices = self._global_time_indices(frame_offsets)  # (num_envs, num_frames)
        joint_vel_all = self.cat_joint_vel[global_indices].to(self.device)  # (num_envs, num_frames, num_joints)
        return joint_vel_all

    @property
    def body_pos_w_future(self) -> torch.Tensor:
        frame_offsets = self._future_frame_offsets()
        global_indices = self._global_time_indices(frame_offsets)  # (num_envs, num_frames)
        body_pos_all = self.cat_body_pos_w[global_indices].to(self.device)  # (num_envs, num_frames, num_bodies, 3)
        body_pos_all = body_pos_all + self._env.scene.env_origins[:, None, None, :]  # Add env origins
        return body_pos_all
    
    @property
    def body_quat_w_future(self) -> torch.Tensor:
        frame_offsets = self._future_frame_offsets()
        global_indices = self._global_time_indices(frame_offsets)  # (num_envs, num_frames)
        body_quat_all = self.cat_body_quat_w[global_indices].to(self.device)  # (num_envs, num_frames, num_bodies, 4)
        return body_quat_all
    
    @property
    def body_lin_vel_w_future(self) -> torch.Tensor:
        frame_offsets = self._future_frame_offsets()
        global_indices = self._global_time_indices(frame_offsets)  # (num_envs, num_frames)
        body_lin_vel_all = self.cat_body_lin_vel_w[global_indices].to(self.device)  # (num_envs, num_frames, num_bodies, 3)
        return body_lin_vel_all
    
    @property
    def body_ang_vel_w_future(self) -> torch.Tensor:
        frame_offsets = self._future_frame_offsets()
        global_indices = self._global_time_indices(frame_offsets)  # (num_envs, num_frames)
        body_ang_vel_all = self.cat_body_ang_vel_w[global_indices].to(self.device)  # (num_envs, num_frames, num_bodies, 3)
        return body_ang_vel_all

    @property
    def anchor_pos_w_future(self) -> torch.Tensor:
        frame_offsets = self._future_frame_offsets()
        global_indices = self._global_time_indices(frame_offsets)  # (num_envs, num_frames)
        anchor_pos_all = self.cat_body_pos_w[global_indices, self.motion_anchor_body_index].to(self.device)  # (num_envs, num_frames, 3)
        anchor_pos_all = anchor_pos_all + self._env.scene.env_origins[:, None, :]  # Add env origins

        return anchor_pos_all

    @property
    def anchor_quat_w_future(self) -> torch.Tensor:
        frame_offsets = self._future_frame_offsets()
        global_indices = self._global_time_indices(frame_offsets)  # (num_envs, num_frames)
        anchor_quat_all = self.cat_body_quat_w[global_indices, self.motion_anchor_body_index].to(self.device)  # (num_envs, num_frames, 4)
        return anchor_quat_all
    
    @property
    def anchor_lin_vel_w_future(self) -> torch.Tensor:
        frame_offsets = self._future_frame_offsets()
        global_indices = self._global_time_indices(frame_offsets)  # (num_envs, num_frames)
        anchor_lin_vel_all = self.cat_body_lin_vel_w[global_indices, self.motion_anchor_body_index].to(self.device)  # (num_envs, num_frames, 3)
        return anchor_lin_vel_all
    
    @property
    def anchor_ang_vel_w_future(self) -> torch.Tensor:
        frame_offsets = self._future_frame_offsets()
        global_indices = self._global_time_indices(frame_offsets)  # (num_envs, num_frames)
        anchor_ang_vel_all = self.cat_body_ang_vel_w[global_indices, self.motion_anchor_body_index].to(self.device)  # (num_envs, num_frames, 3)
        return anchor_ang_vel_all
    
    @property
    def time_offsets_future(self) -> torch.Tensor:
        frame_offsets = self._future_frame_offsets()
        return frame_offsets.unsqueeze(-1).to(self.device)

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]
    
    @property
    def robot_anchor_lin_acc_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_acc_w[:, self.robot_anchor_body_index]
    
    @property
    def body_pos_relative_w(self) -> torch.Tensor:
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        return delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)
    
    @property
    def body_quat_relative_w(self) -> torch.Tensor:
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        return quat_mul(delta_ori_w, self.body_quat_w)

    @property
    def mode(self) -> torch.Tensor:
        return self._mode

    @property
    def four_arm_conditioned(self) -> torch.Tensor:
        if self.four_arm is None or self.is_evaluating:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self.four_arm.conditioned_mask

    @property
    def four_arm_condition_visible(self) -> torch.Tensor:
        if self.four_arm is None or self.is_evaluating:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self.four_arm.support(self.time_steps, device=self.device)[0]

    @property
    def four_arm_constraint_active(self) -> torch.Tensor:
        if self.four_arm is None or self.is_evaluating:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self.four_arm.support(self.time_steps, device=self.device)[1]

    @property
    def four_arm_height(self) -> torch.Tensor:
        if self.four_arm is None or self.is_evaluating:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        return self.four_arm.support(self.time_steps, device=self.device)[2]

    @property
    def four_arm_robot_top(self) -> torch.Tensor:
        if self.four_arm is None:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        if self._four_arm_robot_top_cache is None:
            body_pos = self.robot.data.body_pos_w.index_select(1, self.four_arm.body_ids)
            body_quat = self.robot.data.body_quat_w.index_select(1, self.four_arm.body_ids)
            self._four_arm_robot_top_cache = robot_top_from_geometry(
                body_pos, body_quat, self.four_arm.corners, self.four_arm.valid_corners
            )
        return self._four_arm_robot_top_cache

    def _apply_four_arm_pelvis_target(self, body_pos: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        if self.four_arm is None or self.is_evaluating or self.cfg.four_arm_variant not in {"b0", "b1"}:
            return body_pos
        _, active, height = self.four_arm.support(time_steps, device=self.device)
        if not bool(active.any()):
            return body_pos
        result = body_pos.clone()
        target = height - self.cfg.four_arm_pelvis_height_offset_m
        if result.ndim == 3:
            target = target + self._env.scene.env_origins[:, 2]
            result[:, self.motion_anchor_body_index, 2] = torch.where(
                active,
                target,
                result[:, self.motion_anchor_body_index, 2],
            )
        else:
            target = target + self._env.scene.env_origins[:, None, 2]
            result[:, :, self.motion_anchor_body_index, 2] = torch.where(
                active,
                target,
                result[:, :, self.motion_anchor_body_index, 2],
            )
        return result

    def _update_metrics(self):
        self._four_arm_robot_top_cache = None
        self.metrics["error_anchor_height"] = torch.abs(self.anchor_pos_w[..., 2] - self.robot_anchor_pos_w[..., 2])
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        self.metrics["error_body_pos_g"] = torch.norm(self.body_pos_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        ) # weishuai: g-mpkpe
        self.metrics["error_body_pos"] = torch.norm(
            (self.body_pos_w - self.body_pos_w[:, [self.motion_anchor_body_index]]) - (self.robot_body_pos_w - self.robot_body_pos_w[:, [self.robot_anchor_body_index]]), dim=-1
        ).mean(dim=-1) # weishuai: l-mpkpe
        self.metrics["error_body_pos_relative"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_w, self.robot_body_quat_w).mean(
            dim=-1
        )
        self.metrics["error_body_rot_relative"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)
        active = self.four_arm_constraint_active
        error = self.four_arm_robot_top - self.four_arm_height
        self.metrics["four_arm_conditioned"] = self.four_arm_conditioned.to(torch.float32)
        self.metrics["four_arm_condition_visible"] = self.four_arm_condition_visible.to(torch.float32)
        self.metrics["four_arm_constraint_active"] = active.to(torch.float32)
        self.metrics["four_arm_condition_abs_error"] = torch.where(active, torch.abs(error), torch.zeros_like(error))
        self.metrics["four_arm_upper_bound_violation"] = torch.where(
            active, torch.clamp_min(error, 0.0), torch.zeros_like(error)
        )

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        
        env_ids_cpu = env_ids.cpu()
        if self.four_arm is not None and not self.is_evaluating:
            self.four_arm.sample_conditioned(env_ids_cpu, self.motion_ids)
        if not self.is_evaluating:
            motion_len = self.time_totals.gather(0, self.motion_ids[env_ids_cpu])
            ordinary = torch.ones(len(env_ids_cpu), dtype=torch.bool)
            if self.four_arm is not None:
                conditioned = self.four_arm.conditioned_mask_cpu[env_ids_cpu]
                ordinary = ~conditioned
                self.time_steps[env_ids_cpu[conditioned]] = self.four_arm.visible_start_for(env_ids_cpu[conditioned])
            if bool(ordinary.any()):
                ordinary_ids = env_ids_cpu[ordinary]
                ordinary_len = motion_len[ordinary]
                if self.randomize_next_resampling:
                    phase = torch.rand(ordinary_len.shape)
                    self.time_steps[ordinary_ids] = (phase * (ordinary_len.float() - 1)).long()
                else:
                    self.time_steps[ordinary_ids] = (self.time_steps[ordinary_ids] + 1) % ordinary_len
            self.randomize_next_resampling = False
        else:
            self.time_steps[env_ids_cpu] = 0

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()
        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        if self.cfg.enable_reset_disturbance:
            range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
            root_pos[env_ids] += rand_samples[:, 0:3]
            orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
            root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
            range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
            root_lin_vel[env_ids] += rand_samples[:, :3]
            root_ang_vel[env_ids] += rand_samples[:, 3:]

            joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
            soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
            joint_pos[env_ids] = torch.clip(
                joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
            )
        
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

        if self.cfg.mode_candidates:
            if self.is_evaluating:
                sampled_mode_ids = self.evaluation_mode_ids[self.motion_ids[env_ids_cpu]].to(self.device)
            else:
                sampled_mode_ids = torch.randint(
                    0,
                    self._mode_table.shape[0],
                    (len(env_ids),),
                    device=self.device,
                    dtype=torch.long,
                )
            self._mode[env_ids] = self._mode_table[sampled_mode_ids].clone() # (env_ids_len, num_links)
            if self.four_arm is not None and not self.is_evaluating:
                conditioned = self.four_arm.conditioned_mask_cpu[env_ids_cpu].to(self.device)
                pelvis_mode_id = list(self.cfg.mode_candidates).index("Pelvis-1")
                self._mode[env_ids[conditioned]] = self._mode_table[pelvis_mode_id]
        else:
            self._mode[env_ids] = torch.bernoulli(
                torch.ones(
                    len(env_ids), len(self.cfg.body_names), 
                    dtype=torch.float32, device=self.device, requires_grad=False
                ) * 0.5
            )
        
        self._rand_timestep[env_ids_cpu] = torch.randint(
            low=self.cfg.rand_timestep_range[0], high=self.cfg.rand_timestep_range[1], size=(len(env_ids_cpu), 1), requires_grad=False
        )

        self._future_manual_cache.clear()
        self._four_arm_robot_top_cache = None

    def _update_command(self):
        self.time_steps += 1
        self._future_manual_cache.clear()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_body_visualizers"):
                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "goal_body_visualizers"):
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_w[:, i], self.body_quat_w[:, i])

    def _balanced_evaluation_mode_ids(self, motion_names: Sequence[str]) -> torch.Tensor:
        mode_ids = torch.empty(len(motion_names), dtype=torch.long)
        for position, motion_id in enumerate(sorted(range(len(motion_names)), key=motion_names.__getitem__)):
            mode_ids[motion_id] = position % self._mode_table.shape[0]
        return mode_ids

    def switch_motion_set(self, split: str):
        if split == "validation":
            if not self.has_validation_set:
                raise ValueError("validation motion set is not configured")
            self.motion_names = self.motion_names_validation
            self.num_motion = self.num_motion_validation
            self.time_totals = self.time_totals_validation
            self.time_offsets = self.time_offsets_validation
            
            self.cat_joint_pos = self.cat_joint_pos_validation
            self.cat_joint_vel = self.cat_joint_vel_validation
            self.cat_body_pos_w = self.cat_body_pos_w_validation
            self.cat_body_quat_w = self.cat_body_quat_w_validation
            self.cat_body_lin_vel_w = self.cat_body_lin_vel_w_validation
            self.cat_body_ang_vel_w = self.cat_body_ang_vel_w_validation
            if self.cfg.mode_candidates:
                self.evaluation_mode_ids = self.evaluation_mode_ids_validation

        elif split == "train":
            self.motion_names = self.motion_names_train
            self.num_motion = self.num_motion_train
            self.time_totals = self.time_totals_train
            self.time_offsets = self.time_offsets_train
            
            self.cat_joint_pos = self.cat_joint_pos_train
            self.cat_joint_vel = self.cat_joint_vel_train
            self.cat_body_pos_w = self.cat_body_pos_w_train
            self.cat_body_quat_w = self.cat_body_quat_w_train
            self.cat_body_lin_vel_w = self.cat_body_lin_vel_w_train
            self.cat_body_ang_vel_w = self.cat_body_ang_vel_w_train
            if self.cfg.mode_candidates:
                self.evaluation_mode_ids = self.evaluation_mode_ids_train
        else:
            raise ValueError(f"unsupported motion split: {split}")

    def load_motions(self, motions: dict, body_indexes: Sequence[int], set_name: str):

        if not self.is_distributed:
            
            names, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, time_step_total = load_motions_np(motions, body_indexes)
        
        else:
            if self.gpu_local_rank == 0:

                names, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, time_step_total = load_motions_np(motions, body_indexes)

                shapes = [
                    joint_pos.shape, joint_vel.shape,
                    body_pos_w.shape, body_quat_w.shape,
                    body_lin_vel_w.shape, body_ang_vel_w.shape,
                    time_step_total.shape
                ]
                dtypes = [
                    joint_pos.dtype, joint_vel.dtype,
                    body_pos_w.dtype, body_quat_w.dtype,
                    body_lin_vel_w.dtype, body_ang_vel_w.dtype,
                    time_step_total.dtype
                ]
                metadata = {
                    'shapes': shapes,
                    'dtypes': [str(dt) for dt in dtypes],  # Convert to string for serialization
                    'names': names
                }

                metadata_bytes = pickle.dumps(metadata)
                metadata_size = len(metadata_bytes)
                data_size = sum(np.prod(shape) * np.dtype(dtype).itemsize 
                                for shape, dtype in zip(shapes, dtypes))
                
                total_size = metadata_size + data_size + 8
                shared_memory_name = f"shared_motionlib_{set_name}"
                print(
                    f"[ScaleTrack motion loader] phase=allocating_shared_memory "
                    f"name={shared_memory_name} size_gib={total_size / 1024**3:.2f}",
                    flush=True,
                )

                try: 
                    shm = SharedMemory(name=shared_memory_name)
                except FileNotFoundError:
                    # Create new shared memory
                    shm = SharedMemory(name=shared_memory_name, create=True, size=total_size)
                    
                    shm.buf[:8] = metadata_size.to_bytes(8, 'little')
                    
                    shm.buf[8:8+metadata_size] = metadata_bytes
                    
                    # Write data arrays
                    offset = 8 + metadata_size
                    arrays = [joint_pos, joint_vel, body_pos_w, 
                                body_quat_w, body_lin_vel_w, body_ang_vel_w, time_step_total]
                    
                    for field, array in zip(
                        (
                            "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                            "body_lin_vel_w", "body_ang_vel_w", "time_step_total",
                        ),
                        arrays,
                    ):
                        started_at = time.monotonic()
                        print(
                            f"[ScaleTrack motion loader] phase=copying_shared_memory "
                            f"field={field} size_gib={array.nbytes / 1024**3:.2f}",
                            flush=True,
                        )
                        shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf, offset=offset)
                        np.copyto(shared_array, array, casting="no")
                        offset += array.nbytes
                        print(
                            f"[ScaleTrack motion loader] phase=copied_shared_memory field={field} "
                            f"elapsed={time.monotonic() - started_at:.1f}s",
                            flush=True,
                        )
                    
                    shm.close()

                torch.distributed.barrier()
            else:
                torch.distributed.barrier()

            setattr(self, f"shm_{set_name}", SharedMemory(name=f"shared_motionlib_{set_name}"))

            metadata_size = int.from_bytes(getattr(self, f"shm_{set_name}").buf[:8], 'little')
            metadata_bytes = bytes(getattr(self, f"shm_{set_name}").buf[8:8+metadata_size])
            metadata = pickle.loads(metadata_bytes)

            names = metadata['names']
            
            offset = 8 + metadata_size
            arrays = []
            
            for shape, dtype_str in zip(metadata['shapes'], metadata['dtypes']):
                dtype = np.dtype(dtype_str)
                array_size = np.prod(shape) * dtype.itemsize
                array = np.frombuffer(getattr(self, f"shm_{set_name}").buf, dtype=dtype, count=np.prod(shape), offset=offset).reshape(shape)
                arrays.append(array)
                offset += array_size
            
            joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, time_step_total = arrays

        joint_pos = torch.from_numpy(joint_pos)
        joint_vel = torch.from_numpy(joint_vel)
        body_pos_w = torch.from_numpy(body_pos_w)
        body_quat_w = torch.from_numpy(body_quat_w)
        body_lin_vel_w = torch.from_numpy(body_lin_vel_w)
        body_ang_vel_w = torch.from_numpy(body_ang_vel_w)
        time_step_total = torch.from_numpy(time_step_total)

        return names, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, time_step_total


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    validation_motion_file: str = ""
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    mode_candidates: dict[str, list[str]] = {}

    four_arm_variant: str = "off"
    four_arm_schedule_file: str = ""
    four_arm_geometry_file: str = ""
    four_arm_conditioned_fraction: float = 0.5
    four_arm_bones_seed_max_fraction: float = 0.6
    four_arm_pelvis_height_offset_m: float = 0.49
    four_arm_sampler_seed: int = 42
    four_arm_sampler_rank: int = 0

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    
    ref_frame_buffer_size: int = 33
    rand_timestep_range: tuple[int, int] = (5,33)

    enable_reset_disturbance: bool = True

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
