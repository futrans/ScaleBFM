from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch


FOUR_ARM_VARIANTS = ("off", "b0", "b1", "c_eq", "c_ub")
SCHEDULE_SCHEMA = "maskmotion.m5.scalebfm_four_arm_schedule.v1"
GEOMETRY_SCHEMA = "maskmotion.m5.scalebfm_four_arm_robot_top_geometry.v1"
ASSIGNMENT_ALGORITHM = "splitmix64(seed,rank,env_id,reset_ordinal)%pool_size"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_schedule_rows(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(row.get("schema_version") != SCHEDULE_SCHEMA for row in rows):
        raise ValueError(f"invalid four-arm schedule manifest: {path}")
    schedule_ids = [str(row["schedule_id"]) for row in rows]
    if len(schedule_ids) != len(set(schedule_ids)):
        raise ValueError("four-arm schedule manifest contains duplicate schedule IDs")
    return rows


def build_schedule_pools(
    rows: Sequence[Mapping[str, object]],
    motion_names: Sequence[str],
) -> tuple[dict[tuple[int, bool], list[int]], list[int]]:
    motion_id_by_name = {str(name): index for index, name in enumerate(motion_names)}
    if len(motion_id_by_name) != len(motion_names):
        raise ValueError("training motion names are not unique")
    pools: dict[tuple[int, bool], list[int]] = defaultdict(list)
    motion_ids = []
    for schedule_index, row in enumerate(rows):
        clip_id = str(row["clip_id"])
        if clip_id not in motion_id_by_name:
            raise ValueError(f"schedule clip is absent from the training manifest: {clip_id}")
        height_bin = int(row["height_bin"])
        if height_bin not in range(4):
            raise ValueError(f"schedule has invalid height bin: {height_bin}")
        is_bones_seed = str(row["source"]) == "bones_seed"
        pools[(height_bin, is_bones_seed)].append(schedule_index)
        motion_ids.append(motion_id_by_name[clip_id])
    for height_bin in range(4):
        for is_bones_seed in (False, True):
            if not pools[(height_bin, is_bones_seed)]:
                label = "bones_seed" if is_bones_seed else "non_bones_seed"
                raise ValueError(f"four-arm sampler has no {label} schedules in height bin {height_bin}")
    return dict(pools), motion_ids


def conditioned_slot_layout(
    num_envs: int,
    *,
    conditioned_fraction: float,
    bones_seed_max_fraction: float,
) -> list[tuple[int, bool]]:
    conditioned = int(round(num_envs * conditioned_fraction))
    if conditioned * 2 != num_envs or conditioned % 4:
        raise ValueError("four-arm training requires an even environment count with conditioned half divisible by four")
    per_height_bin = conditioned // 4
    bones_slots = int(per_height_bin * bones_seed_max_fraction)
    layout = []
    for height_bin in range(4):
        layout.extend([(height_bin, True)] * bones_slots)
        layout.extend([(height_bin, False)] * (per_height_bin - bones_slots))
    return layout


def deterministic_assignment_index(
    *, seed: int, rank: int, env_id: int, reset_ordinal: int, pool_size: int
) -> int:
    if pool_size < 1 or min(rank, env_id, reset_ordinal) < 0:
        raise ValueError("assignment tape indices and pool size must be valid")
    mask = (1 << 64) - 1
    value = (
        int(seed)
        + 0x9E3779B97F4A7C15 * (int(rank) + 1)
        + 0xD1B54A32D192ED03 * (int(env_id) + 1)
        + 0x94D049BB133111EB * (int(reset_ordinal) + 1)
    ) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return int(value % pool_size)


def load_robot_top_geometry(
    path: str | Path,
    robot_body_names: Sequence[str],
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != GEOMETRY_SCHEMA or payload.get("quaternion_order") != "wxyz":
        raise ValueError(f"invalid four-arm robot-top geometry: {path}")
    body_index_by_name = {name: index for index, name in enumerate(robot_body_names)}
    geometry_names = [str(name) for name in payload["body_names"]]
    if any(name not in body_index_by_name for name in geometry_names):
        raise ValueError("robot articulation does not match four-arm robot-top geometry")
    body_ids = torch.tensor([body_index_by_name[name] for name in geometry_names], dtype=torch.long, device=device)
    raw_corners = payload["visual_box_corners_by_body"]
    maximum_corners = max(len(values) for values in raw_corners)
    corners = torch.zeros((len(raw_corners), maximum_corners, 3), dtype=torch.float32, device=device)
    valid = torch.zeros((len(raw_corners), maximum_corners), dtype=torch.bool, device=device)
    for body_index, values in enumerate(raw_corners):
        if values:
            count = len(values)
            corners[body_index, :count] = torch.tensor(values, dtype=torch.float32, device=device)
            valid[body_index, :count] = True
    return body_ids, corners, valid


def _quat_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    qvec = quaternion[..., 1:]
    uv = torch.cross(qvec, vector, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return vector + 2.0 * (quaternion[..., :1] * uv + uuv)


def robot_top_from_geometry(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    corners_body: torch.Tensor,
    valid_corners: torch.Tensor,
) -> torch.Tensor:
    if body_pos_w.shape[:2] != body_quat_w.shape[:2] or body_quat_w.shape[-1] != 4:
        raise ValueError("body positions and quaternions are not aligned")
    expanded_quat = body_quat_w[:, :, None, :].expand(-1, -1, corners_body.shape[1], -1)
    expanded_corners = corners_body[None].expand(body_pos_w.shape[0], -1, -1, -1)
    world = _quat_apply_wxyz(expanded_quat, expanded_corners) + body_pos_w[:, :, None, :]
    top = world[..., 2].masked_fill(~valid_corners[None], -torch.inf)
    return top.amax(dim=(1, 2))


def mask_conditioned_task_features(
    values: torch.Tensor,
    conditioned: torch.Tensor,
    *,
    body_count: int,
    features_per_body: int,
    kept_body_features: Mapping[int, Sequence[int]],
) -> torch.Tensor:
    """Remove forbidden reference-pose features only for conditioned environments."""
    shaped = values.view(values.shape[0], values.shape[1], body_count, features_per_body)
    allowed = torch.zeros_like(shaped)
    for body_index, feature_indices in kept_body_features.items():
        allowed[:, :, body_index, list(feature_indices)] = shaped[:, :, body_index, list(feature_indices)]
    return torch.where(conditioned[:, None, None, None], allowed, shaped).flatten(start_dim=2)


def condition_observation(
    *,
    variant: str,
    visible: torch.Tensor,
    active: torch.Tensor,
    height: torch.Tensor,
    robot_top: torch.Tensor,
    future_count: int,
) -> torch.Tensor:
    """Encode (visible height, active flag, current robot top) for each task token."""
    if variant not in FOUR_ARM_VARIANTS[1:]:
        raise ValueError(f"invalid active four-arm variant: {variant}")
    features = torch.zeros((height.shape[0], future_count, 3), dtype=height.dtype, device=height.device)
    if variant in {"c_eq", "c_ub"}:
        features[..., 0] = torch.where(visible, height, torch.zeros_like(height))[:, None]
        features[..., 1] = active.to(height.dtype)[:, None]
        features[..., 2] = robot_top[:, None]
    return features


def specialized_tracking_mask(
    *, variant: str, conditioned: torch.Tensor, active: torch.Tensor
) -> torch.Tensor:
    """Select frames where the arm-specific tracking objective replaces the baseline."""
    if variant == "b1":
        return active
    if variant in {"c_eq", "c_ub"}:
        return conditioned
    return torch.zeros_like(conditioned)


def condition_reward_value(
    *,
    variant: str,
    robot_top: torch.Tensor,
    height: torch.Tensor,
    active: torch.Tensor,
    std: float,
) -> torch.Tensor:
    error = robot_top - height
    if variant == "c_ub":
        error = torch.clamp_min(error, 0.0)
    elif variant != "c_eq":
        raise ValueError("condition reward is defined only for c_eq and c_ub")
    reward = torch.exp(-torch.square(error) / std**2)
    return torch.where(active, reward, torch.zeros_like(reward))


def heading_tan_norm_wxyz(target_quat: torch.Tensor, reference_quat: torch.Tensor) -> torch.Tensor:
    """Return a six-dimensional relative-yaw representation with roll and pitch removed."""
    def heading(quaternion: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quaternion.unbind(dim=-1)
        return torch.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))

    angle = heading(target_quat) - heading(reference_quat)
    tangent = torch.stack((torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)), dim=-1)
    normal = torch.zeros_like(tangent)
    normal[..., 2] = 1.0
    return torch.cat((tangent, normal), dim=-1)


class FourArmRuntime:
    def __init__(
        self,
        *,
        variant: str,
        schedule_path: str | Path,
        geometry_path: str | Path,
        motion_names: Sequence[str],
        robot_body_names: Sequence[str],
        num_envs: int,
        device: torch.device | str,
        conditioned_fraction: float,
        bones_seed_max_fraction: float,
        sampler_seed: int,
        sampler_rank: int,
    ) -> None:
        if variant not in FOUR_ARM_VARIANTS[1:]:
            raise ValueError(f"invalid active four-arm variant: {variant}")
        self.variant = variant
        self.rows = load_schedule_rows(schedule_path)
        self.pools, schedule_motion_ids = build_schedule_pools(self.rows, motion_names)
        self.schedule_motion_ids = torch.tensor(schedule_motion_ids, dtype=torch.long)
        self.visible_start = torch.tensor(
            [int(row["condition_visible"]["start_frame"]) for row in self.rows], dtype=torch.long
        )
        self.visible_end = torch.tensor(
            [int(row["condition_visible"]["end_frame_exclusive"]) for row in self.rows], dtype=torch.long
        )
        self.active_start = torch.tensor(
            [int(row["constraint_active"]["start_frame"]) for row in self.rows], dtype=torch.long
        )
        self.active_end = torch.tensor(
            [int(row["constraint_active"]["end_frame_exclusive"]) for row in self.rows], dtype=torch.long
        )
        self.heights = torch.tensor([float(row["h_m"]) for row in self.rows], dtype=torch.float32)
        self.slot_layout = conditioned_slot_layout(
            num_envs,
            conditioned_fraction=conditioned_fraction,
            bones_seed_max_fraction=bones_seed_max_fraction,
        )
        self.conditioned_count = len(self.slot_layout)
        self.schedule_ids = torch.full((num_envs,), -1, dtype=torch.long)
        self.conditioned_mask_cpu = torch.arange(num_envs) < self.conditioned_count
        self.conditioned_mask = self.conditioned_mask_cpu.to(device)
        self.sampler_seed = int(sampler_seed)
        self.sampler_rank = int(sampler_rank)
        self.reset_ordinals = torch.zeros(num_envs, dtype=torch.long)
        self.body_ids, self.corners, self.valid_corners = load_robot_top_geometry(
            geometry_path,
            robot_body_names,
            device=device,
        )
        self.schedule_sha256 = sha256_file(schedule_path)
        self.geometry_sha256 = sha256_file(geometry_path)

    def sample_conditioned(self, env_ids: torch.Tensor, motion_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(device="cpu", dtype=torch.long)
        for env_id in env_ids[self.conditioned_mask_cpu[env_ids]]:
            env_index = int(env_id)
            height_bin, is_bones_seed = self.slot_layout[env_index]
            pool = self.pools[(height_bin, is_bones_seed)]
            reset_ordinal = int(self.reset_ordinals[env_index])
            pool_index = deterministic_assignment_index(
                seed=self.sampler_seed,
                rank=self.sampler_rank,
                env_id=env_index,
                reset_ordinal=reset_ordinal,
                pool_size=len(pool),
            )
            schedule_id = pool[pool_index]
            self.schedule_ids[env_id] = schedule_id
            motion_ids[env_id] = self.schedule_motion_ids[schedule_id]
            self.reset_ordinals[env_index] += 1

    def support(self, time_steps: torch.Tensor, *, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        safe_ids = self.schedule_ids.clamp_min(0)
        visible_start = self.visible_start[safe_ids]
        visible_end = self.visible_end[safe_ids]
        active_start = self.active_start[safe_ids]
        active_end = self.active_end[safe_ids]
        conditioned = self.conditioned_mask_cpu
        height = self.heights[safe_ids]
        while conditioned.ndim < time_steps.ndim:
            conditioned = conditioned.unsqueeze(-1)
            visible_start = visible_start.unsqueeze(-1)
            visible_end = visible_end.unsqueeze(-1)
            active_start = active_start.unsqueeze(-1)
            active_end = active_end.unsqueeze(-1)
            height = height.unsqueeze(-1)
        visible = conditioned & (time_steps >= visible_start) & (time_steps < visible_end)
        active = conditioned & (time_steps >= active_start) & (time_steps < active_end)
        return visible.to(device), active.to(device), height.to(device)

    def visible_start_for(self, env_ids: torch.Tensor) -> torch.Tensor:
        env_ids = env_ids.to(device="cpu", dtype=torch.long)
        safe_ids = self.schedule_ids[env_ids].clamp_min(0)
        return self.visible_start[safe_ids]

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": "scaletrack.four_arm_sampler_state.v1",
            "assignment_tape_sha256": self.sampler_receipt()["matched_sampling_contract_sha256"],
            "schedule_ids": self.schedule_ids.clone(),
            "reset_ordinals": self.reset_ordinals.clone(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("schema_version") != "scaletrack.four_arm_sampler_state.v1":
            raise ValueError("invalid four-arm sampler state schema")
        if state.get("assignment_tape_sha256") != self.sampler_receipt()["matched_sampling_contract_sha256"]:
            raise ValueError("four-arm sampler state belongs to a different assignment tape")
        for name, target in (("schedule_ids", self.schedule_ids), ("reset_ordinals", self.reset_ordinals)):
            value = state.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != target.shape:
                raise ValueError(f"invalid four-arm sampler tensor: {name}")
            target.copy_(value.to(device="cpu", dtype=torch.long))

    def sampler_receipt(self) -> dict[str, object]:
        slot_counts = {
            f"height_bin_{height_bin}/{('bones_seed' if is_bones else 'other')}": self.slot_layout.count(
                (height_bin, is_bones)
            )
            for height_bin in range(4)
            for is_bones in (True, False)
        }
        matched_contract = {
            "num_envs": len(self.schedule_ids),
            "conditioned_envs": self.conditioned_count,
            "rehearsal_envs": len(self.schedule_ids) - self.conditioned_count,
            "slot_counts": slot_counts,
            "pool_counts": {
                f"height_bin_{height_bin}/{('bones_seed' if is_bones else 'other')}": len(
                    self.pools[(height_bin, is_bones)]
                )
                for height_bin in range(4)
                for is_bones in (True, False)
            },
            "schedule_sha256": self.schedule_sha256,
            "geometry_sha256": self.geometry_sha256,
            "assignment_tape": {
                "algorithm": ASSIGNMENT_ALGORITHM,
                "seed": self.sampler_seed,
                "rank": self.sampler_rank,
                "pools": {
                    f"height_bin_{height_bin}/{('bones_seed' if is_bones else 'other')}": [
                        str(self.rows[index]["schedule_id"])
                        for index in self.pools[(height_bin, is_bones)]
                    ]
                    for height_bin in range(4)
                    for is_bones in (True, False)
                },
            },
        }
        return {
            "schema_version": "scaletrack.four_arm_sampler_receipt.v1",
            "variant": self.variant,
            **matched_contract,
            "matched_sampling_contract_sha256": _sha256_json(matched_contract),
        }
