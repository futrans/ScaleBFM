# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--eval_interval", type=int, default=None, help="Policy evaluation interval in PPO updates.")
parser.add_argument("--motion_file", type=str, required=True, help="The name of the motion file.")
parser.add_argument(
    "--validation_motion_file",
    type=str,
    default="",
    help="Validation motion YAML used only for evaluation during training.",
)
parser.add_argument("--distributed", action="store_true", default=False, help="Distributed training.")
parser.add_argument(
    "--four_arm_variant",
    choices=("off", "b0", "b1", "c_eq", "c_ub"),
    default="off",
    help="Four-arm development variant. 'off' preserves the original ScaleTrack environment.",
)
parser.add_argument("--four_arm_schedule_file", type=str, default="", help="Four-arm training schedule manifest.")
parser.add_argument("--four_arm_geometry_file", type=str, default="", help="Four-arm robot-top geometry sidecar.")
parser.add_argument(
    "--shared_init_checkpoint",
    type=str,
    default="",
    help="Expanded shared-init checkpoint for a new four-arm continuation run.",
)
parser.add_argument("--four_arm_conditioned_fraction", type=float, default=0.5)
parser.add_argument("--four_arm_bones_seed_max_fraction", type=float, default=0.6)
parser.add_argument("--four_arm_pelvis_height_offset_m", type=float, default=0.49)
parser.add_argument("--four_arm_condition_reward_weight", type=float, default=1.0)
parser.add_argument("--four_arm_condition_reward_std", type=float, default=0.05)
parser.add_argument("--four_arm_nonfoot_contact_weight", type=float, default=-0.1)
parser.add_argument("--four_arm_contact_threshold", type=float, default=1.0)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import scaletrack.tasks  # noqa: F401

from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner
from scaletrack.tasks.tracking.four_arm_cfg import configure_four_arm_env

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _run_metadata(args: argparse.Namespace, num_envs_per_rank: int) -> dict:
    scalebfm_root = Path(__file__).resolve().parents[4]
    maskmotion_root = scalebfm_root.parent.parent
    manifests = {}
    for split, raw_path in (
        ("train", args.motion_file),
        ("validation", args.validation_motion_file),
    ):
        if raw_path:
            path = Path(raw_path).resolve()
            provenance_path = path.with_name(f"{path.stem}.provenance.jsonl")
            manifests[split] = {
                "path": str(path),
                "sha256": _sha256_file(str(path)),
                "provenance_path": str(provenance_path) if provenance_path.is_file() else None,
                "provenance_sha256": _sha256_file(str(provenance_path)) if provenance_path.is_file() else None,
            }
    four_arm = {"variant": args.four_arm_variant}
    four_arm["runtime"] = {
        "conditioned_fraction": args.four_arm_conditioned_fraction,
        "bones_seed_max_fraction": args.four_arm_bones_seed_max_fraction,
        "pelvis_height_offset_m": args.four_arm_pelvis_height_offset_m,
        "condition_reward_weight": args.four_arm_condition_reward_weight,
        "condition_reward_std": args.four_arm_condition_reward_std,
        "nonfoot_contact_weight": args.four_arm_nonfoot_contact_weight,
        "contact_threshold": args.four_arm_contact_threshold,
        "sampler_seed": args.seed,
        "sampler_rank": int(os.getenv("RANK", "0")),
    }
    for field, raw_path in (
        ("schedule", args.four_arm_schedule_file),
        ("geometry", args.four_arm_geometry_file),
        ("shared_init", args.shared_init_checkpoint),
    ):
        if raw_path:
            path = Path(raw_path).resolve()
            four_arm[f"{field}_path"] = str(path)
            four_arm[f"{field}_sha256"] = _sha256_file(str(path))
    return {
        "manifests": manifests,
        "git_commits": {
            "MaskMotion": _git_commit(maskmotion_root),
            "ScaleBFM": _git_commit(scalebfm_root),
        },
        "distributed_world_size": int(os.getenv("WORLD_SIZE", "1")),
        "num_envs_per_rank": num_envs_per_rank,
        "four_arm": four_arm,
    }


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    agent_cfg.eval_interval = args_cli.eval_interval if args_cli.eval_interval is not None else agent_cfg.eval_interval

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env_cfg.commands.motion.motion_file = args_cli.motion_file
    env_cfg.commands.motion.validation_motion_file = args_cli.validation_motion_file
    env_cfg.commands.motion.debug_vis = args_cli.video
    if args_cli.four_arm_variant != "off":
        configure_four_arm_env(
            env_cfg,
            variant=args_cli.four_arm_variant,
            schedule_file=args_cli.four_arm_schedule_file,
            geometry_file=args_cli.four_arm_geometry_file,
            conditioned_fraction=args_cli.four_arm_conditioned_fraction,
            bones_seed_max_fraction=args_cli.four_arm_bones_seed_max_fraction,
            pelvis_height_offset_m=args_cli.four_arm_pelvis_height_offset_m,
            condition_reward_weight=args_cli.four_arm_condition_reward_weight,
            condition_reward_std=args_cli.four_arm_condition_reward_std,
            nonfoot_contact_weight=args_cli.four_arm_nonfoot_contact_weight,
            contact_threshold=args_cli.four_arm_contact_threshold,
            sampler_seed=args_cli.seed,
            sampler_rank=int(os.getenv("RANK", "0")),
        )
    elif args_cli.shared_init_checkpoint:
        raise ValueError("--shared_init_checkpoint requires an active --four_arm_variant")
    if agent_cfg.resume and args_cli.shared_init_checkpoint:
        raise ValueError("resume and shared-init are mutually exclusive")

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

        gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        gpu_global_rank = int(os.getenv("RANK", "0"))
        gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        assert gpu_world_size > 1, f"You have enabled distributed training but there is only {gpu_world_size} GPUs detected!"
        
        # weishuai: We init communication at the very beginning instead of the runner as the initialization of some environment components may need sync.
        distributed_timeout_seconds = int(os.getenv("SCALETRACK_DISTRIBUTED_TIMEOUT_SECONDS", "10800"))
        if gpu_global_rank == 0:
            print(f"[INFO] Distributed initialization timeout: {distributed_timeout_seconds}s")
        torch.distributed.init_process_group(
            backend="nccl",
            rank=gpu_global_rank,
            world_size=gpu_world_size,
            timeout=timedelta(seconds=distributed_timeout_seconds),
        )
        torch.cuda.set_device(gpu_local_rank)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    # log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # if agent_cfg.run_name:
    #     log_dir += f"_{agent_cfg.run_name}"
    log_dir = agent_cfg.run_name
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner_cfg = agent_cfg.to_dict()
    runner_cfg["run_metadata"] = _run_metadata(args_cli, env_cfg.scene.num_envs)
    print(
        "[INFO] Resolved logger: "
        f"type={runner_cfg.get('logger')} "
        f"wandb_mode={runner_cfg.get('wandb_mode') or os.getenv('WANDB_MODE') or 'online'}"
    )
    runner = OnPolicyRunner(env, runner_cfg, log_dir=log_dir, device=agent_cfg.device)
    if args_cli.four_arm_variant != "off":
        command = env.unwrapped.command_manager.get_term("motion")
        receipt = command.four_arm.sampler_receipt()
        receipt["run_metadata"] = runner_cfg["run_metadata"]
        rank = int(os.getenv("RANK", "0"))
        receipt_path = Path(log_dir) / f"four_arm_sampler_receipt.rank-{rank:05d}.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[ScaleBFM four-arm] sampler receipt: {receipt_path}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    try:
        # save resume path before creating a new log_dir
        if agent_cfg.resume:
            # get path to previous checkpoint
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            # load previously trained model
            runner.load(resume_path)
            runner.reset_environment_after_resume()
        elif args_cli.shared_init_checkpoint:
            print(f"[INFO]: Loading four-arm shared init from: {args_cli.shared_init_checkpoint}")
            runner.load_shared_init(args_cli.shared_init_checkpoint)
            runner.reset_environment_after_resume()

        # run training
        remaining_iterations = max(0, agent_cfg.max_iterations - runner.current_learning_iteration)
        print(
            f"[INFO] Training updates: completed={runner.current_learning_iteration} "
            f"target={agent_cfg.max_iterations} remaining={remaining_iterations}"
        )
        runner.learn(num_learning_iterations=remaining_iterations, init_at_random_ep_len=not agent_cfg.resume)
    finally:
        runner.close()
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
