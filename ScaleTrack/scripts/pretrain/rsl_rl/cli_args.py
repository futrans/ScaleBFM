from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default="debug", help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    arg_group.add_argument(
        "--wandb-project",
        "--wandb_project",
        dest="wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project. Overrides the runner config and WANDB_PROJECT.",
    )
    arg_group.add_argument(
        "--wandb-entity",
        "--wandb_entity",
        dest="wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity. Falls back to WANDB_ENTITY.",
    )
    arg_group.add_argument(
        "--wandb-mode",
        "--wandb_mode",
        dest="wandb_mode",
        type=str,
        default=None,
        choices={"disabled", "offline", "online"},
        help="Weights & Biases mode.",
    )
    arg_group.add_argument(
        "--wandb-run-name",
        "--wandb_run_name",
        dest="wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name. Defaults to the local log directory name.",
    )
    arg_group.add_argument(
        "--wandb-group",
        "--wandb_group",
        dest="wandb_group",
        type=str,
        default=None,
        help="Weights & Biases run group.",
    )
    arg_group.add_argument(
        "--wandb-tag",
        "--wandb_tag",
        dest="wandb_tags",
        action="append",
        default=None,
        help="Weights & Biases tag. Repeat the argument to add multiple tags.",
    )
    arg_group.add_argument(
        "--wandb-run-id",
        "--wandb_run_id",
        dest="wandb_run_id",
        type=str,
        default=None,
        help="Stable Weights & Biases run ID used to resume the same remote run.",
    )
    arg_group.add_argument(
        "--wandb_path", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    if args_cli.wandb_project is not None:
        agent_cfg.wandb_project = args_cli.wandb_project
    if args_cli.wandb_entity is not None:
        agent_cfg.wandb_entity = args_cli.wandb_entity
    if args_cli.wandb_mode is not None:
        agent_cfg.wandb_mode = args_cli.wandb_mode
    if args_cli.wandb_run_name is not None:
        agent_cfg.wandb_run_name = args_cli.wandb_run_name
    if args_cli.wandb_group is not None:
        agent_cfg.wandb_group = args_cli.wandb_group
    if args_cli.wandb_tags is not None:
        agent_cfg.wandb_tags = args_cli.wandb_tags
    if args_cli.wandb_run_id is not None:
        agent_cfg.wandb_run_id = args_cli.wandb_run_id

    return agent_cfg
