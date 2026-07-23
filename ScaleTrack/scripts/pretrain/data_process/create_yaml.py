import argparse
from pathlib import Path

import yaml


def create_yaml(data_dir: str, output_path: str) -> Path:
    """Create one YAML file containing every processed motion in data_dir."""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise NotADirectoryError(f"Motion directory does not exist: {data_path}")

    motion_files = sorted(data_path.rglob("*.npz"))
    if not motion_files:
        raise FileNotFoundError(f"No processed .npz motion files found under: {data_path}")

    motions = {}
    for motion_file in motion_files:
        motion_name = motion_file.stem
        if motion_name in motions:
            raise ValueError(
                f"Duplicate motion name '{motion_name}' found in:\n"
                f"  {motions[motion_name]}\n"
                f"  {motion_file}"
            )
        motions[motion_name] = str(motion_file)

    yaml_path = Path(output_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as yaml_file:
        yaml.safe_dump(motions, yaml_file, sort_keys=False)

    print(f"Saved {len(motions)} motions to: {yaml_path}")
    return yaml_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one YAML file for all processed motion files."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing processed .npz motions.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="source/scaletrack/data/yaml_files/motions.yaml",
        help="Path of the YAML file to create.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args_cli = parse_args()
    create_yaml(args_cli.data_dir, args_cli.output_path)
