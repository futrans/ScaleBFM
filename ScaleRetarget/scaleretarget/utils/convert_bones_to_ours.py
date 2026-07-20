from pathlib import Path

import joblib
import numpy as np
import typer


EXPECTED_DOF_COUNT = 29
EXPECTED_COLUMN_COUNT = 1 + 3 + 3 + EXPECTED_DOF_COUNT


def euler_xyz_degrees_to_quaternion(euler_degrees: np.ndarray) -> np.ndarray:
    half_angles = np.deg2rad(euler_degrees) / 2.0
    roll, pitch, yaw = half_angles.T
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.column_stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )


def convert_motion(
    source_path: Path,
    destination_path: Path,
    source_fps: int,
    target_fps: int,
) -> None:
    data = np.genfromtxt(source_path, delimiter=",", skip_header=1)
    data = np.atleast_2d(data)

    if data.shape[1] != EXPECTED_COLUMN_COUNT:
        raise ValueError(
            f"{source_path} has {data.shape[1]} columns; expected "
            f"{EXPECTED_COLUMN_COUNT} (frame + root position/rotation + "
            f"{EXPECTED_DOF_COUNT} G1 joints)."
        )
    if not np.isfinite(data).all():
        raise ValueError(f"{source_path} contains missing or non-numeric values.")

    sampling_step = source_fps // target_fps
    data = data[::sampling_step]

    root_position = data[:, 1:4] / 100.0
    root_euler_degrees = data[:, 4:7]
    joint_degrees = data[:, 7:]

    converted = {
        "root_pos": root_position,
        "root_rot": euler_xyz_degrees_to_quaternion(root_euler_degrees),
        "dof_pos": np.deg2rad(joint_degrees),
        "fps": target_fps,
    }

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(converted, destination_path)


def main(
    input_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, readable=True
    ),
    output_dir: Path = typer.Argument(..., file_okay=False),
    source_fps: int = typer.Option(120, min=1, help="Frame rate of the CSV files."),
    target_fps: int = typer.Option(30, min=1, help="Output frame rate."),
    overwrite: bool = typer.Option(False, help="Replace existing PKL files."),
) -> None:
    if target_fps > source_fps or source_fps % target_fps != 0:
        raise typer.BadParameter(
            "source-fps must be evenly divisible by target-fps, and target-fps "
            "cannot exceed source-fps."
        )

    source_paths = sorted(input_dir.rglob("*.csv"))
    if not source_paths:
        raise typer.BadParameter(f"No CSV files were found beneath {input_dir}.")

    converted_count = 0
    skipped_count = 0
    for source_path in source_paths:
        relative_path = source_path.relative_to(input_dir).with_suffix(".pkl")
        destination_path = output_dir / relative_path
        if destination_path.exists() and not overwrite:
            skipped_count += 1
            continue

        convert_motion(source_path, destination_path, source_fps, target_fps)
        converted_count += 1

    typer.echo(
        f"Converted {converted_count} motion(s); skipped {skipped_count} existing "
        f"motion(s). Output: {output_dir}"
    )


if __name__ == "__main__":
    typer.run(main)
