import argparse  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Utility to convert a URDF into USD format."
)
parser.add_argument("input", type=str, help="The path to the input URDF file.")
parser.add_argument("output", type=str, help="The path to store the USD file.")
parser.add_argument(
    "--fix-base",
    action="store_true",
    default=False,
    help="Fix the base to where it is imported.",
)
parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=False,
    help="Make the asset instanceable for efficient cloning.",
)
parser.add_argument(
    "--merge-joints",
    action="store_true",
    default=False,
    help="Consolidate links that are connected by fixed joints.",
)
parser.add_argument(
    "--self-collision",
    action="store_true",
    default=False,
    help="Whether to enable collision with asset itself.",
)
parser.add_argument(
    "--replace-cylinders-with-capsules",
    action="store_true",
    default=False,
    help="Whether to replace cylinders with capsules.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib  # noqa: E402
import os  # noqa: E402

import carb  # noqa: E402
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
import omni.kit.app  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import check_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402


def main():
    # check valid file path
    urdf_path = args_cli.input
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid file path: {urdf_path}")
    # create destination path
    dest_path = args_cli.output
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)

    # create the converter configuration
    urdf_converter_cfg = UrdfConverterCfg(
        fix_base=args_cli.fix_base,
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        force_usd_conversion=True,
        make_instanceable=args_cli.make_instanceable,
        merge_fixed_joints=args_cli.merge_joints,
        joint_drive=None, # weishuai: you may add if you do not want to set pd gains during training time.
        self_collision=args_cli.self_collision,
        replace_cylinders_with_capsules=args_cli.replace_cylinders_with_capsules
    )

    # Print info
    print("-" * 80)
    print("-" * 80)
    print(f"Input URDF file: {urdf_path}")
    print("URDF importer config:")
    print_dict(urdf_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)

    # Create Urdf converter and import the file
    urdf_converter = UrdfConverter(urdf_converter_cfg)
    # print output
    print("URDF importer output:")
    print(f"Generated USD file: {urdf_converter.usd_path}")
    print("-" * 80)
    print("-" * 80)

    # Determine if there is a GUI to update:
    # acquire settings interface
    carb_settings_iface = carb.settings.get_settings()
    # read flag for whether a local GUI is enabled
    local_gui = carb_settings_iface.get("/app/window/enabled")
    # read flag for whether livestreaming GUI is enabled
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    # Simulate scene (if not headless)
    if local_gui or livestream_gui:
        # Open the stage with USD
        stage_utils.open_stage(urdf_converter.usd_path)
        # Reinitialize the simulation
        app = omni.kit.app.get_app_interface()
        # Run simulation
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                # perform step
                app.update()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
