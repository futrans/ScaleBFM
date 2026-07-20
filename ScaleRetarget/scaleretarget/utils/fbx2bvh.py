"""
This code is a variation of https://github.com/rubenvillegas/cvpr2018nkn/blob/master/datasets/fbx2bvh.py
"""

import os
import bpy
import glob
import typer
from pathlib import Path


def main(
    input_path: str, output_path: str
)-> None:
    print(f"Input Path: {input_path}")
    print(f"Output Path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    if os.path.isfile(input_path):
        fbx_list = [input_path]
    elif os.path.isdir(input_path):
        fbx_list = glob.glob(f"{input_path}/**/*.fbx", recursive=True)
    else:
        raise NotImplementedError
    
    for fbx_file in fbx_list:
        print(fbx_file)
        file_name = Path(fbx_file).stem
        out_file_path = os.path.join(output_path, f"{file_name}.bvh")
        
        bpy.ops.import_scene.fbx(filepath=fbx_file)

        action = bpy.data.actions[-1]
        assert action.frame_range[0] < 9999 and action.frame_range[1] > -9999
        
        bpy.ops.export_anim.bvh(
            filepath=out_file_path,
            frame_start=int(action.frame_range[0]),
            frame_end=int(action.frame_range[1]),
            # root_transform_only=True,
        )
        bpy.data.actions.remove(bpy.data.actions[-1])


if __name__ == "__main__":
    typer.run(main)
