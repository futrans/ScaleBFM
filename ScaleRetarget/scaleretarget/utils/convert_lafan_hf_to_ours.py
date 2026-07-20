import os
import joblib
import numpy as np
import typer
from pathlib import Path

def main(
    lafan_dir: str, save_dir: str
) -> None:
    
    all_lafan_files = Path(lafan_dir).glob(f"*.csv")
    
    os.makedirs(save_dir, exist_ok=True)
    
    for motion_path in all_lafan_files:

        data = np.genfromtxt(motion_path, delimiter=",")

        dic = {
            "root_pos": data[:, :3],
            "root_rot": data[:, 3:7], # xyzw
            "dof_pos": data[:, 7:],
            "fps": 30
        }
        
        joblib.dump(dic, Path(save_dir) / motion_path.with_suffix(".pkl").name)

if __name__ == "__main__":
    typer.run(main)