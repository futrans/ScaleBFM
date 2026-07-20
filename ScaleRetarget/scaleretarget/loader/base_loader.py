import os
import glob
from enum import Enum
from pathlib import Path
from omegaconf import DictConfig
from loguru import logger

class Mode(Enum):
    file = 0
    dir = 1

class BaseLoader:
    def __init__(self, config: DictConfig):
        self.config = config
        self.format = self.config.data_format
        self.overwrite = self.config.get('overwrite', False)
        self.target_fps = self.config.get('target_fps', 30)
        self.output_dir = os.path.expanduser(self.config.get('output_dir', 'retargeted_dataset'))
        
    def load(self, data_path):
        self.data_path = data_path

        if os.path.isdir(self.data_path):
            self.mode = Mode.dir
            self.data_list = glob.glob(f"{self.data_path}/**/*{self.format}", recursive=True)
            self.data_num = len(self.data_list)
        elif os.path.isfile(self.data_path):
            suffix = os.path.splitext(self.data_path)[-1]
            assert suffix == self.format, f"You are using the data loader for {self.format} format but the data you give is {suffix} format!"
            self.mode = Mode.file
            self.data_list = [self.data_path]
            self.data_num = 1
        else:
            raise Exception(f"Check your data path: {self.data_path}")
        
        logger.info(f"[Loader] Total number of motions: {self.data_num}")
        
    def __iter__(self):
        self.current_idx = 0
        
        while self.current_idx < self.data_num:
            try: 
                sample_path = self.data_list[self.current_idx]
                if self.mode == Mode.file:
                    relative_path = os.path.basename(sample_path)
                else:
                    data_dir_name = Path(self.data_path).resolve().name
                    relative_path = os.path.join(
                        data_dir_name,
                        os.path.relpath(sample_path, self.data_path),
                    )
                relative_path = os.path.splitext(relative_path)[0] + '.pkl'
                save_path = os.path.join(self.output_dir, relative_path)

                if os.path.abspath(save_path) == os.path.abspath(sample_path):
                    raise ValueError(
                        f"Output path resolves to the source file: {sample_path}. "
                        "Choose a different output_dir."
                    )
                if os.path.exists(save_path) and not self.overwrite:
                    logger.warning(f"[Loader] {save_path} already exists!")
                    self.current_idx += 1
                    continue
                frames, extras = self._load_sample(sample_path)
                yield save_path, frames, extras
                self.current_idx += 1
            except Exception as e:
                logger.warning(f"[Loader] File {self.data_list[self.current_idx]} is broken! Skip it for error {e}")
                self.current_idx += 1

    def _load_sample(self, sample_path: Path):
        raise NotImplementedError
