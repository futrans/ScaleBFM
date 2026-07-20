import os
import joblib
from hydra.utils import instantiate
from loguru import logger

_retargeter = None
_formatter = None

def initialize_worker(retargeter_config, formatter_config):
    global _retargeter, _formatter
    _retargeter = instantiate(retargeter_config)
    _formatter = instantiate(formatter_config)


def process_single_item(args):
    save_path, frames, extras = args
    if _retargeter is None or _formatter is None:
        raise RuntimeError("Worker models were not initialized")
            
    _retargeter.update(extras)
    qpos_list = _retargeter.retarget(frames)
    formatted_results = _formatter.format(qpos_list, extras)
            
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(formatted_results, save_path)
    logger.info(f"Retargeted motion saved to {save_path}")

    return save_path


def producer(data_loader, queue):
    try:
        for item in data_loader:
            # The bounded queue blocks here and provides backpressure.
            queue.put(item)
    finally:
        queue.put(None)
