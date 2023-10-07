import logging
import logging.config
from pathlib import Path
from utils import read_json


def setup_logging(save_dir='./log', log_config='logger/logger_config.json', file_name=None):
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        filename = file_name if file_name else config['filename']
        log_file = str(Path(save_dir) / filename)
        config['filename'] = log_file
        logging.basicConfig(**config)
