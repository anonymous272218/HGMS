import os
import sys
import logging
import datetime
from .config import config

logger = logging.getLogger("Main Logger")
if len(logger.handlers) == 0:
    logger.propagate = 0
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
    rank = int(os.getenv('RANK', -1))
    level = logging.INFO if rank in {-1, 0} else logging.ERROR
    logger.setLevel(level)
    # console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter
    logger.addHandler(console_handler)
    # log file
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root = config.save_root / 'log'
    log_root.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_root / nowTime)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
