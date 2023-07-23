import time

from loguru import logger


def init_logger(file_name):
    file_name = f"logs/{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) }_{file_name}.log"
    logger.add(file_name)
    return file_name


def log(msg):
    logger.info(msg)


def log_args(args):
    for k, v in args.__dict__.items():
        if str(k).startswith("__"):
            continue
        log(f"{k}: {v}")


def log_time(desc: str, elapsed_time: float):
    logger.info(f"[{desc}]: {elapsed_time:0.4f} seconds")
