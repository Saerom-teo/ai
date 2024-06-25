import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.info")


def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("uvicorn.info")

    return logger

def log_error(message):
    logger.error(message)

def log_warning(message):
    logger.warning(message)