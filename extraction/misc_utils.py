import sys
import logging


def get_logger(module_name):
    """ Logger to stdout

    Typical use: logger = get_module_logger(__name__)
    """
    logger = logging.getLogger(module_name)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger
