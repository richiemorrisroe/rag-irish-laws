import logging
import sys


def setup_logger():
    logger = logging.getLogger()
    
    logging.basicConfig(filename=f'{__FILE__}.log',
                        encoding='utf-8', level=logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
