import logging
import re
import sys


def setup_logger(caller_file=None):
    if not caller_file:
        logfile_name = f'{__file__}.log'
    else:
        filtered_caller_file = re.sub(r'\.py$', "", caller_file)
        logfile_name = f'{filtered_caller_file}.log'
        
    logger = logging.getLogger()
    
    logging.basicConfig(filename=logfile_name,
                        encoding='utf-8', level=logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
