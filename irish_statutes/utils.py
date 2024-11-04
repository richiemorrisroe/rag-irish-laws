import logging
from pprint import pprint
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
    handler.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        pprint(text_md)
        print(p.get_template())
        # display(Markdown("<br><br>"))
