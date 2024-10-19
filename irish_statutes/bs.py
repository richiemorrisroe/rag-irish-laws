import logging
import re
import sys

from bs4 import BeautifulSoup
import pandas as pd

logger = logging.getLogger()

logging.basicConfig(filename='law_index.log',
                    encoding='utf-8', level=logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)




def extract_text_of_act(dct):
    html = BeautifulSoup(dct['full_text'], "lxml")
    txts = [x.text for x in html.find_all('td')]
    full_text = ' '.join([x for x in txts])
    full_text_strip_newlines = full_text.replace('\n', ' ')
    dct['plain_text'] = full_text_strip_newlines
    return dct

def get_html_and_act_name(laws):
    laws_s = laws[['name', 'year', 'full_text']]
    laws_dict = laws_s.to_dict(orient='records')
    return laws_dict


def write_acts_to_folder(act_name, act_year, act_text):
    act_name_underscores = re.sub(r"\W+", "_", act_name).lower()
    base_folder = 'csv_laws'
    if len(act_name_underscores)==1:
        logger.warning(f"empty_filename")
        return False
    else:
        file_name = f"{base_folder}/{act_name_underscores}.txt"
        logger.info(f"writing to {file_name}")
        with open(file_name, 'w') as f:
            f.write(act_text)
        return True

    
laws = pd.read_json("laws.jsonl", lines=True)
laws_dict = get_html_and_act_name(laws)
all_plain_text = [extract_text_of_act(doc) for doc in laws_dict]

for act in all_plain_text:
    write_acts_to_folder(act['name'], act['year'], act['plain_text'])
