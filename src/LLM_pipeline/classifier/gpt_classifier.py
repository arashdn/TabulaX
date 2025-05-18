import json
import os
import pathlib
import pickle
import re
import sys

import openai

BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute()
CODE_BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.absolute()
PROMPT_CACHE_PATH = BASE_PATH / "cache/classifier_prompts"
# @TODO: Caching ignores temp and other running params

with open(BASE_PATH / 'openai.key', 'r') as f:
    API_KEY = f.read()

MODEL_NAME = "gpt-4o-2024-05-13"
# MODEL_NAME = "gpt-4o-mini-2024-07-18"
PROMPT_VERSION = "v002"

if "-mini" in MODEL_NAME:
    ALL_CLASSES_JSON = str(BASE_PATH / "data/Classes/gpt_4o-mini_classified.json")
else:
    ALL_CLASSES_JSON = str(BASE_PATH / "data/Classes/gpt_classified.json")

sys.path.append(str(CODE_BASE_PATH))


from util.dataset import sample_data
from classifierutil import get_gold_label, ALLOWED_CLASSES
from report_metrics import report_metrics


EXAMPLE_SIZE = 5
EXAMPLE_SIZE_TYPE = "fixed"
DS_PATHS = [
    str(BASE_PATH / "data/Datasets/DataXFormer"),
    str(BASE_PATH / "data/Datasets/AutoJoin"),
    str(BASE_PATH / "data/Datasets/FlashFill"),
    str(BASE_PATH / "data/Datasets/All_TDE"),
    ]



def get_prediction(examples):
    cache_file = PROMPT_CACHE_PATH / f"{MODEL_NAME}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fp:
            cache_dict = pickle.load(fp)
    else:
        cache_dict = {}

    mdl = MODEL_NAME
    if mdl.startswith("gpt-4o-"):
        mdl = "gpt-4o"

    with open(CODE_BASE_PATH / f"classifier/prompts/{mdl}/class_prompt_{PROMPT_VERSION}.txt") as f:
        pmpt = f.read()

    str_examp = ""

    for exp in examples:
        str_examp += f"(\"{exp[0]}\" -> \"{exp[1]}\"),"

    prompt = pmpt.format(examples=str_examp)

    if prompt in cache_dict:
        respond = cache_dict[prompt].choices[0].message.content
        # print("Hit cache")
    else:
        client = openai.OpenAI(api_key=API_KEY)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0000001,
            seed=12345,
            max_tokens=100,
            # frequency_penalty=0.0

        )
        respond = completion.choices[0].message.content
        cache_dict[prompt] = completion

    with open(cache_file, 'wb') as fp:
        pickle.dump(cache_dict, fp)

    out = re.split('\s+', respond.replace("Class:", "").strip())[0]
    if out in ALLOWED_CLASSES:
        return out
    print(out)
    raise ValueError(f"Invalid out: {out}")





all_labels = {}
def save_all_classes():
    for ds_path in DS_PATHS:
        ds_name = pathlib.Path(ds_path).name
        print(f"Working on {ds_name}:")
        cnt = 1
        tables = sample_data(ds_path, EXAMPLE_SIZE, EXAMPLE_SIZE_TYPE)
        for name, table in tables.items():
            gold_label = get_gold_label(name, ds_path)
            prediction = get_prediction(table['train'])

            print(f"{cnt}/{len(tables)}: {name} -> {prediction} (expected {gold_label})")

            cnt += 1

            assert name not in all_labels
            all_labels[name] = {
                "golden_value": gold_label,
                "predicted_value": prediction,
                "full_name": f"{ds_name}/{name}",
            }

    json.dump(all_labels, open(ALL_CLASSES_JSON, 'w'), indent=2)





if __name__ == "__main__":
    save_all_classes()
    report_metrics(ALL_CLASSES_JSON)





