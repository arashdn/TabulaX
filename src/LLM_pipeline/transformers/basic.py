import os
import pathlib
import pickle
import time

import openai


USE_TQDM = False

BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute()
CODE_BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.absolute()
BRIDGE_PROMPT_CACHE_PATH = BASE_PATH / "cache/basic_bridge_prompts"
# @TODO: Caching ignores temp and other running params

with open(BASE_PATH / 'openai.key', 'r') as f:
    API_KEY = f.read()

with open(BASE_PATH / 'deepseek.key', 'r') as f:
    DEEPSEEK_API_KEY = f.read()


bridge_cache_dict = None
bridge_cache_file = None


def predict_bridge_value(examples, src, model_name, prompt_version, sleep=-1):

    mdl = model_name
    if mdl.startswith("gpt-4o-"):
        mdl = "gpt-4o"
    if mdl == "llama3.1-8b":
        mdl = "llama3"
    if mdl.startswith("deepseek"):
        mdl = "deepseek"

    with open(CODE_BASE_PATH / f"transformers/prompts/{mdl}/basic_bridge_prompt_{prompt_version}.txt") as f:
        pmpt = f.read()

    str_examp = ""

    for exp in examples:
        str_examp += f"{exp[0]} -> {exp[1]}\n"

    prompt = pmpt.format(examples=str_examp, src_value=src)
    # print(f"=========\n{prompt}\n***")
    if prompt in bridge_cache_dict:
        respond = bridge_cache_dict[prompt].choices[0].message.content
        # print("Hit cache")
    else:
        api_model_name = model_name
        if model_name.startswith("gpt"):
            client = openai.OpenAI(api_key=API_KEY)
        elif model_name == "llama3.1-8b":
            client = openai.OpenAI(
                api_key="None",
                base_url="http://localhost:8000/v1",
            )
            api_model_name = "meta-llama/Llama-3.1-8B-Instruct"
        elif model_name.startswith("deepseek"):
            client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")

        completion = client.chat.completions.create(
            model=api_model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0000001,
            seed=12345,
            max_tokens=100,
            # frequency_penalty=0.0
        )
        respond = completion.choices[0].message.content
        bridge_cache_dict[prompt] = completion

        with open(bridge_cache_file, 'wb') as fp:
            pickle.dump(bridge_cache_dict, fp)

        if sleep > 0:
            time.sleep(sleep)

    out = respond.strip()
    # print(out)

    return out


def get_basic_values(examples, test, params):

    model_name = params["model_name"]
    prompt_version = params["prompt_version"]

    global bridge_cache_dict, bridge_cache_file

    bridge_cache_file = BRIDGE_PROMPT_CACHE_PATH / f"{model_name}.pkl"
    if os.path.exists(bridge_cache_file):
        with open(bridge_cache_file, 'rb') as fp:
            bridge_cache_dict = pickle.load(fp)
    else:
        bridge_cache_dict = {}

    # (bridge, target, src)
    exmps = [(exp[2], exp[1]) for exp in examples]

    test_new = []
    try:
        if not USE_TQDM:
            raise ImportError
        import tqdm
        itr = tqdm.tqdm(test)
        print()
        time.sleep(0.1)
    except Exception:
        itr = test

    for exp in itr:
        bridge = predict_bridge_value(exmps, exp[2], model_name, prompt_version)
        test_new.append((bridge, exp[1], exp[2]))

    return test_new, {

    }

