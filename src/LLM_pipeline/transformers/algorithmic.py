import os
import pathlib
import pickle
import re

import openai

BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute()
CODE_BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.absolute()
REL_PROMPT_CACHE_PATH = BASE_PATH / "cache/alg_rel_prompts"
CODE_PROMPT_CACHE_PATH = BASE_PATH / "cache/alg_code_prompts"
# @TODO: Caching ignores temp and other running params

with open(BASE_PATH / 'openai.key', 'r') as f:
    API_KEY = f.read()





def get_relation(examples, model_name, prompt_version):
    cache_file = REL_PROMPT_CACHE_PATH / f"{model_name}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fp:
            cache_dict = pickle.load(fp)
    else:
        cache_dict = {}

    mdl = model_name
    if mdl.startswith("gpt-4o-"):
        mdl = "gpt-4o"
    if mdl == "llama3.1-8b":
        mdl = "llama3"

    with open(CODE_BASE_PATH / f"transformers/prompts/{mdl}/alg_rel_prompt_{prompt_version}.txt") as f:
        pmpt = f.read()

    str_examp = ""

    for exp in examples:
        str_examp += f"(\"{exp[0]}\" -> \"{exp[1]}\"),"

    prompt = pmpt.format(examples=str_examp)

    if prompt in cache_dict:
        respond = cache_dict[prompt].choices[0].message.content
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
        cache_dict[prompt] = completion

    with open(cache_file, 'wb') as fp:
        pickle.dump(cache_dict, fp)

    out = respond.replace("Relationship:", "").strip()
    return out



def get_code(examples, relation, model_name, prompt_version):
    cache_file = CODE_PROMPT_CACHE_PATH / f"{model_name}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fp:
            cache_dict = pickle.load(fp)
    else:
        cache_dict = {}

    mdl = model_name
    if mdl.startswith("gpt-4o-"):
        mdl = "gpt-4o"
    if mdl == "llama3.1-8b":
        mdl = "llama3"

    with open(CODE_BASE_PATH / f"transformers/prompts/{mdl}/alg_code_prompt_{prompt_version}.txt") as f:
        pmpt = f.read()

    str_examp = ""

    for exp in examples:
        str_examp += f"Input: \"{exp[0]}\"\nExpected Output:\"{exp[1]}\"\n***\n"

    prompt = pmpt.format(examples=str_examp, relation=relation)


    if prompt in cache_dict:
        respond = cache_dict[prompt].choices[0].message.content
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
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")

        completion = client.chat.completions.create(
            model=api_model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0000001,
            seed=12345,
            max_tokens=1000,
            # frequency_penalty=0.0
        )
        respond = completion.choices[0].message.content
        cache_dict[prompt] = completion

    with open(cache_file, 'wb') as fp:
        pickle.dump(cache_dict, fp)


    return respond, {
        'relationship': relation,
        'prompt': prompt,
        'respond': respond,
    }




def get_algorithmic_function(examples, model_name, prompt_version):
    exmps = [(exp[0], exp[1]) for exp in examples]
    relation = get_relation(exmps, model_name, prompt_version)
    func, detail = get_code(exmps, relation, model_name, prompt_version)


    return detail, func

