
import pathlib



from transformers.general import get_bridge_values
from transformers.basic import get_basic_values


# class obj:
#     def __init__(self, dict1):
#         self.__dict__.update(dict1)


BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.absolute()

# PROMPT_CACHE_PATH = BASE_PATH / "cache/bridge_prompts"
# @TODO: Caching ignores temp and other running params

# with open(BASE_PATH / 'openai.key', 'r') as f:
#     API_KEY = f.read()


# MODEL_NAME = "gpt-4"  # ["gpt-4"]
# PROMPT_VERSION = "v_001"


def _parse_num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def parse_numbers(train, test, params):
    # (bridge, target, src)
    train_new = [(_parse_num(t[0]), t[1], t[2]) for t in train]
    test_new = [(_parse_num(t[0]), t[1], t[2]) for t in test]

    return train_new, test_new, {}




def get_bridge_table(train, test, params):
    # (bridge, target, src)
    train_new = [(None, exp[1], exp[2]) for exp in train]
    test_new, detail = get_bridge_values(train, test, params)

    return train_new, test_new, detail


def get_basic_bridge(train, test, params):
    # (bridge, target, src)
    train_new = [(None, exp[1], exp[2]) for exp in train]
    test_new, detail = get_basic_values(train, test, params)

    return train_new, test_new, detail


# def gpt4_bridge(train, test):
#     inputs = io.StringIO()
#     writer = csv.writer(inputs)
#     for row in train:
#         writer.writerow(row[:2])
#     inputs = inputs.getvalue()
#
#     test_input = io.StringIO()
#     writer2 = csv.writer(test_input)
#     for row in test:
#         writer2.writerow(row[:1])
#     test_input = test_input.getvalue()
#
#     cache_file = PROMPT_CACHE_PATH / f"{MODEL_NAME}.pkl"
#     if os.path.exists(cache_file):
#         with open(cache_file, 'rb') as fp:
#             cache_dict = pickle.load(fp)
#     else:
#         cache_dict = {}
#
#     with open(BASE_PATH / f"src/gpt4_string/prompts/gpt4/bridge_prompt_{PROMPT_VERSION}.txt") as f:
#         pmpt = f.read()
#
#     prompt = pmpt.format(inputs=inputs, inputs_outputs_full=inputs+test_input)
#
#     if prompt in cache_dict:
#         respond = cache_dict[prompt].choices[0].message.content
#
#     else:
#         client = openai.OpenAI(api_key=API_KEY, timeout=150)
#         try:
#             completion = client.chat.completions.create(
#                 model=MODEL_NAME,
#                 messages=[
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.00000001,
#                 # max_tokens=8000,
#                 # frequency_penalty=0.0
#
#             )
#             respond = completion.choices[0].message.content
#             cache_dict[prompt] = completion
#
#
#
#         except Exception as e:
#             print(str(e))
#             respond = "Failed - " + str(e)
#             dt = {'choices': [{'message': {'content': respond }}]}
#             cache_dict[prompt] = json.loads(json.dumps(dt), object_hook=obj)
#
#
#
#         with open(cache_file, 'wb') as fp:
#             pickle.dump(cache_dict, fp)
#
#
#     # print("PP Done")
#     final_tbl = inputs + respond
#
#     final_tbl = io.StringIO(final_tbl)
#     reader = csv.reader(final_tbl, delimiter=',')
#
#     dict_tbl = {}
#     for r in reader:
#         try:
#             dict_tbl[r[0].strip()] = r[1].strip()
#         except Exception as e:
#             pass
#
#     new_train = []
#     new_test = []
#     for row in train:
#         new_train.append((
#             dict_tbl.get(row[0], row[0]),
#             row[1],
#             row[0]
#         ))
#
#     for row in test:
#         new_test.append((
#             dict_tbl.get(row[0], row[0]),
#             row[1],
#             row[0]
#         ))
#
#     details = {
#         'prompt': prompt,
#         'respond': respond
#     }
#     return new_train, new_test, details
