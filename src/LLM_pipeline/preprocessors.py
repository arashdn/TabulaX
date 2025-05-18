
import pathlib
from sys import stderr

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
        try:
            return float(s)
        except ValueError:
            print(f"Error Parsing Numbers!: {s}", file=stderr)
            return 0


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

