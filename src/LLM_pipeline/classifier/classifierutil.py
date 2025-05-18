import json
import pathlib

ALLOWED_CLASSES = ("String", "General", "Numbers", "Algorithmic")

BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute()
CODE_BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.absolute()
DFX_CLASSES_PATH = str(CODE_BASE_PATH / "classifier" / "DFX_classes.csv")
TDE_CLASSES_PATH = str(CODE_BASE_PATH / "classifier" / "TDE_classes.csv")

ALL_CLASSES_JSON = str(BASE_PATH / "data/Classes/gpt_classified.json")


DFX_CLASSES = {}
TDE_CLASSES = {}


def get_gold_label(tbl_name, ds_path):
    ds_name = pathlib.Path(ds_path).name


    if ds_name in ("AutoJoin", "FlashFill"):
        return "String"

    elif ds_name == "DataXFormer":
        if len(DFX_CLASSES) < 80:
            with open(DFX_CLASSES_PATH, 'r') as f:
                lines = f.readlines()
            rows = [line.strip().split(',') for line in lines]
            for row in rows:
                assert len(row) == 2
                assert row[1] in ALLOWED_CLASSES
                DFX_CLASSES[row[0]] = row[1]

        return DFX_CLASSES[tbl_name]


    elif ds_name == "All_TDE":
        if len(TDE_CLASSES) < 229:
            with open(TDE_CLASSES_PATH, 'r') as f:
                lines = f.readlines()
            rows = [line.strip().split(',') for line in lines]
            for row in rows:
                assert row[1] in ALLOWED_CLASSES
                TDE_CLASSES[row[0]] = row[1]

        return TDE_CLASSES[tbl_name]

    else:
        raise NotImplementedError(f"The {ds_name} dataset has no golden labels")




def get_gpt_label(tbl_name, ds_path):
    with open(ALL_CLASSES_JSON, 'r') as f:
        labels_dict = json.load(f)

    return labels_dict[tbl_name]['predicted_value']
