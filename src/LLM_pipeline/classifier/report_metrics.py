import json
import pathlib

BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute()

def report_metrics(json_file):
    with open(json_file, 'r') as f:
        classes = json.load(f)

    y_true, y_pred = [], []

    tables = {}

    for _, cls in classes.items():
        ds_name, tbl_name = cls['full_name'].split("/")
        pred = cls['predicted_value']
        gold = cls['golden_value']


        if pred != gold:
            print(f"{cls['full_name']}: pred: {pred}, gold: {gold}")

        if ds_name not in tables:
            tables[ds_name] = {'y_pred': [], 'y_true': []}

        tables[ds_name]['y_pred'].append(pred)
        tables[ds_name]['y_true'].append(gold)
        y_true.append(gold)
        y_pred.append(pred)

    from sklearn.metrics import classification_report, accuracy_score

    # Compute classification report
    report = classification_report(y_true, y_pred, output_dict=True)

    # Print classification report for each class
    print("Classification Report:")
    print(classification_report(y_true, y_pred))



if __name__ == "__main__":
    report_metrics(str(BASE_PATH / "data/Classes/gpt_classified.json"))
