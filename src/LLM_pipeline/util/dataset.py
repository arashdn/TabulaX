import csv
import os
import random

RANDOM_SEED = 12345


def get_pairs_from_files(ds_path, tbl_names=[]):
    assert os.path.isdir(ds_path)
    dirs = [dI for dI in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, dI))]

    res = {}
    res['inputs'] = {}

    for dir in dirs:
        if len(tbl_names) > 0 and dir not in tbl_names:
            continue

        ds_dir = ds_path+'/' + dir
        # assert os.path.exists(ds_dir + "/source.csv")
        # assert os.path.exists(ds_dir + "/target.csv")
        assert os.path.exists(ds_dir + "/rows.txt")
        assert os.path.exists(ds_dir + "/ground truth.csv")

        src_col, target_col = "", ""

        with open(ds_dir + "/rows.txt") as f:
            l = f.readline().strip().split(':')
            src_col = l[0]
            target_col = l[1]
            direction = f.readline().strip()


        pairs = []

        with open(ds_dir + "/ground truth.csv", newline='') as f:
            reader = csv.reader(f)
            titles = next(reader)

            if not "source-" + src_col in titles:
                print(ds_dir)

            assert "source-" + src_col in titles
            assert "target-" + target_col in titles

            src_idx = titles.index("source-" + src_col)
            target_idx = titles.index("target-" + target_col)

            if direction.lower() == "target":
                src_idx, target_idx = target_idx, src_idx

            for items in reader:
                pairs.append((items[src_idx], items[target_idx]))

        res['inputs'][dir] = pairs


    return res


def sample_data(ds_path, example_size, example_size_type="fixed"):
    pairs = get_pairs_from_files(ds_path, [])

    tables = dict()

    for table, rows in pairs['inputs'].items():
        # print(f"working on {table}")
        random.seed(RANDOM_SEED)
        random.shuffle(rows)
        # print(rows[1])

        if example_size_type == "fixed":
            train_size = min(example_size, len(rows) - 1)
        else:
            raise NotImplementedError
            train_size = max(2, len(rows) * example_size)

        tables[table] = {
            'train': rows[:train_size],
            'test': rows[train_size:],
        }

    return tables

