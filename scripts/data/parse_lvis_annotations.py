import os
import json

import pandas as pd


def main():
    with open("lvis-annotations.json", "r") as f:
        data = json.load(f)

    num_classes = len(data.keys())
    num_files = len([k for k, v in data.items() for _ in v])
    print(f"classes: {num_classes}, files: {num_files}")

    # Save to csv with the following columns:
    # id, class_id, class_name
    ids_labels_list = []
    for class_id, (class_name, ids) in enumerate(data.items()):
        for id in ids:
            ids_labels_list.append((id, class_name, class_id))

    df = pd.DataFrame(ids_labels_list, columns=["id", "class_name", "class_id"])
    print(df.head())

    df.to_csv("lvis-annotations.csv", index=False, header=False)


if __name__ == "__main__":
    main()
