import pandas as pd
from datasets import ClassLabel, Dataset

def read_examples_from_file(file_path: str, label_path = "your_path") -> list:
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("DCT\t") or line == "" or line == "\n":
                if words:
                    examples.append({"labels": labels, "tokens": words})
                    words = []
                    labels = []
            else:
                splits = line.split("\t")
                words.append(splits[0])
                # print("SPLITS", splits)
                # print("word", splits[0])
                # print("labels", splits[4])
                if len(splits) > 4:
                    labels.append(splits[4].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append({"labels": labels, "tokens": words})
    for row in examples:
        assert len(row["tokens"]) == len(row["labels"])
    dataset = Dataset.from_pandas(pd.DataFrame.from_dict(examples))
    dataset.features['labels'].feature = ClassLabel(names_file=label_path)
    return dataset
