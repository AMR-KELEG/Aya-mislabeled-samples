import re
import pandas as pd
from functools import reduce

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_colwidth", None)

from datasets import load_dataset

import os
from pathlib import Path

PROJECT_DIR = Path(os.path.abspath(__file__)).parent.parent


def load_lang_predict(dataset_name: str, model_name: str, split: str):
    # TODO: This structure/names of the lang prediction files might change
    # TODO: Use the file with the ensembled lid instead of that of an individual model
    lang_pred_filename = str(
        Path(
            PROJECT_DIR,
            "predictions",
            f"{re.sub('/', '_', dataset_name)}_{split}_{model_name}_predictions.csv",
        )
    )
    lang_pred = pd.read_csv(lang_pred_filename)

    for column in ["inputs_lid", "targets_lid"]:
        lang_pred[column] = lang_pred[column].apply(lambda s: s[1:-1].split(":")[0])

    return lang_pred


if __name__ == "__main__":
    dataset_name = "CohereForAI/aya_dataset"
    split = "train"
    lang_pred_df = load_lang_predict(dataset_name, split)

    dataset = load_dataset(dataset_name)
    df = pd.DataFrame(dataset[split])
    df[["inputs_lid", "targets_lid"]] = lang_pred_df[["inputs_lid", "targets_lid"]]

    df['reason'] = ''
    df['category'] = ''

    df.loc[(df["inputs_lid"] == "eng") & (df["targets_lid"] == "hin"), 'reason'] += "hindi_translation_of_english_input, "
    df.loc[(df["inputs_lid"] == "eng") & (df["targets_lid"] == "hin"), 'category'] += "instruction_not_followed, "

    df.loc[(df["inputs_lid"] == "eng") & (df["targets_lid"] == "guj"), 'reason'] += "gujarati_translation_of_english_input, "
    df.loc[(df["inputs_lid"] == "eng") & (df["targets_lid"] == "guj"), 'category'] += "instruction_not_followed, "

    df.loc[(df['inputs'].apply(lambda x: len(x)) < 5) & (df['targets'].apply(lambda x: len(x)) < 5), 'reason'] += "input and output len too short, "
    df.loc[(df['inputs'].apply(lambda x: len(x)) < 5) & (df['targets'].apply(lambda x: len(x)) < 5), 'reason'] += "low quality examples, "

    df[df['reason'] != ''].reset_index().to_csv(f"mislabeled_samples_{re.sub('/', '_', dataset_name)}_{split}.csv")
