import re
import pandas as pd
from functools import reduce

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_colwidth", None)
from tqdm import tqdm

tqdm.pandas()
from datasets import load_dataset

import os
from pathlib import Path

PROJECT_DIR = Path(os.path.abspath(__file__)).parent.parent


def load_lang_predict(dataset_name, split):
    # TODO: This structure/names of the lang prediction files might change
    # TODO: Use the file with the ensembled lid instead of that of an individual model
    lang_pred_filename = str(
        Path(
            PROJECT_DIR,
            "predictions",
            f"{re.sub('/', '_', dataset_name)}_{split}_lid201_predictions.csv",
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
    for column in ["inputs_lid", "targets_lid"]:
        df[column] = lang_pred_df[column]

    mislabeled_samples_dfs = []

    # Category `instruction_not_followed`
    instruction_not_followed_ids = df[
        (df["inputs_lid"] == "eng") & (df["targets_lid"] == "hin")
    ].index.tolist()

    mislabeled_samples_dfs.append(
        df.loc[instruction_not_followed_ids].copy(deep=True).reset_index()
    )
    category = "instruction_not_followed"
    mislabeled_samples_dfs[-1]["category"] = category
    mislabeled_samples_dfs[-1]["comment"] = "hindi_translation_of_english_input"

    # Category `mistranslation`
    mistranslation_ids = df[
        (df["inputs_lid"] == "eng") & (df["targets_lid"] == "guj")
    ].index.tolist()
    mislabeled_samples_dfs.append(
        df.loc[mistranslation_ids].copy(deep=True).reset_index()
    )
    category = "instruction_not_followed"
    mislabeled_samples_dfs[-1]["category"] = category
    mislabeled_samples_dfs[-1]["comment"] = "gujarati_translation_of_english_input"

    # TODO: Another joining/merging functions is needed because the same sample might be flagged multiple times for different reasons
    # So, it is better to merge the dataframes based on the index instead of concatenating them
    concat_mislabeled_samples_df = pd.concat(mislabeled_samples_dfs)
    concat_mislabeled_samples_df.sort_values(by="index", inplace=True)

    # TODO: Upload the mislabeled samples to hf directly?
    # TODO: But, inspection is needed so might be better to do so in a different script
    concat_mislabeled_samples_df.to_csv(
        f"mislabeled_samples_{re.sub('/', '_', dataset_name)}_{split}.csv", index=False
    )
