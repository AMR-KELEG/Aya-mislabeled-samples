#!/usr/bin/env python
# coding: utf-8
import pandas as pd

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_colwidth", None)
from tqdm import tqdm

tqdm.pandas()

from datasets import load_dataset

penlid_pred = pd.read_csv(
    "predictions/CohereForAI_aya_dataset_train_lid201_predictions.csv"
)

dataset = load_dataset("CohereForAI/aya_dataset")
df = pd.DataFrame(dataset["train"])

# TODO: Use a better tokenizer than whitespace splitting especially for languages like Mandarin!
df["inputs_n_tokens"] = df["inputs"].apply(lambda s: len(s.split()))
df["targets_n_tokens"] = df["targets"].apply(lambda s: len(s.split()))

# TODO: Make use of the probs?
for column in ["inputs", "targets"]:
    df[f"{column}_lid"] = openlid_pred[f"{column}_lid"].apply(
        lambda s: s[1:-1].split(":")[0]
    )

stats_df = (
    df.groupby("language_code")
    .apply(lambda gdf: gdf[["inputs_lid", "targets_lid"]].value_counts(normalize=True))
    .reset_index()
)

stats_df.columns = ["Aya_language_code", "input_lid", "target_lid", "percentage "]

stats_df["n_samples"] = stats_df.progress_apply(
    lambda row: df[
        (df["language_code"] == row["Aya_language_code"])
        & (df["inputs_lid"] == row["input_lid"])
        & (df["targets_lid"] == row["target_lid"])
    ].shape[0],
    axis=1,
)

stats_df["total_n_samples_for_language"] = stats_df.progress_apply(
    lambda row: df[(df["language_code"] == row["Aya_language_code"])].shape[0],
    axis=1,
)

stats_df.to_csv("aya_dataset_lid_stats.csv", index=False)

# Manual inspection of potentially problematic samples!

# df[(df["language_code"]=="hin") & (df["inputs_lid"]=="eng")].head(n=1)
# df[(df["language_code"]=="ibo") & (df["inputs_lid"]=="kin")].head(n=1)
# df[(df["language_code"]=="xho") & (df["inputs_lid"]=="eng")].head(n=1)
# df[(df["language_code"]=="zul") & (df["inputs_n_tokens"]<3) & (df["targets_n_tokens"]<3)]
