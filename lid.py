# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

from datasets import load_dataset

pd.set_option("display.max_rows", 150)
pd.set_option("display.max_colwidth", None)

from argparse import ArgumentParser
from lid_utils import (
    FASTTEXTLIDModel,
    LANGDETECTModel,
    GLOTILD_MODEL_NAME,
    OPENILD_MODEL_NAME,
)


def main():
    # Note: Each dataset is a dictionary whose keys represent the different splits!

    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="CohereForAI/aya_dataset")
    parser.add_argument(
        "--lid_model",
        type=str,
        choices=[GLOTILD_MODEL_NAME, OPENILD_MODEL_NAME, "langdetect"],
        required=True,
    )
    parser.add_argument("--dataset_split", type=str, default="train")

    args = parser.parse_args()
    split = args.dataset_split

    LID_model = (
        FASTTEXTLIDModel(model_bin_path=args.lid_model)
        if args.lid_model != "langdetect"
        else LANGDETECTModel()
    )

    # Load the annotations dataset
    # # Note: The Aya collection has multiple datasets within
    # aya_collection = load_dataset("CohereForAI/aya_collection", "translated_flan_qa")
    dataset = load_dataset(args.dataset_name)
    df = pd.DataFrame(dataset[split])

    # Perform the LID
    # TODO: The names of the columns change from a dataset to another
    df["inputs_lid"] = df["inputs"].progress_apply(lambda s: LID_model.predict(s))
    df["targets_lid"] = df["targets"].progress_apply(lambda s: LID_model.predict(s))

    # Store the predictions
    OUTPUT_DIR = "predictions"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df[["inputs_lid", "targets_lid"]].to_csv(
        str(
            Path(
                OUTPUT_DIR,
                f"{re.sub('/', '_', args.dataset_name)}_{split}_{args.lid_model.split('-model')[0]}_predictions.csv",
            )
        ),
        index=True,
    )


if __name__ == "__main__":
    main()
