# -*- coding: utf-8 -*-
import re
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from lid_utils import (
    GLOTILD_MODEL_NAME,
    OPENILD_MODEL_NAME,
    FASTTEXTLIDModel,
    LANGDETECTModel,
)

OUTPUT_DIR = "predictions"


def main():
    pd.set_option("display.max_rows", 150)
    pd.set_option("display.max_colwidth", None)

    tqdm.pandas()

    # Note: Each dataset is a dictionary whose keys represent the different splits!
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="CohereForAI/aya_dataset")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
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
    df[["inputs_lid", "inputs_lid_proba"]] = df["inputs"].progress_apply(
        lambda s: pd.Series(LID_model.predict(s))
    )
    df[["targets_lid", "targets_lid_proba"]] = df["targets"].progress_apply(
        lambda s: pd.Series(LID_model.predict(s))
    )

    # Store the predictions
    predictions_dir = Path(args.output_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Use three underscores to replace / in dataset name
    clean_dataset_name = re.sub("/", "___", args.dataset_name)
    clean_model_name = args.lid_model.split("-model")[0]
    output_filename = f"{clean_dataset_name}_{split}_{clean_model_name}_predictions.csv"
    output_path = predictions_dir / output_filename

    df[["inputs_lid", "inputs_lid_proba", "targets_lid", "targets_lid_proba"]].to_csv(
        str(output_path),
        index=True,
    )


if __name__ == "__main__":
    main()
