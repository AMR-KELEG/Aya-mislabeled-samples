import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from collections import Counter
from tqdm import tqdm


def load_prediction_files(predictions_dir):
    # Load all CSV files in the predictions directory
    prediction_files = Path(predictions_dir).glob("*.csv")
    dataframes = [pd.read_csv(file, index_col=0) for file in prediction_files]
    return dataframes


def filter_by_probability(lid_column, proba_column, threshold):
    return lambda row: row[lid_column] if row[proba_column] >= threshold else None


def aggregate_predictions(
    dataframes,
    inputs_col_pattern,
    targets_col_pattern,
    inputs_proba_thr,
    targets_proba_thr,
):
    # Concatenate all dataframes along the columns axis,
    # make sure the columns are unique
    concatenated_df = pd.concat(
        [df.add_suffix(f"_{i}") for i, df in enumerate(dataframes)], axis=1
    )

    # Extract language and probability columns
    inputs_lid_columns = [
        col
        for col in concatenated_df.columns
        if inputs_col_pattern in col and "proba" not in col
    ]
    inputs_proba_columns = [
        col
        for col in concatenated_df.columns
        if inputs_col_pattern in col and "proba" in col
    ]
    targets_lid_columns = [
        col
        for col in concatenated_df.columns
        if targets_col_pattern in col and "proba" not in col
    ]
    targets_proba_columns = [
        col
        for col in concatenated_df.columns
        if targets_col_pattern in col and "proba" in col
    ]

    # Filter out languages based on probability thresholds
    for lid_col, proba_col in zip(inputs_lid_columns, inputs_proba_columns):
        concatenated_df[lid_col] = concatenated_df.apply(
            filter_by_probability(lid_col, proba_col, inputs_proba_thr), axis=1
        )

    for lid_col, proba_col in zip(targets_lid_columns, targets_proba_columns):
        concatenated_df[lid_col] = concatenated_df.apply(
            filter_by_probability(lid_col, proba_col, targets_proba_thr), axis=1
        )

    # Function to get the most common language ignoring None values
    def most_common_language(languages):
        languages = [lang for lang in languages if lang is not None]
        return Counter(languages).most_common(1)[0][0] if languages else None

    # Apply the aggregation to each row
    concatenated_df["aggregated_inputs_lid"] = concatenated_df[
        inputs_lid_columns
    ].progress_apply(most_common_language, axis=1)
    concatenated_df["aggregated_targets_lid"] = concatenated_df[
        targets_lid_columns
    ].progress_apply(most_common_language, axis=1)

    # Select only the aggregated results
    aggregated_df = concatenated_df[["aggregated_inputs_lid", "aggregated_targets_lid"]]
    return aggregated_df


def save_aggregated_results(aggregated_df, output_file):
    aggregated_df.to_csv(output_file, index=True)


def main():
    tqdm.pandas()

    parser = ArgumentParser()
    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Directory containing model prediction CSV files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file for aggregated results.",
    )
    parser.add_argument(
        "--inputs_col_pattern",
        type=str,
        default="inputs_lid",
        help="Pattern to match input LID columns.",
    )
    parser.add_argument(
        "--targets_col_pattern",
        type=str,
        default="targets_lid",
        help="Pattern to match target LID columns.",
    )
    parser.add_argument(
        "--inputs_proba_thr",
        type=float,
        default=0.0,
        help="Threshold for input LID probabilities.",
    )
    parser.add_argument(
        "--targets_proba_thr",
        type=float,
        default=0.0,
        help="Threshold for target LID probabilities.",
    )

    args = parser.parse_args()

    dataframes = load_prediction_files(args.predictions_dir)
    aggregated_df = aggregate_predictions(
        dataframes,
        args.inputs_col_pattern,
        args.targets_col_pattern,
        args.inputs_proba_thr,
        args.targets_proba_thr,
    )
    save_aggregated_results(aggregated_df, args.output_file)

    print(f"Aggregated results saved to {args.output_file}")


if __name__ == "__main__":
    main()
