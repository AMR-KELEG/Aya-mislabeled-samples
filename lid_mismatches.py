import pandas as pd
from argparse import ArgumentParser
from datasets import load_dataset
from collections import Counter


def main():
    parser = ArgumentParser(
        description="Filter dataset rows based on language ID mismatch and output distribution of mismatches."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="CohereForAI/aya_dataset",
        help="Name of the HuggingFace dataset.",
    )
    parser.add_argument(
        "--csv_filename",
        type=str,
        required=True,
        help="CSV file with language identification results.",
    )
    parser.add_argument(
        "--inputs_lid_col",
        type=str,
        default="inputs_lid",
        help="Column name for inputs language ID.",
    )
    parser.add_argument(
        "--targets_lid_col",
        type=str,
        default="targets_lid",
        help="Column name for targets language ID.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use (e.g., train, test, validation).",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["and", "or"],
        default="and",
        help="Condition to use for mismatch filtering between the `input` field and `targets` field (can be 'and' or 'or').",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="mismatched_languages.csv",
        help="Filename for the output CSV with mismatches.",
    )
    parser.add_argument(
        "--min_inputs_length",
        type=int,
        default=0,
        help="Minimum length of the `inputs` field to keep the row.",
    )
    parser.add_argument(
        "--min_targets_length",
        type=int,
        default=0,
        help="Minimum length of the `targets` field to keep the row.",
    )

    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset_name)
    df = pd.DataFrame(dataset[args.dataset_split])

    # Load the CSV file
    lid_df = pd.read_csv(args.csv_filename)

    # Merge dataset with LID results
    m_df = pd.concat([df, lid_df], axis=1)

    # Filter out rows based on minimum length criteria
    merged_df = m_df[
        (m_df["inputs"].str.len() >= args.min_inputs_length)
        & (m_df["targets"].str.len() >= args.min_targets_length)
    ]

    # Filter rows where `language_code` does not match
    # either `inputs_lid` or `targets_lid`
    # Filter rows based on the selected condition
    if args.condition == "and":
        mismatches_df = merged_df[
            (merged_df["language_code"] != merged_df[args.inputs_lid_col])
            & (merged_df["language_code"] != merged_df[args.targets_lid_col])
        ]
    else:  # 'or' condition
        mismatches_df = merged_df[
            (merged_df["language_code"] != merged_df[args.inputs_lid_col])
            | (merged_df["language_code"] != merged_df[args.targets_lid_col])
        ]

    # Save the mismatches to a CSV file
    mismatches_df.to_csv(args.output_csv, index=False)
    print(f"Mismatches saved to {args.output_csv}")

    # Calculate and print the distribution of mismatched languages
    mismatch_distribution = Counter(mismatches_df["language_code"])
    print("\nDistribution of mismatched languages:")
    for lang, count in mismatch_distribution.items():
        print(f"{lang}: {count}")

    # Calculate per-language mismatch statistics
    languages = df["language_code"].unique()
    print("\nPer-language mismatch statistics:")
    for lang in languages:
        total_count = len(df[df["language_code"] == lang])
        mismatch_count = mismatch_distribution.get(lang, 0)
        match_count = total_count - mismatch_count
        print(f"Language: {lang}")
        print(f"  Total instances: {total_count}")
        print(f"  Matches: {match_count}")
        print(f"  Mismatches: {mismatch_count}")
        print(f"  Mismatch percentage: {mismatch_count / total_count * 100:.2f}%")


if __name__ == "__main__":
    main()
