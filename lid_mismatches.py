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
        "--lid_csv_filename",
        type=str,
        required=True,
        help="CSV file with language identification results.",
    )
    parser.add_argument(
        "--tokenized_csv_filename",
        type=str,
        required=True,
        help="CSV file with tokenized inputs and targets.",
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
    parser.add_argument(
        "--clean_samples_csv",
        type=str,
        default="lid_clean_samples.csv",
        help="Filename for the output CSV with clean samples where LIDs match the assigned language.",
    )
    parser.add_argument(
        "--duplicated_samples_csv", type=str, default="lid_duplicated_samples.csv"
    )
    parser.add_argument(
        "--short_samples_csv", type=str, default="lid_short_samples.csv"
    )
    parser.add_argument(
        "--wrong_language_csv",
        type=str,
        default="lid_wrong_language_samples.csv",
        help="Filename for the output CSV with samples where both LIDs are wrong.",
    )
    parser.add_argument(
        "--mismatch_samples_csv",
        type=str,
        default="lid_mismatch_samples.csv",
        help="Filename for the output CSV with samples where either the input or target LID differs from the assigned language.",
    )

    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset_name)
    df = pd.DataFrame(dataset[args.dataset_split])

    # Load the tokenized CSV file
    tokenized_df = pd.read_csv(args.tokenized_csv_filename)

    # Load the lid CSV file
    lid_df = pd.read_csv(args.lid_csv_filename)

    # Merge dataset with LID results
    m_df = pd.concat([df, lid_df, tokenized_df], axis=1)

    # 1. Drop duplicates
    duplicated_df = m_df[m_df.duplicated(subset=["inputs", "targets"], keep="first")]
    duplicated_df.to_csv(args.duplicated_samples_csv, index=False)
    print(f"Duplicated samples saved to {args.duplicated_samples_csv}")
    m_df.drop_duplicates(subset=["inputs", "targets"], keep="first", inplace=True)

    # 2. Filter out rows based on minimum length criteria
    short_df = m_df[
        (m_df["inputs_n_tokens_stanza"] < 4)
        & (m_df["targets_n_tokens_stanza"] < 4)
        & (df["inputs"].apply(lambda s: not any([s.strip().endswith(c) for c in "?؟"])))
    ]
    short_df.to_csv(args.short_samples_csv, index=False)
    print(f"Short samples saved to {args.short_samples_csv}")
    m_df.drop(short_df.index, inplace=True)

    # Filter out rows based on minimum length criteria
    merged_df = m_df[
        (m_df["inputs"].str.len() >= args.min_inputs_length)
        & (m_df["targets"].str.len() >= args.min_targets_length)
    ]

    # 1. Generate clean samples where LIDs match the assigned language
    clean_samples_df = merged_df[
        (merged_df["language_code"] == merged_df[args.inputs_lid_col])
        & (merged_df["language_code"] == merged_df[args.targets_lid_col])
    ]
    clean_samples_df.to_csv(args.clean_samples_csv, index=False)
    print(f"Clean samples saved to {args.clean_samples_csv}")

    # 2. Generate samples where both LIDs are wrong
    wrong_language_df = merged_df[
        (merged_df["language_code"] != merged_df[args.inputs_lid_col])
        & (merged_df["language_code"] != merged_df[args.targets_lid_col])
    ]
    wrong_language_df.to_csv(args.wrong_language_csv, index=False)
    print(f"Samples with both LIDs wrong saved to {args.wrong_language_csv}")

    # 3. Generate samples with mismatched LIDs (either input or target differs from assigned language)
    mismatch_samples_df = merged_df[
        (merged_df["language_code"] != merged_df[args.inputs_lid_col])
        ^ (merged_df["language_code"] != merged_df[args.targets_lid_col])
    ]
    mismatch_samples_df.to_csv(args.mismatch_samples_csv, index=False)
    print(f"Mismatched LID samples saved to {args.mismatch_samples_csv}")

    # Calculate and print the distribution of mismatched languages
    mismatch_distribution = Counter(mismatch_samples_df["language_code"])
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
