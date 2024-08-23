import stanza
import langcodes
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from argparse import ArgumentParser


def get_language_code(language_name):
    """Get the language code from the language name,
    so as to allow Stanza to download the correct model"""
    try:
        return langcodes.find(language_name).language
    except langcodes.LanguageTagError:
        return "unknown"  # Handle languages that can't be found


def download_and_cache_models(language_codes):
    """Download and cache Stanza models"""
    pipelines = {}
    for lang_code in language_codes:
        print(lang_code)
        try:
            stanza.download(lang_code, processors="tokenize")
            nlp_pipeline = stanza.Pipeline(lang=lang_code, processors="tokenize")
            pipelines[lang_code] = nlp_pipeline
        except Exception as e:
            print(f"Stanza does not support language code '{lang_code}'. Error: {e}")
    return pipelines


def tokenize_with_stanza(text, lang_code, pipelines):
    """tokenize text using Stanza"""
    nlp_pipeline = pipelines.get(lang_code, None)
    if nlp_pipeline:
        doc = nlp_pipeline(text)
        tokens = [token.text for sentence in doc.sentences for token in sentence.tokens]
        return tokens
    else:
        # Fallback to a simple whitespace tokenizer if Stanza is not available
        return text.split()


def main():
    tqdm.pandas()

    # Note: Each dataset is a dictionary whose keys represent the different splits!
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="CohereForAI/aya_dataset")
    parser.add_argument("--dataset_split", type=str, default="train")

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file for tokenized inputs/targets.",
    )

    args = parser.parse_args()
    split = args.dataset_split

    # Load the annotations dataset
    # # Note: The Aya collection has multiple datasets within
    # aya_collection = load_dataset("CohereForAI/aya_collection", "translated_flan_qa")
    dataset = load_dataset(args.dataset_name)
    df = pd.DataFrame(dataset[split])

    # Infer the ISO 639-1 codes for the samples of the dataframe
    df["language_code"] = df["language"].apply(get_language_code)

    # Download and cache models
    cached_pipelines = download_and_cache_models(list(df["language_code"].unique()))

    # Tokenize the inputs and targets
    df["inputs_tokens_stanza"] = df.progress_apply(
        lambda row: tokenize_with_stanza(
            row["inputs"], row["language_code"], cached_pipelines
        ),
        axis=1,
    )
    df["targets_tokens_stanza"] = df.progress_apply(
        lambda row: tokenize_with_stanza(
            row["targets"], row["language_code"], cached_pipelines
        ),
        axis=1,
    )

    # Compute the number of tokens in each input and target
    df["inputs_n_tokens_stanza"] = df["inputs_tokens_stanza"].apply(len)
    df["targets_n_tokens_stanza"] = df["targets_tokens_stanza"].apply(len)

    # Specify the rows where Stanza was used
    df["stanza_used"] = df["language_code"].apply(lambda l: l in cached_pipelines)

    # Saving the output to a CSV file
    df[
        [
            "inputs_tokens_stanza",
            "inputs_n_tokens_stanza",
            "targets_tokens_stanza",
            "targets_n_tokens_stanza",
            "stanza_used",
        ]
    ].to_csv(args.output_file, index=True)

    print(f"Aggregated results saved to {args.output_file}")


if __name__ == "__main__":
    main()
