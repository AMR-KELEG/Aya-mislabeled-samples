import stanza
import langcodes
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
tqdm.pandas()
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_colwidth", None)

dataset = load_dataset("CohereForAI/aya_dataset")
df = pd.DataFrame(dataset["train"])
df.drop_duplicates(inplace=True)


# TODO: correct the file path
openlid_pred = pd.read_csv(
    "predictions/CohereForAI_aya_dataset_train_lid201_predictions.csv"
)

for column in ["inputs", "targets"]:
    df[f"{column}_lid"] = openlid_pred[f"{column}_lid"].apply(lambda s: s[1:-1].split(":")[0])


"""Define a function to get the language code from the language name,
so as to allow Stanza to download the correct model"""
def get_language_code(language_name):
    try:
        return langcodes.find(language_name).language
    except langcodes.LanguageTagError:
        return 'unknown'  # Handle languages that can't be found


# Define functions to download and cache Stanza models
def download_and_cache_models(language_codes):
    pipelines = {}
    for lang_code in language_codes:
        try:
            stanza.download(lang_code)
            nlp_pipeline = stanza.Pipeline(lang=lang_code, processors='tokenize')
            pipelines[lang_code] = nlp_pipeline
        except Exception as e:
            print(f"Stanza does not support language code '{lang_code}'. Error: {e}")
    return pipelines

# Define a function to tokenize text using Stanza
# TODO: modify this to create a new column with the result, "stanza_used" or "not_stanza_used"
def tokenize_with_stanza(text, lang_code, pipelines):
    nlp_pipeline = pipelines.get(lang_code, None)
    if nlp_pipeline:
        doc = nlp_pipeline(text)
        tokens = [token.text for sentence in doc.sentences for token in sentence.tokens]
        return tokens
    else:
        # Fallback to a simple whitespace tokenizer if Stanza is not available
        return text.split()

# Define a function to apply tokenization to each row in the DataFrame
def apply_tokenization(row, pipelines):
    lang_code = row['language_code']
    row['inputs_n_tokens_stanza'] = len(tokenize_with_stanza(row['inputs'], lang_code, pipelines))
    row['targets_n_tokens_stanza'] = len(tokenize_with_stanza(row['targets'], lang_code, pipelines))
    return row


""" NB: The function below is used to just filter out the dataframe for checking the shortprompts
and not used in the pipeline"""
# Define a function to return only those rows where Stanza was used.
def was_stanza_used(row):
    lang_code = row['language_code']
    nlp_pipeline = cached_pipelines.get(lang_code, None)
    # If Stanza was used, the token count should be different from a simple split
    return (
        nlp_pipeline is not None and 
        row['inputs_n_tokens_stanza'] != len(row['inputs'].split()) and
        row['targets_n_tokens_stanza'] != len(row['targets'].split())
    )


"""Runnning the pipeline and scripts on the dataframe.
The final output is a CSV file with the ideal short prompts."""

df['language_code'] = df['language'].apply(get_language_code) # Applying the language code function to the DataFrame

# Download and cache models
cached_pipelines = download_and_cache_models(list(df['language_codes'].unique()))

# Apply tokenization to the DataFrame
df = df.apply(lambda row: apply_tokenization(row, cached_pipelines), axis=1)

# Apply the was_stanza_used function to filter rows where Stanza was used and
# get get rows where Stanza was used and token counts are <= 5"""
df['stanza_used'] = df.apply(was_stanza_used, axis=1)
ideal_short_promts = df[
    (df['stanza_used']) &
    ((df['inputs_n_tokens_stanza'] <= 5) | (df['targets_n_tokens_stanza'] <= 5)) # using the 'or' criteria
]

# Droping the 'stanza_used' column as it's no longer needed
ideal_short_promts = ideal_short_promts.drop(columns=['stanza_used'])
ideal_short_promts.to_csv('ideal_short_promts.csv', index=False) # Saving the output to a CSV file
