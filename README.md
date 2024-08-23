## Setting up the envionment

`conda create -n aya python=3.10.12`

`conda activate aya`

Install `poetry` as described in <https://python-poetry.org/docs/#installation>
and run `poetry install`

`./install.sh`

## Running the code saves the whole output to a csv file

`poetry run python lid.py --lid_model glotlid-model.bin --dataset_split train`

## "Ensembling" the results

To produce an ensemble (via majority voting) of various outputs (in our
example stored in the `predictions/`) folder, one can run the following:

    poetry run python ensemble.py --predictions_dir predictions/ --output_file ensemble.csv

Which will then save the resulting ensemble to the `ensemble.csv` file.

The script also supports a threshold setting, which will make it so
predictions with lower probability than the threshold will get ignored.

For instance, to only consider language identification predictions
whose probability is over 0.75, one can run:

     poetry run python ensemble.py --predictions_dir predictions/ --output_file ensemble-i-0.75-t-0.75.csv --inputs_proba_thr 0.75 --targets_proba_thr 0.75

## Tokenize the data using stanza
`poetry run python stanza_tokenize.py --output_file tokenized_data.csv`
