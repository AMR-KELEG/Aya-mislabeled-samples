## Setting up the envionment
`conda create -n aya python=3.10.12`

`conda activate aya`

Install `poetry` as described in https://python-poetry.org/docs/#installation and run `poetry install`

`./install.sh`

## Running the code saves the whole output to a csv file
`poetry run python lid.py --lid_model glotlid-model.bin --dataset_split train`
