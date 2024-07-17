set -e
pip install datasets

# Install langdetect
pip install langdetect

# Install the fasttext python package
pip install git+https://github.com/facebookresearch/fastText.git
# This version of numpy is required for fasttext models!
pip install numpy==1.25.2

pip install black

# Download the openlid model
wget https://data.statmt.org/lid/lid201-model.bin.gz
gzip -d lid201-model.bin.gz

# Download the glotlid model
wget https://huggingface.co/cis-lmu/glotlid/resolve/main/model.bin -O "glotlid-model.bin"
