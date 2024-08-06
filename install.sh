#!/bin/bash
set -euo pipefail

# Download the openlid model
wget https://data.statmt.org/lid/lid201-model.bin.gz
gzip -d lid201-model.bin.gz

# Download the glotlid model
wget https://huggingface.co/cis-lmu/glotlid/resolve/main/model.bin -O "glotlid-model.bin"
