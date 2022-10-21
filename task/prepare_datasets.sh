#!/bin/bash

mkdir -p data
cd data
# downloading udtreebanks-v2.9
mkdir -p universal_dependency
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4611{/ud-treebanks-v2.9.tgz}
tar zxvf ud-treebanks-v2.9.tgz -C universal_dependency/ud-treebanks-v2.9

# downloading XNLI and multiNLI
wget https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip
unzip XNLI-1.0.zip
wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
unzip multinli_1.0.zip

# We don't need to download wikiann as we rely on the dataset from huggingface