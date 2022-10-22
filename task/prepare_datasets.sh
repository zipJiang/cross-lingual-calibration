#!/bin/bash

mkdir -p data
cd data
# downloading udtreebanks-v2.9
mkdir -p universal_dependency
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4611{/ud-treebanks-v2.9.tgz}
tar zxvf ud-treebanks-v2.9.tgz

# also create subsampled datasets
mkdir -p subsampled_ud-treebanks
python3 ../scripts/subsample_dataset.py \
    --input_path "ud-treebanks-v2.9/UD_English-EWT/en_ewt-ud-train.conllu" \
    --num_samples 50 \
    --write_to "subsampled_ud-treebanks/subsampled_en_ewt-ud-train.conllu"

python3 ../scripts/subsample_dataset.py \
    --input_path "ud-treebanks-v2.9/UD_English-EWT/en_ewt-ud-train.conllu" \
    --num_samples 10 \
    --write_to "subsampled_ud-treebanks/subsubsampled_en_ewt-ud-train.conllu"

# downloading XNLI and multiNLI
wget https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip
unzip XNLI-1.0.zip
wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
unzip multinli_1.0.zip

# need to split the XNLI dataset to allow sampling
python3 ../scripts/split_xnli_by_language.py  --file_path "XNLI-1.0/xnli.test.jsonl" --dump_dir "xnli-test"

# We don't need to download wikiann as we rely on the dataset from huggingface