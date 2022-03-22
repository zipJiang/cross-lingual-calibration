#!/bin/bash

# usage batch_rename.sh ori_stem new_stem
ori_stem=
new_stem=

ori_stem=$1
new_stem=$2

if [[ -z ${ori_stem} ]]; then
    echo "Usage batch_rename.sh ori_stem new_stem"
    exit -1
fi

for name in ./${ori_stem}*;
do
    mv ${name} ${name/${ori_stem}/${new_stem}}
done