#!/bin/bash -ex

mkdir -p ../miccai
cd ../miccai
kaggle datasets init
cp ../code/dataset-metadata.json .
cp ../code/README.md .
kaggle datasets create
