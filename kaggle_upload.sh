#!/bin/bash -ex
if [ ! -d ../miccai ]
then
    echo "kaggle_init.sh must be run"
    exit
fi
cd ../miccai
for file in dataset.py util.py model.py predict.py README.md
do
    cp ../code/$file .
done

rsync -av ../output/*.pth .
cp ../code/dataset-metadata.json .
kaggle datasets version -m "New version"
