#!/bin/bash -ex

export TF_CPP_MIN_LOG_LEVEL=2

time python3 ../code/train.py --epochs 10
