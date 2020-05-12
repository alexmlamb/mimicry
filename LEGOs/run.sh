#!/bin/sh

module load python/3.6

python3 src/model/main.py --model Resnet18_MLP --path ~/scratch/data/LEGOs/Datasets/ --use_nfl

