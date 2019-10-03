#!/bin/bash
set -e

# python ../inclearn/__main__.py --dataset cityscapes --root "/data/romilb/datasets/cityscapes" --list-name "sample_lists/citywise/aachen" --sample-incremental -e 20 --batch-size 128 --use-train-for-test
python ../inclearn/__main__.py --dataset cityscapes --root "/data/romilb/datasets/cityscapes" --list-name "sample_lists/citywise/bremen" --sample-incremental -e 5 --batch-size 128 --use-train-for-test
python ../inclearn/__main__.py --dataset cityscapes --root "/data/romilb/datasets/cityscapes" --list-name "sample_lists/citywise/dusseldorf" --sample-incremental -e 5 --batch-size 128 --use-train-for-test