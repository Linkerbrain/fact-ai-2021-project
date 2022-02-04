#!/bin/bash

python benchmark_attack.py --data_path ~/data --transform_mode aug --aug_list "3-1-7" --epochs 200 --reaugment translate_clipped1 --optim inversed --recalculate "$@"
python benchmark_attack.py --data_path ~/data --transform_mode aug --aug_list "43-18-18" --epochs 200 --reaugment translate_clipped2 --optim inversed --recalculate "$@"
python benchmark_attack.py --data_path ~/data --transform_mode aug --aug_list "3-1-7+43-18-18" --epochs 200 --reaugment translate_clipped3 --optim inversed --recalculate "$@"
