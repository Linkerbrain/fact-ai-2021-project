#!/bin/bash

python benchmark_attack.py --data_path ~/data --transform_mode aug --aug_list "3-1-7" --epochs 200 --reaugment translate_clipped1 "$@"
#python benchmark_attack.py --data_path ~/data --transform_mode aug --aug_list "3-1-7" --epochs 200 --reaugment shiftL "$@"
#python benchmark_attack.py --data_path ~/data --transform_mode aug --aug_list "3-1-7" --epochs 200 --reaugment shiftR "$@"
#python benchmark_attack.py --data_path ~/data --transform_mode aug --aug_list "3-1-7" --shift_left --epochs 200 "$@"
#python benchmark_attack.py --data_path ~/data --transform_mode aug --aug_list "3-1-7" --shift_right --epochs 200 "$@"
#python benchmark_attack.py --data_path ~/data --transform_mode aug --aug_list "3-1-7" --moment_matching --epochs 200 "$@"
#python benchmark_attack.py --data_path ~/data --transform_mode crop --moment_matching --epochs 200 "$@"
#python benchmark_attack.py --data_path ~/data --transform_mode aug --aug_list "43-18-18" --moment_matching --epochs 200 "$@"
#python benchmark_attack.py --data_path ~/data --transform_mode aug --aug_list "3-1-7+43-18-18" --moment_matching --epochs 200 "$@"
