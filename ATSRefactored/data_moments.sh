#!/bin/bash

python data_moments.py --data_path ~/data --transform_mode crop "$@"
python data_moments.py --data_path ~/data --transform_mode aug --aug_list "3-1-7" "$@"
python data_moments.py --data_path ~/data --transform_mode aug --aug_list "43-18-18" "$@"
python data_moments.py --data_path ~/data --transform_mode aug --aug_list "3-1-7+43-18-18" "$@"
