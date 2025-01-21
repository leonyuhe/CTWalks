#!/usr/bin/env bash

cd ..

python main.py -d mooc --data_usage 1.0 --mode i --n_degree 64 1 --pos_dim 108 --pos_sample multinomial --pos_enc lp \
--negs 1 --solver rk4 --step_size 0.125 \
--bs 32 --gpu 0 --seed 0 --limit_ngh_span --ngh_span 320 8

