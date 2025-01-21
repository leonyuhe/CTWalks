#!/usr/bin/env bash

cd ..

python main.py -d TaobaoSmall --data_usage 0.02 --mode t --n_degree 32 1 --pos_dim 108 --pos_sample multinomial \
--pos_enc saw  --negs 8 --solver rk4 --step_size 0.125 \
--bs 32 --gpu 0 --seed 0 --cpu_cores 1
