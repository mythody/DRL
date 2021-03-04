#!/bin/bash
python3 run_grasp_learning_exp.py \
--obs_type RGB \
--add_obs None \
--cam_view_option 0 \
--exp_name sac_off_policy \
--num_iterations 500000 \
--eval_every_n_iterations 50000 \
--initial_memory True \
--initial_memory_path /work/silvester/grasp_data/ \

