#!/bin/bash
python3 run_grasp_learning_exp.py \
--obs_type RGB \
--add_obs None \
--cam_view_option 0 \
--exp_name sac_on_policy \
--num_iterations 500000 \
--eval_every_n_iterations 1000 \
--n_episodes_per_eval 50 \
--on_policy True 
