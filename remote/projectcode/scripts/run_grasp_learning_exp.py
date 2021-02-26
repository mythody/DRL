import os
import time
import sys

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = CURR_DIR + '/../..'
sys.path.append(PACKAGE_DIR)
print(PACKAGE_DIR)

from projectcode.infrastructure.rl_trainer import RL_Trainer
from projectcode.agents.dqn_agent import DQNAgent
from projectcode.agents.sac_agent import SACAgent


class Q_Trainer(object):

    def __init__(self, params):
        self.params = params

        train_args = {
            'LEARNING_RATE': 1e-4,
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
            'train_batch_size': params['BATCH_SIZE'],
            'double_q': params['double_q'],
        }

        env_args = {'STACK_SIZE': 1,
                    'MEMORY_CAPACITY': 1000,
                    }

        epsilon_greedy_args = {
            'GAMMA': 0.99,
            'EPS_START': 0.9,
            'EPS_END': 0.1,
            'EPS_DECAY': 200,
            'EPS_DECAY_LAST_FRAME': 10 ** 4,
            'TARGET_UPDATE': 1000,
        }

        self.agent_params = {**train_args, **env_args, **params, **epsilon_greedy_args} ### argsis
        self.params['agent_class'] = SACAgent
        self.params['agent_params'] = self.agent_params
        self.params['train_batch_size'] = params['BATCH_SIZE']

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(self.agent_params['num_iterations'])

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env_name',
        default='MsPacman-v0',
        choices=('PongNoFrameskip-v4', 'LunarLander-v3', 'MsPacman-v0')
    )
    ###################
    ## Here are the interesting arguments

    parser.add_argument('--debug_mode', type=bool, default=True)

    ## own params
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--obs_type', type=str, default='RGB', choices=('BW', 'RGB', 'depth', 'segmentation'))
    parser.add_argument('--add_obs', type=str, default='None', choices=('None', 'JointsState', 't-steps'))
    parser.add_argument('--cam_view_option', type=int, default=0) # 0: default fixed cam, 1: cam=endeffector without rotation, 2: endeffector with rotation
    parser.add_argument('--obs_size', type=int, default=64)
    #########################################################

    #parser.add_argument('--collect_data_every_n_iterations', type=int, default=1000)
    parser.add_argument('--num_iterations', type=int, default=10000)  #10000
    #parser.add_argument('--n_episodes_collected_per_iteration', type=int, default=50)
    parser.add_argument('--BATCH_SIZE', type=int, default=2) #32

    parser.add_argument('--eval_every_n_iterations', type=int, default=1000) #1000
    parser.add_argument('--n_episodes_per_eval', type=int, default=2)  #50
    parser.add_argument('--log_loss_frequ', type=int, default=5)  #50

    parser.add_argument('--on_policy', type=bool, default=True)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--MaxSteps', type=int, default=15)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--eval_batch_size', type=int, default=20)


    ### Unused params so far
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--double_q', action='store_true')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--scalar_log_freq', type=int, default=int(1e4))
    parser.add_argument('--video_log_freq', type=int, default=-1)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)
    params['video_log_freq'] = -1 # This param is not used for DQN


    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = 'drl_grasp__' + args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    trainer = Q_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
