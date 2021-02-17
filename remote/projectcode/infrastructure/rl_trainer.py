from collections import OrderedDict
import pickle
import sys
import pdb
import torch

from tensorboardX import SummaryWriter
import timeit
from itertools import count
from projectcode.envs.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from projectcode.infrastructure.utils import *
import pybullet as p

import collections


# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the projectcode.below


class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params

        '''
        #self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############

        if 'env_wrappers' in self.params:
            # These operations are currently only for Atari envs
            self.env = wrappers.Monitor(self.env, os.path.join(self.params['logdir'], "gym"), force=True)
            self.eval_env = wrappers.Monitor(self.eval_env, os.path.join(self.params['logdir'], "gym"), force=True)
            self.env = params['env_wrappers'](self.env)
            self.eval_env = params['env_wrappers'](self.eval_env)
            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')
        
        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2

        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes

        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep
        elif 'env_wrappers' in self.params:
            self.fps = 30 # This is not actually used when using the Monitor wrapper
        elif 'video.frames_per_second' in self.env.env.metadata.keys():
            self.fps = self.env.env.metadata['video.frames_per_second']
        else:
            self.fps = 10

        '''
        img_size = self.params['obs_size']
        env = KukaDiverseObjectEnv(renders=False, isDiscrete=True, removeHeightHack=False, maxSteps=20, width=img_size,
                                   height=img_size)
        env._cam_view_option = self.params['cam_view_option']
        env.cid = p.connect(p.DIRECT)
        obs = env.reset()

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(env, self.params['agent_params'])  #self.env, self.params['agent_params'] as arguments for agent

    def run_training_loop(self, n_iter=None, collect_policy=None, eval_policy=None,
                          buffer_name=None,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        num_episodes = self.agent.params['num_episodes'] #1000
        writer = SummaryWriter(self.params['logdir'])
        total_rewards = []
        ten_rewards = 0
        best_mean_reward = None
        start_time = timeit.default_timer()
        PATH = 'policy_dqn.pt'

        env = self.agent.env
        STACK_SIZE = self.agent.params['STACK_SIZE']

        for i_episode in range(num_episodes):
            # Initialize the environment and state
            if (i_episode % 10 == 0):
                print("Episode: ", i_episode)
            env.reset()
            state = self.agent.get_screen()
            stacked_states = collections.deque(STACK_SIZE * [state], maxlen=STACK_SIZE)
            for t in count():
                stacked_states_t = torch.cat(tuple(stacked_states), dim=1)
                # Select and perform an action
                action = self.agent.select_action(stacked_states_t, i_episode)
                _, reward, done, _ = env.step(action.item())
                reward = torch.tensor([reward], device=self.agent.device)

                # Observe new state
                next_state = self.agent.get_screen()
                if not done:
                    next_stacked_states = stacked_states
                    next_stacked_states.append(next_state)
                    next_stacked_states_t = torch.cat(tuple(next_stacked_states), dim=1)
                else:
                    next_stacked_states_t = None

                # Store the transition in memory
                self.agent.memory.push(stacked_states_t, action, next_stacked_states_t, reward)

                # Move to the next state
                stacked_states = next_stacked_states

                # Perform one step of the optimization (on the target network)
                self.agent.optimize_model()
                if done:
                    reward = reward.cpu().numpy().item()
                    ten_rewards += reward
                    total_rewards.append(reward)
                    mean_reward = np.mean(total_rewards[-100:]) * 100
                    writer.add_scalar("epsilon", self.agent.eps_threshold, i_episode)
                    if (best_mean_reward is None or best_mean_reward < mean_reward) and i_episode > 100:
                        # For saving the model and possibly resuming training
                        torch.save({
                            'policy_net_state_dict': self.agent.policy_net.state_dict(),
                            'target_net_state_dict': self.agent.target_net.state_dict(),
                            'optimizer_policy_net_state_dict': self.agent.optimizer.state_dict()
                        }, PATH)
                        if best_mean_reward is not None:
                            print(
                                "Best mean reward updated %.1f -> %.1f, model saved" % (best_mean_reward, mean_reward))
                        best_mean_reward = mean_reward
                    break

            if i_episode % 10 == 0:
                writer.add_scalar('train 10 episodes mean rewards', ten_rewards / 10.0, i_episode)
                if(best_mean_reward is not None):
                    writer.add_scalar('train_best_mean_reward', best_mean_reward, i_episode)
                ten_rewards = 0
            if i_episode % 100 == 0:
                self.eval_agent(writer)
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.agent.params['TARGET_UPDATE'] == 0:
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
            if i_episode >= 200 and mean_reward > 50:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode + 1, mean_reward))
                break

        print('Average Score: {:.2f}'.format(mean_reward))
        elapsed = timeit.default_timer() - start_time
        print("Elapsed time: {}".format(timedelta(seconds=elapsed)))
        writer.close()
        env.close()


    def eval_agent(self, writer=None, n_iter=None, collect_policy=None, eval_policy=None,
                          buffer_name=None,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        num_episodes = 20 #1000
        total_rewards = []
        summed_rewards = 0

        env = self.agent.env
        env.isTest = True
        STACK_SIZE = self.agent.params['STACK_SIZE']

        for i_episode in range(num_episodes):
            # Initialize the environment and state
            env.reset()
            state = self.agent.get_screen()
            stacked_states = collections.deque(STACK_SIZE * [state], maxlen=STACK_SIZE)
            for t in count():
                stacked_states_t = torch.cat(tuple(stacked_states), dim=1)
                # Select and perform an action
                action = self.agent.select_action(stacked_states_t, i_episode)
                _, reward, done, _ = env.step(action.item())
                reward = torch.tensor([reward], device=self.agent.device)

                # Observe new state
                next_state = self.agent.get_screen()
                if not done:
                    next_stacked_states = stacked_states
                    next_stacked_states.append(next_state)

                # Move to the next state
                stacked_states = next_stacked_states

                if done:
                    reward = reward.cpu().numpy().item()
                    summed_rewards += reward
                    total_rewards.append(reward)
                    break

        mean_reward = summed_rewards/num_episodes
        print('Mean test score: {:.2f}'.format(mean_reward))
        if(writer is not None):
            writer.add_scalar('test 100 episodes mean rewards', mean_reward, i_episode)
            writer.close()
        env.isTest = False




    ####################################
    ####################################

    def collect_training_trajectories(self, itr, initial_expertdata, collect_policy, num_transitions_to_sample, save_expert_data_to_disk=False):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param num_transitions_to_sample:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        # TODO: get this from hw1 or hw2
        if itr == 0 and load_initial_expertdata:
            with open(load_initial_expertdata, 'rb') as f:
                loaded_paths = pickle.load(f)
            return loaded_paths, 0, None

        # TODO collect `batch_size` samples to be used for training
        # HINT1: use sample_trajectories from utils
        # HINT2: you want each of these collected rollouts to be of length self.params['ep_len']

        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = utils.sample_trajectories(
            self.env, collect_policy, batch_size, self.params['ep_len'])

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            ## TODO look in utils and implement sample_n_trajectories
            train_video_paths = utils.sample_n_trajectories(
                self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        return paths, envsteps_this_batch, train_video_paths

    ####################################
    ####################################

    def train_agent(self):
        # TODO: get this from Piazza
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])
            # import ipdb; ipdb.set_trace()
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

    ####################################
    ####################################

    def do_relabel_with_expert(self, expert_policy, paths):
        raise NotImplementedError
        # get this from hw1 or hw2 or ignore it b/c it's not used for this hw

    ####################################
    ####################################
    
    def perform_dqn_logging(self, all_logs):
        last_log = all_logs[-1]

        episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

        logs = OrderedDict()

        logs["Train_EnvstepsSoFar"] = self.agent.t
        print("Timestep %d" % (self.agent.t,))
        if self.mean_episode_reward > -5000:
            logs["Train_AverageReturn"] = np.mean(self.mean_episode_reward)
        print("mean reward (100 episodes) %f" % self.mean_episode_reward)
        if self.best_mean_episode_reward > -5000:
            logs["Train_BestReturn"] = np.mean(self.best_mean_episode_reward)
        print("best mean reward %f" % self.best_mean_episode_reward)

        if self.start_time is not None:
            time_since_start = (time.time() - self.start_time)
            print("running time %f" % time_since_start)
            logs["TimeSinceStart"] = time_since_start

        logs.update(last_log)
        
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.eval_env, self.agent.eval_policy, self.params['eval_batch_size'], self.params['ep_len'])
        
        eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]
        eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

        logs["Eval_AverageReturn"] = np.mean(eval_returns)
        logs["Eval_StdReturn"] = np.std(eval_returns)
        logs["Eval_MaxReturn"] = np.max(eval_returns)
        logs["Eval_MinReturn"] = np.min(eval_returns)
        logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)
        
        logs['Buffer size'] = self.agent.replay_buffer.num_in_buffer

        sys.stdout.flush()

        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, self.agent.t)
        print('Done logging...\n\n')

        self.logger.flush()

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):

        last_log = all_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        # save eval rollouts as videos in tensorboard event file
        if self.logvideo and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            #save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        #######################

        # save eval metrics
        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                try:
                    self.logger.log_scalar(value, key, itr)
                except:
                    pdb.set_trace()
            print('Done logging...\n\n')

            self.logger.flush()

    def dump_density_graphs(self, itr):
        import matplotlib.pyplot as plt
        self.fig = plt.figure()
        filepath = lambda name: self.params['logdir']+'/curr_{}.png'.format(name)

        num_states = self.agent.replay_buffer.num_in_buffer - 2
        states = self.agent.replay_buffer.obs[:num_states]
        if num_states <= 0: return
        
        H, xedges, yedges = np.histogram2d(states[:,0], states[:,1], range=[[0., 1.], [0., 1.]], density=True)
        plt.imshow(np.rot90(H), interpolation='bicubic')
        plt.colorbar()
        plt.title('State Density')
        self.fig.savefig(filepath('state_density'), bbox_inches='tight')

        plt.clf()
        ii, jj = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
        obs = np.stack([ii.flatten(), jj.flatten()], axis=1)
        density = self.agent.exploration_model.forward_np(obs)
        density = density.reshape(ii.shape)
        plt.imshow(density[::-1])
        plt.colorbar()
        plt.title('RND Value')
        self.fig.savefig(filepath('rnd_value'), bbox_inches='tight')

        plt.clf()
        exploitation_values = self.agent.exploitation_critic.qa_values(obs).mean(-1)
        exploitation_values = exploitation_values.reshape(ii.shape)
        plt.imshow(exploitation_values[::-1])
        plt.colorbar()
        plt.title('Predicted Exploitation Value')
        self.fig.savefig(filepath('exploitation_value'), bbox_inches='tight')

        plt.clf()
        exploration_values = self.agent.exploration_critic.qa_values(obs).mean(-1)
        exploration_values = exploration_values.reshape(ii.shape)
        plt.imshow(exploration_values[::-1])
        plt.colorbar()
        plt.title('Predicted Exploration Value')
        self.fig.savefig(filepath('exploration_value'), bbox_inches='tight')
