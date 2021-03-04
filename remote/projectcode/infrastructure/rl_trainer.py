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
import matplotlib.pyplot as plt
from datetime import timedelta

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

        img_size = self.params['obs_size']
        _maxSteps = self.params['MaxSteps']
        env = KukaDiverseObjectEnv(renders=False, isDiscrete=False, removeHeightHack=False, maxSteps=_maxSteps, width=img_size,
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
        num_iterations = self.agent.params['num_iterations'] #1000
        writer = SummaryWriter(self.params['logdir'])
        summed_loss = 0
        start_time = timeit.default_timer()

        log_loss_frequ = self.params['log_loss_frequ']
        n_test_episodes = self.params['n_episodes_per_eval']

        for itr in range(num_iterations):
            # Initialize the environment and state
            if (itr % 10 == 0):
                print("Episode: ", itr)
            if(itr==0 and self.params['debug_mode']):
                print("check_point")
                state = self.agent.get_screen()
                print("get_screen/state.shape: ", state.shape)
                self.agent.display_screen(state)

            if(itr % self.params['eval_every_n_iterations'] == 0):
                #paths, acc_rewards = self.collect_training_trajectories()
                print("Memory size before collecting: ", len(self.agent.memory))
                print("Evaluating on training set ... ")
                mean_reward_train = self.eval_agent(writer, itr, isTest=False, n_test_episodes=self.params['n_episodes_per_eval'])
                print("Evaluating on test set ... ")
                mean_reward_test = self.eval_agent(writer, itr, isTest=True, n_test_episodes=self.params['n_episodes_per_eval'])
                print("Memory size after collecting: ", len(self.agent.memory))
                print("... and go on with training")

                print('Mean {dataset} score: {mean_reward}'.format(dataset='train', mean_reward=mean_reward_train))
                print('Mean {dataset} score: {mean_reward}'.format(dataset='test', mean_reward=mean_reward_test))

                if (writer is not None):
                    writingOn = 'avg_train_reward_over_' + str(n_test_episodes) + 'episodes'
                    writer.add_scalar(writingOn, mean_reward_train, itr)

                    writingOn = 'avg_test_reward_over_' + str(n_test_episodes) + 'episodes'
                    writer.add_scalar(writingOn, mean_reward_test, itr)

            loss = self.agent.optimize_model()

            if(loss is not None):
                summed_loss += loss

                # LOG
                if(writer is not None and itr % log_loss_frequ == 0):
                    writingOn = 'avg_loss_over_past_' + str(log_loss_frequ) + '_iterations'
                    writer.add_scalar(writingOn, summed_loss / log_loss_frequ, itr)
                    summed_loss = 0

            #if itr % self.agent.params['TARGET_UPDATE'] == 0:
            #    self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())


        final_reward = self.eval_agent(writer, itr, isTest=True, n_test_episodes=200)
        print('Average Score: {:.2f}'.format(final_reward))
        elapsed = timeit.default_timer() - start_time
        print("Elapsed time: {}".format(timedelta(seconds=elapsed)))
        writer.close()
        print("DONE")


    def eval_agent(self, writer=None, train_episode=0, isTest=False, n_test_episodes=50):
        total_rewards = []
        summed_rewards = 0

        env = self.agent.env
        env.isTest = isTest
        STACK_SIZE = self.agent.params['STACK_SIZE']

        for i_episode in range(n_test_episodes):
            # Initialize the environment and state
            env.reset()
            state = self.agent.get_screen()
            stacked_states = collections.deque(STACK_SIZE * [state], maxlen=STACK_SIZE)
            for t in count():
                stacked_states_t = torch.cat(tuple(stacked_states), dim=1)
                # Select and perform an action
                action = self.agent.select_action(stacked_states_t, i_episode, t)
                action_ext = np.append(action, 0.0)
                _, reward, done, _ = env._step_continuous(action_ext)
                reward = torch.tensor([reward], dtype=torch.float32, device=self.agent.device)

                # Observe new state
                next_state = self.agent.get_screen()
                if not done:
                    next_stacked_states = stacked_states
                    next_stacked_states.append(next_state)
                    next_stacked_states_t = torch.cat(tuple(next_stacked_states), dim=1)
                else:
                    next_stacked_states_t = None

                # Store the transition in memory
                if(self.params['on_policy'] == True and isTest == False):
                    self.agent.memory.push(stacked_states_t, torch.tensor([action], device=self.agent.device), next_stacked_states_t, reward, torch.tensor([t], dtype=torch.float32, device=self.agent.device))
                    #print("imgRGB.shape when pushed during online: ", stacked_states_t.shape)
                    #print("action.shape when pushed during online: ", torch.tensor([action], device=self.agent.device).shape)
                # Move to the next state
                stacked_states = next_stacked_states

                if done:
                    reward = reward.cpu().numpy().item()
                    summed_rewards += reward
                    total_rewards.append(reward)
                    break

        mean_reward = summed_rewards/n_test_episodes

        env.isTest = False

        return mean_reward


    ####################################
    ####################################

    def collect_training_trajectories(self):
        num_episodes = self.params['n_episodes_collected_per_iteration']

        env = self.agent.env
        STACK_SIZE = self.agent.params['STACK_SIZE']

        for i_episode in range(num_episodes):

            env.reset()
            state = self.agent.get_screen()
            stacked_states = collections.deque(STACK_SIZE * [state], maxlen=STACK_SIZE)

            for t in count():
                stacked_states_t = torch.cat(tuple(stacked_states), dim=1)
                # Select and perform an action
                action = self.agent.select_action(stacked_states_t, i_episode)
                #print("action right before: ", action)
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
                #for upts in range(50):

                if done:
                    reward = reward.cpu().numpy().item()
                    acc_rewards += reward
                    #total_rewards.append(reward)
                    #mean_reward = np.mean(total_rewards[-100:]) * 100
                    writer.add_scalar("epsilon", self.agent.eps_threshold, i_episode)

                    break

        return paths, acc_rewards

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
