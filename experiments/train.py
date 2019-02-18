import argparse
import numpy as np
import tensorflow as tf
import time
import dill

import maddpg_pytorch.common.torch_utils as U
from maddpg_pytorch.trainer.maddpg import MADDPGAgentTrainer

import torch
from torch import nn




#################################################################################################

parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
# Environment
parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
# Core training parameters
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
# Checkpointing
parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
# Evaluation
parser.add_argument("--restore", action="store_true", default=False)
parser.add_argument("--display", action="store_true", default=False)
parser.add_argument("--benchmark", action="store_true", default=False)
parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
opt = parser.parse_args()


observation_size = 10  # placeholder
num_actions      = 4   # placeholder


MLP = nn.Sequential(
    nn.Linear(observation_size, opt.num_units),
    nn.ReLU(),
    nn.Linear(opt.num_units, opt.num_units),
    nn.ReLU(),
    nn.Linear(opt.num_units, num_actions)
)

def mlp_model(input: torch.tensor, num_outputs: int, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    return MLP(input)


def make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, num_adversaries, obs_shape_n):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, opt,
            local_q_func=(opt.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, opt,
            local_q_func=(opt.good_policy=='ddpg')))
    return trainers





def train():
    # Create environment
    env = make_env(opt.scenario, opt, opt.benchmark)

    # Create agent trainers
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    num_adversaries = min(env.n, opt.num_adversaries)
    trainers = get_trainers(env, num_adversaries, obs_shape_n)
    print('Using good policy {} and adv policy {}'.format(opt.good_policy, opt.adv_policy))

    # Load previous results, if necessary
    if opt.load_dir == "":
        opt.load_dir = opt.save_dir
    if opt.display or opt.restore or opt.benchmark:
        print('Loading previous state...')
        # U.load_state(opt.load_dir)
        raise NotImplementedError

    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    agent_info = [[[]]]  # placeholder for benchmarking info
    saver = tf.train.Saver()
    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()

    print('Starting iterations...')
    # all_obs = []  # to collect observations for GAN training
    while True:
        # get action
        action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        # all_obs.append(new_obs_n)  # kcorder

        episode_step += 1
        done = all(done_n)
        terminal = (episode_step >= opt.max_episode_len)
        # collect experience
        for i, agent in enumerate(trainers):
            agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        if done or terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])

        # increment global step counter
        train_step += 1

        # for benchmarking learned policies
        if opt.benchmark:
            for i, info in enumerate(info_n):
                agent_info[-1][i].append(info_n['n'])
            if train_step > opt.benchmark_iters and (done or terminal):
                file_name = opt.benchmark_dir + opt.exp_name + '.pkl'
                print('Finished benchmarking, now saving...')
                with open(file_name, 'wb') as fp:
                    dill.dump(agent_info[:-1], fp)
                break
            continue

        # for displaying learned policies
        if opt.display:
            time.sleep(0.1)
            env.render()
            continue

        # update all trainers, if not in display or benchmark mode
        loss = None
        for agent in trainers:
            agent.preupdate()
        for agent in trainers:
            loss = agent.update(trainers, train_step)

        # save model, display training output
        if terminal and (len(episode_rewards) % opt.save_rate == 0):
            # todo: U.save_state(opt.save_dir, saver=saver)
            raise NotImplementedError

            # print statement depends on whether or not there are adversaries
            if num_adversaries == 0:
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                    round(time.time() - t_start, 3)))
            else:
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                    [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

        # saves final episode reward for plotting training curve later
        if len(episode_rewards) > arglist.num_episodes:
            rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
            with open(rew_file_name, 'wb') as fp:
                pickle.dump(final_ep_rewards, fp)
            agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
            with open(agrew_file_name, 'wb') as fp:
                pickle.dump(final_ep_ag_rewards, fp)
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            break


    with open('all_observations.pkl', 'wb') as fd:
        dill.dump(all_obs, fd)

if __name__ == '__main__':
    train()
