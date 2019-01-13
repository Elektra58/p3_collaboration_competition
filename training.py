from collections import deque

import numpy as np
import torch

from maddpg_agent import Agent


def maddpg(config):
    """Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

    Params
    ======
        n_episodes (int)      : maximum number of training episodes
        max_t (int)           : maximum number of timesteps per episode
        train_mode (bool)     : if 'True' set environment to training mode

    """
    scores_window = deque(maxlen=config.window)
    scores = []
    ma_scores = []
    std_scores = []
    solved = False

    # instantiate two agents in a list
    agents = [Agent(config) for _ in range(config.n_agents)]

    for i_episode in range(1, config.n_episodes + 1):
        env_info = config.env.reset(train_mode=True)[config.brain_name]

        states = np.reshape(env_info.vector_observations, (1, 48))

        for agent in agents:
            agent.reset()

        score = np.zeros(config.n_agents)

        for t in range(config.max_t):
            actions = np.concatenate([agent.act(states, add_noise=True) for agent in agents], axis=0).reshape((1, 4))

            env_info = config.env.step(actions)[config.brain_name]
            next_states = np.reshape(env_info.vector_observations, (1, 48))
            rewards = env_info.rewards
            dones = env_info.local_done

            for agent_num, reward in enumerate(rewards):
                agents[agent_num].step(t, states, actions, reward, next_states, dones, agent_num)

            states = next_states

            score += np.max(rewards)

            if np.any(dones):
                break

        max_score = np.max(score)
        scores_window.append(max_score)
        scores.append(max_score)

        ma_score = np.mean(scores_window)
        ma_scores.append(ma_score)

        std_score = np.std(scores_window)
        std_scores.append(std_score)

        str_ = f'\rEpisode {i_episode:>4}\tCurrent Score {max_score:5.2f}' + \
               f'\tAvg Score {ma_score:5.2f}\tStd Score {std_score:5.2f}'

        if i_episode % config.n_print == 0:
            print(str_)
        else:
            print(str_, end='')

        if ma_score >= config.target:
            if not solved:
                str_ = f'\nEnvironment solved in {i_episode - config.window:d} episodes' + \
                       f'\tAverage Score : {ma_score:.2f}'
                print(str_)

                for agent_num, agent in enumerate(agents):
                    torch.save(agent.actor_local.state_dict(),
                               f'results/model {config.model_id}/checkpoint_actor_{agent_num}.pth')
                    torch.save(agent.critic_local.state_dict(),
                               f'results/model {config.model_id}/checkpoint_critic_{agent_num}.pth')

                config.target_episode = i_episode - config.window
                config.target_score = ma_score
                solved = True

            if not config.all_episodes:
                break

    return scores, ma_scores, std_scores
