import torch
import torch.nn as nn

import numpy as np 
import gym

from torch.distributions.categorical import Categorical
from torch.optim import Adam


env = gym.make('SpaceInvaders-v0')

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# Small conv net for the actual policy.
def get_policy_net():
	conv1 = nn.Conv2d(4, 5, kernel_size=(3, 3), stride=(1, 1))
	conv2 = nn.Conv2d(5, 2, kernel_size=(4, 4), stride=(2, 2))
	conv3 = nn.Conv2d(2, 2, kernel_size=(5, 5), stride=(2, 2))

	flatten = Flatten()

	linear1 = nn.Linear(in_features=3700, out_features=800, bias=True)
	linear2 = nn.Linear(in_features=800, out_features=env.action_space.n, bias=True)

	return nn.Sequential(*[conv1, conv2, conv3, flatten, linear1, linear2])


# Policy network.
policy = get_policy_net()
lr = 0.001

# Get probability distributions over actions for every observation
# in the obs tensor.
def get_probs(obs):
	return Categorical(logits=policy(obs))


# Sample an action for the given observation.
def get_action(obs):
	return get_probs(obs).sample().item()


# Not a loss function, per-se. We merely use this formulation so that
# upon differentiation, we get the policy gradients we're interested in.
def compute_loss(rewards, obs, act):
	log_probs = get_probs(obs).log_prob(act)
	return -(log_probs * (rewards - 2700)).mean()


def preprocess_observation(obs):
	obs = np.mean(obs, -1)
	obs = torch.as_tensor(obs, dtype=torch.float32)
	return obs


# Discount rewards and cumulate sums.
def get_discounted_rewards(rewards, gamma=0.9):
	discounted_rewards = [0] * len(rewards)
	cumulative_reward = 0.0

	for i in range(len(rewards) - 1, -1, -1):
		cumulative_reward = cumulative_reward * gamma + rewards[i]
		discounted_rewards[i] = cumulative_reward

	return discounted_rewards


# Run an actual trajectory and compute return useful stats.
# Total reward, observations, actions and rewards.
def run_episode(gamma=0.99, batch_size=5000, render=False):
	obs = env.reset()
	obs_stack = [preprocess_observation(obs)]

	rewards = []
	actions = []
	observations = []
	done = False

	# Collect first set of stacked frames.
	for i in range(2):
		obs, _, _, _  = env.step(0)
		obs_stack.append(preprocess_observation(obs))

	while not done:
		if render:
			env.render()

		proc_obs = preprocess_observation(obs.copy())
		obs_stack.append(proc_obs)
		observations.append(torch.stack(obs_stack))
		obs_stack.pop(0)

		act = get_action(observations[-1][None, :])
		obs, reward, done, _ = env.step(act)

		rewards.append(act)
		actions.append(act)

		if len(observations) > batch_size:
			break

	total_reward = sum(rewards)
	discounted_rewards = get_discounted_rewards(rewards, gamma)

	actions = torch.as_tensor(actions, dtype=torch.int32)
	discounted_rewards = torch.as_tensor(discounted_rewards, dtype=torch.float32)
	observations = torch.stack(observations)

	return total_reward, observations, actions, discounted_rewards


def train(optimizer, gamma=0.99, MAX_EPISODES=500):
	for i in range(MAX_EPISODES):
		total_reward, observations, actions, rewards = run_episode(gamma)
		print(i, ':', total_reward)

		optimizer.zero_grad()
		batch_loss = compute_loss(rewards, observations, actions)

		batch_loss.backward()
		optimizer.step()

	run_episode(gamma=gamma, batch_size=5000, render=True)


if __name__ == '__main__':
	optimizer = Adam(policy.parameters(), lr=lr)
	print(torch.cuda.is_available())
	train(optimizer)