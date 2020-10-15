import gym
import numpy as np


class Policy(object):
	def __init__(self, theta, lr=0.001, gamma=0.9):
		self.theta = theta
		self.lr = lr
		self.gamma = gamma

	# Plain 1D sigmoid - only two actions possible,
	# so one explicit probability necessary.
	def sigmoid(self, x):
		return 1.0 / (1 + np.exp(-x))

	# Get probabilities of choosing either left or right for this state.
	def probs(self, state):
		p = self.sigmoid(state @ self.theta)
		return np.array([p, 1.0 - p])

	# Sample action according to the underlying distribution.
	# Return chosen action and its probability for further use.
	def act(self, state):
		probs = self.probs(state)
		action = np.random.choice([0, 1], p=probs)
		return action, probs[action]

	# Return gradients for either of the two actions (log sigmoid grads).
	def grad_log_p(self, state):
		log_grad_p0 = state * (1 - self.sigmoid(self.theta @ state))
		log_grad_p1 = -state * self.sigmoid(self.theta @ state)
		return np.array([log_grad_p0, log_grad_p1])

	# Using the rewards encountered on a trajectory, get cumulative discounted
	# rewards.
	def get_discounted_rewards(self, rewards):
		discounted_rewards = [0] * len(rewards)
		cumulative_reward = 0.0

		for i in range(len(rewards) - 1, - 1, -1):
			cumulative_reward = cumulative_reward * self.gamma + rewards[i]
			discounted_rewards[i] = self.gamma * cumulative_reward

		return discounted_rewards

	# Update our policy parameters via gradient ascent.
	def update(self, observations, actions, rewards):
		discounted_rewards = self.get_discounted_rewards(rewards)
		grad_log_p = np.array([self.grad_log_p(observation)[action] for observation, action in zip(observations, actions)])

		assert grad_log_p.shape == (len(observations), 4)

		policy_grad = grad_log_p.T @ discounted_rewards
		self.theta += self.lr * policy_grad


def run_episode(env, policy, render=False):

	total_reward = 0
	observation = env.reset()

	observations = []
	actions = []
	rewards = []

	done = False

	while not done:
		if render:
			env.render()

		observations.append(observation)
		action, _ = policy.act(observation)

		observation, reward, done, info = env.step(action)

		total_reward += reward
		actions.append(action)
		rewards.append(reward)

	env.close()
	return total_reward, np.array(rewards), np.array(observations), np.array(actions)


def train(theta, lr=0.02, gamma=0.99, MAX_EPISODES=1000):
	env = gym.make('CartPole-v0')
	policy = Policy(theta, lr, gamma)

	for i in range(MAX_EPISODES):
		total_reward, episode_rewards, observations, actions = run_episode(env, policy)

		print(i, ':', total_reward)
		policy.update(observations, actions, episode_rewards)

	run_episode(env, policy, True)

if __name__ == '__main__':
	theta = np.random.rand(4)
	train(theta)