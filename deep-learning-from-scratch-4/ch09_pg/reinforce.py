import sys
sys.path.append('..')
import dezero.functions as F
import gymnasium as gym
from ch09_pg.simple_pg import Agent as SimplePgAg


class Agent(SimplePgAg):
    def update(self):
        self.pi.cleargrads()

        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G
            loss += -F.log(prob) * G

        for reward, prob in self.memory:
            loss += -F.log(prob) * G

        loss.backward()
        self.optimizer.update()
        self.memory = []


if __name__ == '__main__':
    episodes = 3000
    env = gym.make('CartPole-v0')
    agent = Agent()
    reward_history = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, prob = agent.get_action(state)
            next_state, reward, done, truncated, info = env.step(action)

            agent.add(reward, prob)
            state = next_state
            total_reward += reward

        agent.update()
        reward_history.append(total_reward)
        if episode % 10 == 0:
            print("episode: {}, total reward: {}".format(episode, total_reward))

    from common.utils import plot_total_reward
    plot_total_reward(reward_history)