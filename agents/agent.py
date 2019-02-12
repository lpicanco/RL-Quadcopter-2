from agents.actor import Actor
from agents.critic import Critic
from agents.ou_noise import OUNoise
from agents.replay_buffer import ReplayBuffer
import numpy as np

class DDPG:
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.001

        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size)

        self.gamma = 0.99
        self.tau = 0.1
        self.learning_rate = 0.0005

        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, learning_rate=self.learning_rate)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, learning_rate=self.learning_rate)

        self.critic_local = Critic(self.state_size, self.action_size, learning_rate=self.learning_rate)
        self.critic_target = Critic(self.state_size, self.action_size, learning_rate=self.learning_rate)

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        self.noise.reset()
        self.last_state = self.task.reset()
        return self.last_state

    def act(self, state):
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())
    
    def step(self, action, reward, next_state, done):
        self.memory.add(self.last_state, action, reward, next_state, done)

        self.total_reward += reward
        self.count += 1
        
        if self.memory.size() > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

        self.last_state = next_state
        

    def learn(self, experiences):
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        actions_next = self.actor_target.model.predict_on_batch(next_states)
        q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        self.critic_local.model.train_on_batch(x=[states, actions], y=q_targets)

        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])

        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)        




