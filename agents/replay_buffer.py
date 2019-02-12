from collections import deque, namedtuple
import random

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        xp = self.Experience(state, action, reward, next_state, done)
        self.buffer.append(xp)

    def size(self):
        return len(self.buffer)

    def sample(self, count):
        return random.sample(self.buffer, k=count)