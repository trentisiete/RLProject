from .base import BaseAgent

class DQNAgent(BaseAgent):
    def __init__(self, obs_space, act_space, config):
        super().__init__(obs_space, act_space, config)
        # build networks, replay buffer, hyperparams from config['agent']

    def select_action(self, state):
        # ε-greedy policy…
        pass

    def observe(self, s, a, r, s2, done):
        # push to replay…
        pass

    def update(self):
        # sample minibatch, gradient step…
        return {"loss": loss_value, "epsilon": self.epsilon}
