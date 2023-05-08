import numpy as np
from gym.spaces import Box, Dict, Discrete
from ray.rllib.utils.spaces.repeated import Repeated
import torch
from collections import Counter


from marketenvDiscrete import MarketEnvDiscrete


class MarketEnvDiscreteNoQuantity(MarketEnvDiscrete):
    def __init__(self, config):
        super(MarketEnvDiscreteNoQuantity, self).__init__(config)

        self.player_space = Dict({
            'prices': Repeated(self.price_space, max_len=self.num_agents + len(self.late_join_ep)),
            'cost': Box(0, np.inf, shape=(1,), dtype=np.float32),
        })

        self.observation_space = Repeated(self.player_space, max_len=self.num_agents + len(self.late_join_ep))

    def _discretize_actions(self, action_dict):
        # map action to price only
        for agent, action in zip(self.agents, action_dict.values()):
            translated_price = self.translate_action(action)
            agent["price"] = self.softplus(torch.from_numpy(agent["price"] + translated_price)).numpy()

    def _next_observation(self):
        obs = {}

        for agent in self.agents:
            # other_agents = [d for d in self.agents if d.get('id') != agent['id']]

            obs[agent['id']] = [{
                'prices': [np.array(agent['price'], dtype=np.float32) for agent in self.agents],
                'cost': np.array([self.cost], dtype=np.float32),
            }]
        return obs

    def step(self, action_dict):
        self.current_step += 1
        rewards = {}

        # translate action to env
        self._discretize_actions(action_dict)
        orders = self._calc_orders()

        # write to agent-list
        for agent in self.agents:
            price = agent['price'][0]
            demand = orders[agent['id']]
            agent["quantity"] = np.array([demand], dtype=np.float32)

            if demand > 0:
                agent['revenue'] = price * demand
                agent['left_over'] = 0
            else:
                # no demand
                agent['revenue'] = 0
                agent['left_over'] = 0

            agent['profit'] = agent['revenue'] - (agent['quantity'] * self.cost)
            # agent['capital'] += profit

            rewards[agent['id']] = agent["profit"][0]  # REWARDS ARE CLIPPED TODO: WHAT HAPPENS IF NOT CLIPPED?

            if self.current_step >= self.max_steps:
                self.done[agent['id']] = True
            else:
                self.done[agent['id']] = False

            # winner check
            if Counter(self.done.values())[True] >= self.num_agents - 1:
                self.done['__all__'] = True
            else:
                self.done['__all__'] = False

        obs = self._next_observation()

        # remove agents that lost from list
        for loser in self.done.items():
            if loser[1] and loser[0] != '__all__':
                self.agents[:] = [d for d in self.agents if d.get('id') != loser[0]]

        return obs, rewards, self.done, {}
