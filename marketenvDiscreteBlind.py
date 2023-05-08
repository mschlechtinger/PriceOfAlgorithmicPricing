import numpy as np

from marketenvDiscrete import MarketEnvDiscrete


class MarketEnvDiscreteBlind(MarketEnvDiscrete):
    def __init__(self, config):
        super(MarketEnvDiscreteBlind, self).__init__(config)
        self.player_space['prices'] = self.price_space

    def _next_observation(self):
        obs = {}
        for agent in self.agents:
            # other_agents = [d for d in self.agents if d.get('id') != agent['id']]
            obs[agent['id']] = [{
                'prices': np.array(agent["price"], dtype=np.float32),
                'cost': np.array([self.cost], dtype=np.float32),
                'quantity': np.array(agent["quantity"], dtype=np.float32)
            }]
        return obs
