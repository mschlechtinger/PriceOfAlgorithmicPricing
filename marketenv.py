from gym.spaces import Box, Dict
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.spaces.repeated import Repeated
import torch
import torch.nn as nn

import copy
import random
import numpy as np
from collections import Counter

MAX_VAL = np.inf  # np.inf not possible with sigmoid


class MarketEnv(MultiAgentEnv):
    def __init__(self, config):
        # Define action and observation space
        print('CONFIG ---------------------------------------------------------------------')
        print(config)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.start_demand = config['start_demand']
        self.demand = self.start_demand
        self.start_capital = float(config['start_capital'])
        self.num_agents = config['num_agents']
        self.price_elasticity = float(config['price_elasticity'])
        self.price_bounds = [0.01, MAX_VAL]
        self.quantity_bounds = [1.0, MAX_VAL]

        # action space
        self.action_space = Dict({
            'price': Box(low=self.price_bounds[0], high=self.price_bounds[1], shape=(1,), dtype=np.float32),
            'quantity': Box(low=self.quantity_bounds[0], high=self.quantity_bounds[1], shape=(1,), dtype=np.float32)
        })

        # obs space
        self.price_space = Box(self.price_bounds[0], self.price_bounds[1], shape=(1,), dtype=np.float32)

        self.player_space = Dict({
            'prices': Repeated(self.price_space, max_len=self.num_agents - 1),
            'cost': Box(0, np.inf, shape=(1,), dtype=np.float32),
            'capital': Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
        })

        self.observation_space = Repeated(self.player_space, max_len=self.num_agents)

        # costs are somewhere between start_capital/self.demand and start_capital --> higher demand = lower costs
        self.cost = np.random.uniform(low=self.start_capital / self.demand, high=self.start_capital)  # no np round?
        self.start_cost = self.cost
        self.total_quantity_sold = 0
        self.total_money_spent = 0
        self.current_step = 0
        self.done = {}

        self.agents = [
            {
                'id': i,
                'capital': np.array([self.start_capital]),
                'price': np.array([self.cost]),
                'quantity': np.array([self.quantity_bounds[0]]),
            } for i in range(self.num_agents)]

        self.min_profits = [-0.1] * self.num_agents
        self.max_profits = [0.1] * self.num_agents

    def step(self, action_dict):
        self.current_step += 1

        rewards = {}

        # check if actions reach inf or result in NANs
        for action in action_dict.values():
            if any([np.isnan(a) for a in action.values()]):
                raise ValueError("An agent's action resulted as NAN: " + str(action) + "BEFORE")
            if any([np.isinf(a) for a in action.values()]):
                raise ValueError("An agent's action reached inf: " + str(action) + "BEFORE")

        backup_action_dict = copy.deepcopy(action_dict)
        self._softplus_actions(action_dict)
        self._calc_sales(action_dict)

        # check if actions reach inf or result in NANs
        for action in action_dict.values():
            if any([np.isnan(a) for a in action.values()]):
                raise ValueError("An agent's action resulted as NAN: " + str(action) + "AFTER")
            if any([np.isinf(a) for a in action.values()]):
                raise ValueError("An agent's action reached inf: " + str(action) + "AFTER")

        # check if agent is already bankrupt/ done
        if len(self.agents) < self.num_agents:
            # which agent is already bankrupt/ done before the step?
            loser_ids = list(set(range(self.num_agents)) - set([agent['id'] for agent in self.agents]))
            for loser_id in loser_ids:
                self.done[loser_id] = True

        for agent in self.agents:
            profit = agent['revenue'] - (agent['quantity'] * self.cost) # no np round?
            agent['profit'] = profit
            agent['capital'] += profit

            rewards[agent['id']] = np.clip(agent["profit"][0], 0, None)

            # check if an agent is bankrupt
            if agent['capital'] <= 0:
                self.done[agent['id']] = True
                # rewards[agent['id']] = reward_dict["bankrupt"]
            else:
                self.done[agent['id']] = False

            # winner check
            if Counter(self.done.values())[True] >= self.num_agents - 1:
                self.done['__all__'] = True
                # if agent["capital"] >= 0:
                #     rewards[agent['id']] = reward_dict["winner"]
            else:
                self.done['__all__'] = False

        obs = self._next_observation()

        # remove agents that lost from list
        for loser in self.done.items():
            if loser[1] and loser[0] != '__all__':
                self.agents[:] = [d for d in self.agents if d.get('id') != loser[0]]

        print(self.current_step)
        print("actions")
        print(action_dict)
        print("observation:")
        print(obs)
        for ob in obs.values():
            if not self.observation_space.contains(ob):
                print("no")

        return obs, rewards, self.done, {}

    def _softplus_actions(self, actions):
        """normalize actions by applying the softplus function"""
        for action in actions.values():
            action["price"] = self.softplus(torch.from_numpy(action["price"])).numpy() + self.price_bounds[0]
            action["quantity"] = self.softplus(torch.from_numpy(action["quantity"])).numpy() + self.quantity_bounds[0]
        return actions

    def _sigmoid_actions(self, actions):
        """normalize actions by applying the sigmoid function"""
        for action in actions.values():
            action["price"] = self.sigmoid(action["price"]) * self.price_bounds[1]
            action["quantity"] = self.sigmoid(action["quantity"]) * self.quantity_bounds[1]
        return actions

    def _calc_sales(self, actions):
        for agent in self.agents:
            agent['price'] = actions[agent['id']]["price"]
            agent['quantity'] = actions[agent['id']]["quantity"]
            price = agent['price'][0]
            quantity = agent['quantity'][0]
            demand = self.calc_demand(price)

            if demand > 0:
                agent['revenue'] = price * np.clip(quantity, None, demand)
                agent['left_over'] = np.clip(quantity - demand, 0, None)
            else:
                # no demand
                agent['revenue'] = 0
                agent['left_over'] = quantity

    def calc_demand(self, price):
        return np.clip(-1 * (-self.start_demand * self.price_elasticity - self.start_demand +
                             (self.start_demand * price * self.price_elasticity) /
                             self.cost), 0, None)

    def reset(self):
        # change costs randomly in a range of +/- 10%
        self.cost = np.random.uniform(low=self.start_cost * 0.95, high=self.start_cost * 1.05)  # no np round?
        self.agents = [
            {
                'id': i,
                'capital': np.array([self.start_capital]),
                'price': np.array([self.cost]),
                'quantity': np.array([self.quantity_bounds[0]]),
            } for i in range(self.num_agents)]

        self.total_quantity_sold = 0
        self.total_money_spent = 0
        self.current_step = 0
        self.done = {}
        # reset capital
        return self._next_observation()

    def _next_observation(self):
        # obs
        # '0': [2.4, 1.6],
        # '0': [3.4, -3.2],

        # obs = [
        #     'prices': [
        #           np.array([100],dtype=np.float64),
        #           np.array([200],dtype=np.float64)
        #           ],
        #     'cost': np.array([20]),
        #     'capital': np.array([2000])
        #     },
        # .... * num_agents
        # ]

        obs = {}
        for agent in self.agents:
            other_agents = [d for d in self.agents if d.get('id') != agent['id']]
            # get prices and quantity of other agents
            other_agents_actions = [
                np.array(other_agent['price'], dtype=np.float64)
                for other_agent in other_agents
            ]
            obs[agent['id']] = [{
                'prices': other_agents_actions,
                'cost': np.array([self.cost], dtype=np.float64),
                'capital': np.array(agent["capital"], dtype=np.float64)
            }]
        return obs

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step {self.current_step}')
        for i, agent in enumerate(self.agents):
            print(f'Agent {agent["id"]}:')
            # print(f'Price: {agent['prince']}')
            # print(f'Quantity: {agent['quantity']}')
            print(f'Profit: {agent["profit"]}; Units left over: {agent["left_over"]}')

        strongest_agent = sorted(self.agents, key=lambda a: a['price'], reverse=True)[0]
        print(f'highest earnings for Agent {strongest_agent["id"]}')

    def seed(self, seed=None):
        random.seed(seed)
