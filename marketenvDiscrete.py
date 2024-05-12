from gym.spaces import Box, Dict, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.spaces.repeated import Repeated
from supervisionHelper import SupervisionHelper
import torch
import torch.nn as nn

import random
import numpy as np
import itertools
from collections import Counter

MAX_VAL = np.inf  # np.inf not possible with sigmoid


class MarketEnvDiscrete(MultiAgentEnv):
    def __init__(self, config):
        # super().__init__(self)
        
        # Define action and observation space
        print('CONFIG ---------------------------------------------------------------------')
        print(config)
        self.softplus = nn.Softplus()
        self.bias = float(config['bias'])
        self.max_steps = int(config['max_steps'])
        self.price_elasticity = float(config['price_elasticity'])
        self.start_demand = config['start_demand']
        self.demand = self.start_demand
        self.max_demand = self.start_demand + (self.price_elasticity * self.start_demand)
        # self.start_capital = float(config['start_capital'])
        config.setdefault('late_join_ep', [])
        self.late_join_ep = config['late_join_ep']
        self.num_agents = config['num_agents']
        self.price_bounds = [-MAX_VAL, MAX_VAL]  # min used to be 0.01
        self.quantity_bounds = [-MAX_VAL, MAX_VAL]  # min used to be 1.0
        self.action_gradient_max = config["max_price"]
        self.action_gradients = config['num_action_gradients']
        self.action_stack = np.hstack(
            (np.geomspace(-self.action_gradient_max, -0.01, int((self.action_gradients - 1) / 2)),
             0,
             np.geomspace(0.01, self.action_gradient_max, int((self.action_gradients - 1) / 2)))
        )
        self.num_actions = int(config["num_actions"])  # default = quantity & price

        # confounders
        try:
            self.demand_change_steps = config['demand_change_steps']
            self.cost_change_steps = config['cost_change_steps']
        except KeyError:
            print("no confounders applied")

        # action space
        self.action_space = Discrete(self.action_gradients ** self.num_actions)  # price & quantity

        # obs space
        self.price_space = Box(self.price_bounds[0], self.price_bounds[1], shape=(1,), dtype=np.float32)

        self.player_space = Dict({
            'prices': Repeated(self.price_space, max_len=self.num_agents + len(self.late_join_ep)),
            'quantity': Box(self.quantity_bounds[0], self.quantity_bounds[1], shape=(1,), dtype=np.float32),
            'cost': Box(0, np.inf, shape=(1,), dtype=np.float32),
            # 'capital': Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
        })
        setattr(self.player_space, "_shape", (3,))  # workaround for SAC Trainer

        self.observation_space = Repeated(self.player_space, max_len=self.num_agents + len(self.late_join_ep))

        setattr(self.observation_space, "_shape", [self.observation_space.max_len] + list(
            self.observation_space.child_space.shape))  # workaround for SAC Trainer

        # create action mapping to translate discrete actions to two values (price & quantity)
        self.action_mapping = []
        for element in itertools.product(*[range(self.action_gradients) for _ in range(self.num_actions)]):
            self.action_mapping.append(element)

        # set other market relevant values
        self.cost = config['unit_cost']
        self.max_price = config['max_price']
        self.start_cost = self.cost
        self.total_quantity_sold = 0
        self.total_money_spent = 0
        self.current_step = 0
        self.current_episode = 0
        self.done = {}

        self.agents = [
            {
                'id': i,
                # 'capital': np.array([self.start_capital]),
                'price': np.array([self.cost]),
                'quantity': np.array([1.0]),
            } for i in range(self.num_agents)]

        # init supervisor
        self.supervisor = bool(config['supervision'])
        if self.supervisor:
            self.supervisor = SupervisionHelper(self.num_agents, self.max_steps, self.action_gradient_max)
            self.classification_accuracies = {}
            self.regression_error_values = {}
            self.supervision_reward_factors = []

    def step(self, action_dict):
        self.current_step += 1
        rewards = {}

        # TODO: sudden demand change
        # try:
        #     for demand_change_step in self.demand_change_steps:
        #         if self.current_step == demand_change_step:
        #             self.demand = np.random.uniform(low=self.demand * 0.5, high=self.demand * 1.5)
        #             self.max_demand = self.demand + (self.price_elasticity * self.demand) - 1
        #
        # # sudden cost change
        #     for cost_change_step in self.cost_change_steps:
        #         if self.current_step == cost_change_step:
        #             self.cost = np.random.uniform(low=self.demand * 0.5, high=self.demand * 1.5)
        # except AttributeError:
        #     pass

        # translate action to env
        self._discretize_actions(action_dict)
        orders = self._calc_orders()

        # write to agent-list
        for agent in self.agents:
            price = agent['price'][0]
            quantity = agent['quantity'][0]
            demand = orders[agent['id']]

            if demand > 0:
                agent['revenue'] = price * np.clip(quantity, None, demand)
                agent['left_over'] = np.clip(quantity - demand, 0, None)
            else:
                # no demand
                agent['revenue'] = 0
                agent['left_over'] = quantity

            agent['profit'] = agent['revenue'] - (agent['quantity'] * self.cost)
            # agent['capital'] += profit

            rewards[agent['id']] = np.clip(agent["profit"][0], 0, None)  # REWARDS ARE CLIPPED
            # check if an agent is bankrupt
            # if agent['capital'] <= 0 or self.current_step >= self.max_steps:
            if self.current_step >= self.max_steps:
                self.done[agent['id']] = True
                # rewards[agent['id']] = reward_dict["bankrupt"]
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

    def _discretize_actions(self, action_dict):
        # map action to two values (price, quantity)
        for agent, action in zip(self.agents, action_dict.values()):
            translated_price, translated_quantity = self.translate_action(action)
            agent["price"] = self.softplus(torch.from_numpy(agent["price"] + translated_price)).numpy()
            agent["quantity"] = self.softplus(torch.from_numpy(agent["quantity"] + translated_quantity)).numpy()

    def translate_action(self, action):
        # map action to two values (price, quantity)
        mapping = self.action_mapping[action]
        return [self.action_stack[i] for i in mapping]

    def _calc_orders(self):
        """
        Market has a set demand (double the start demand if price elasticity is 1).
        An Agent can satisfy the demand by choosing the optimal price.
        Buyer behavior is influenced by a probability distribution.
        e.g., max price of 2; 3 agents with prices of 1.50, 1, 0.50.
        probability distribution: (2-1.5)/3 = 1/6, (2-1)/3 = 1/3, (2-0.50)/3 = 1/2
        """
        orders = dict.fromkeys(range(self.num_agents), 0)

        # calc sum of the inverses
        sum_inverse_prices = sum(
            [np.clip(self.max_price - a['price'], 0, self.max_price) for a in self.agents]
        )[0]

        if sum_inverse_prices > 0:  # if at least one price is below 2
            # calc buying probs
            buying_probabilities = [
                ((self.max_price - np.clip(a['price'], 0, self.max_price)) / sum_inverse_prices)[0]
                for a in self.agents]

            # if bias is applied
            if self.bias > 0:
                prices = [a["price"] for a in self.agents]
                agents_with_lowest_price = [self.agents[i] for i in
                                            np.where(prices == np.array([a["price"] for a in self.agents]).min())[0]]
                agents_with_higher_prices = [a for a in self.agents
                                             if a["price"] not in [b["price"] for b in agents_with_lowest_price]]

                if len(agents_with_higher_prices) != 0:

                    # calculate the sum of what will be added to the highest buying behavior(s)
                    biased_amount = 1 - sum([buying_probabilities[agent["id"]] for agent in agents_with_lowest_price])

                    # add bias to probability of lowest price(s)
                    for a in agents_with_lowest_price:
                        buying_probabilities[a["id"]] += (self.bias * biased_amount) / len(agents_with_lowest_price)

                    spare_prob = 1 - sum([buying_probabilities[a["id"]] for a in agents_with_lowest_price])
                    sum_higher_price_probs = sum([buying_probabilities[a["id"]] for a in agents_with_higher_prices])

                    # subtract bias from probability of higher prices
                    if sum_higher_price_probs > 0:
                        for a in agents_with_higher_prices:
                            buying_probabilities[a["id"]] = \
                                (buying_probabilities[a["id"]] / sum_higher_price_probs) * spare_prob

            # V1: calc orders with straight probabilities
            # for a in self.agents:
            #     orders[a["id"]] = self.calc_demand(a["price"]) * buying_probabilities[a["id"]]

            # V2: calc orders with randomizer, initialized by probabilities
            max_demand_for_prices = self.calc_demand(min([a["price"] for a in self.agents]))
            for agent_id in np.random.choice(range(len(self.agents)), size=int(max_demand_for_prices),
                                             p=buying_probabilities):
                orders[agent_id] += 1

        return orders

    def calc_demand(self, price):
        return np.clip(-1 * (-self.start_demand * self.price_elasticity - self.start_demand +
                             (self.start_demand * price * self.price_elasticity)), 0, None)

    def reset(self):
        # change costs randomly in a range of +/- 10%
        # self.cost = np.random.uniform(low=self.start_cost * 0.95, high=self.start_cost * 1.05)  # no np round?
        print(self.current_episode)
        if self.current_episode in self.late_join_ep:
            self.num_agents += 1

        self.cost = self.start_cost  # TODO: apply varying costs
        self.agents = [
            {
                'id': i,
                # 'capital': np.array([self.start_capital]),
                'price': np.array([np.random.uniform(low=self.cost * .5, high=self.cost * 1.5)]),
                'quantity': np.array([np.random.uniform(low=1, high=self.demand * 2)]),
            } for i in range(self.num_agents)]

        self.total_quantity_sold = 0
        self.total_money_spent = 0
        self.current_step = 0
        self.current_episode += 1
        self.done = {}
        return self._next_observation()

    def _next_observation(self):
        obs = {}

        # assure that prices are filled even if a player is bankrupt
        # prices = [np.array([0], dtype=np.float32) for _ in range(self.num_agents)]
        # if len(self.agents) < self.num_agents:
        #     for agent in self.agents:
        #         prices[agent['id']] = np.array(agent['price'], dtype=np.float32)
        #
        # else:
        #     prices = [
        #         np.array(agent['price'], dtype=np.float32)
        #         for agent in self.agents
        #     ]

        for agent in self.agents:
            # other_agents = [d for d in self.agents if d.get('id') != agent['id']]

            obs[agent['id']] = [{
                'prices': [np.array(agent['price'], dtype=np.float32) for agent in self.agents],
                'cost': np.array([self.cost], dtype=np.float32),
                # 'capital': np.array(agent["capital"], dtype=np.float32),
                'quantity': np.array(agent["quantity"], dtype=np.float32)
            }]
        return obs

    def seed(self, seed=None):
        random.seed(seed)
