import argparse
import csv
import os
import time
from typing import Dict

import numpy as np
import ray
import uuid
from ray.rllib import BaseEnv
from ray.rllib.agents import ppo, dqn
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import try_import_torch
from ray.tune import register_env

from marketenvDiscreteNoQuantity import MarketEnvDiscreteNoQuantity
from marketenvDiscreteNoQuantityBlind import MarketEnvDiscreteNoQuantityBlind
from marketenvDiscrete import MarketEnvDiscrete

torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", type=str, default="PPO")  # PPO, DQN, SAC
parser.add_argument("--gpuid", type=str)
parser.add_argument("--num-agents", type=int, default=3)
parser.add_argument("--late-join-ep", '--list', action='append', help="--late-join-ep 10 --late-join-ep 20", type=list, default=[])
parser.add_argument('--framework', choices=["tf", "tf2", "tfe", "torch"], default="tf2")  # torch or tf2
parser.add_argument('--manual-log', help='use this to create a csv log file', action='store_true', default=False)
parser.add_argument('--bias', help="0 = no bias, 1 = full bias towards lowest price", type=float, default=0)
parser.add_argument('--blind', action='store_true', default=False)
parser.add_argument('--no-quantity', action='store_true', default=False)
parser.add_argument('--shared-policy', action='store_true', default=False)
parser.add_argument('--local-log', action='store_true', default=False)


class TensorboardCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.episode_counter = 0
        manual_log_path = "./local_logs/" if args.local_log else "/ceph/mischlec/RL/ppo market runs results/new/"
        self.num_agents = args.num_agents + len(args.late_join_ep)
        blind = "blind_" if args.blind else ""
        shared_policy = "shared_policy_" if args.shared_policy else ""
        myuuid = str(uuid.uuid4())[:8]
        self.episode_rows = []
        self.filename = manual_log_path + \
                        shared_policy + \
                        args.algorithm + "_" + \
                        str(self.num_agents) + "a_" + \
                        blind + \
                        "bias" + str(args.bias) + "_" + \
                        time.strftime("%Y_%m_%d-%H_%M_%S") + \
                        myuuid + '.csv'

    def on_episode_start(self, worker: RolloutWorker, base_env: MarketEnvDiscrete, policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        episode.user_data["agents"] = []
        episode.user_data["agent_rewards"] = None
        self.episode_counter += 1
        self.episode_rows = []

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, **kwargs):
        # executed before step
        unwrapped_env = base_env.get_unwrapped()[0]
        if unwrapped_env.current_step > 0:
            agents = {}
            for agent in unwrapped_env.agents:
                agents[agent["id"]] = {
                    "price": agent["price"],
                    "quantity": agent["quantity"]
                }

            episode.user_data["agents"].append(agents)

        # manual logging to csv
        for agent in unwrapped_env.agents:
            left_over, revenue, profit, sold_units = 0, 0, 0, 0
            if "left_over" in agent:
                left_over = agent['left_over']
            if "revenue" in agent:
                revenue = agent['revenue']
            if "profit" in agent:
                profit = agent['profit'][0]
            if "sold_units" in agent:
                sold_units = revenue/agent["price"][0]

            self.episode_rows.append(
                {'agent_id': agent["id"],
                 'episode_id': self.episode_counter,
                 'step': unwrapped_env.current_step,
                 'price': agent["price"][0],
                 'quantity': agent["quantity"][0],
                 # 'capital': agent["capital"][0],
                 'costs': unwrapped_env.cost,
                 'revenue': revenue,
                 'profit': profit,
                 'demand_for_price': unwrapped_env.calc_demand(agent["price"])[0],
                 'sold_units': sold_units,
                 'leftover_units': left_over,
                 },
            )

    def on_episode_end(self, worker: RolloutWorker, base_env: MarketEnvDiscrete, policies: Dict[str, Policy],
                       episode: MultiAgentEpisode, **kwargs):
        # WRITE TO FILE
        if args.manual_log:
            file_exists = os.path.isfile(self.filename)

            with open(self.filename, 'a') as csvfile:

                fieldnames = ['agent_id', 'episode_id', 'step', 'price', 'quantity', 'capital', 'costs',
                              'profit', 'revenue', 'demand_for_price', 'sold_units', 'leftover_units']
                writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()  # file doesn't exist yet, write a header

                for row in self.episode_rows:
                    writer.writerow(row)

        # LOG CUSTOM METRICS
        for agentId in range(self.num_agents):
            # rewards
            episode_reward = episode.agent_rewards.get((agentId, 'default_policy'))
            episode.custom_metrics["episode_reward/" + str(agentId)] = episode_reward if episode_reward is not None else 0

            # action avg
            agent_actions = [
                step[agentId]
                for step in episode.user_data["agents"]
                if agentId in step
            ]
            if len(agent_actions) > 0:
                # noinspection PyTypeChecker
                episode.custom_metrics["avg_price/" + str(agentId)] = np.mean(
                    [action["price"] for action in agent_actions]
                )
                # noinspection PyTypeChecker
                episode.custom_metrics["avg_quantity/" + str(agentId)] = np.mean(
                    [action["quantity"] for action in agent_actions]
                )

                # demand
                market_potential = base_env.get_unwrapped()[0].calc_demand(
                    episode.custom_metrics["avg_price/" + str(agentId)]
                )
                episode.custom_metrics["unused_market_potential/" + str(agentId)] = \
                    market_potential - episode.custom_metrics["avg_quantity/" + str(agentId)]

            else:
                episode.custom_metrics["avg_price/" + str(agentId)] = 0
                episode.custom_metrics["avg_quantity/" + str(agentId)] = 0
                episode.custom_metrics["unused_market_potential/" + str(agentId)] = 0

    def _calc_optimal_price(self, env):
        cost = env.cost
        max_earnings = 0
        optimal_price = 0
        optimal_quantity = 0
        for price in range(int(cost*3)):
            for quantity in range(int(cost*3)):
                sold_units = (env.calc_demand(price) - (np.clip(env.calc_demand(price)-quantity, 0, np.inf)))
                earnings = price * sold_units - cost * quantity
                if earnings > max_earnings:
                    max_earnings = earnings
                    optimal_price = price
                    optimal_quantity = quantity
        return optimal_price, optimal_quantity


"""custom algo"""


"""run"""
if __name__ == "__main__":
    args = parser.parse_args()

    if args.gpuid:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
        ray.init(dashboard_host="0.0.0.0", dashboard_port=8905)
    else:
        ray.init(log_to_driver=False)

    np.seterr('raise')

    env_config = {
            "num_agents": args.num_agents,
            "late_join_ep": [int(''.join(episodeStr)) for episodeStr in args.late_join_ep],
            "max_steps": 365,
            "start_demand": 100,
            "price_elasticity": 1,  # isoelastic Demand # Price Elasticity= (% Change in Quantity)/(% Change in Price)
            # "start_capital": 1000,
            "unit_cost": 1,
            "max_price": 2,
            "bias": args.bias,
            "num_action_gradients": 7,   # was 83
            "num_actions": 1 if args.no_quantity else 2
            # TODO: add confounding factors
            # "demand_change_steps": [5e+6, 6e+6],
            # "cost_change_steps": [5e+6, 6e+6],
            # "confounders": tune.grid_search(["modulate_no_of_agents", "modulate_costs", "modulate_demand"])
        }
    # generate num of default policies
    policies = {str(agent_id): PolicySpec() for agent_id in range(args.num_agents)}
    ppoconfig = {
        "env_config": env_config,
        "env": "marketenv",
        "model": {
            "fcnet_hiddens": [256, 256]
            #     "custom_model": "my_model",
            #     "vf_share_layers": True,
        },
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": lambda agent_id: str(agent_id)
        } if not args.shared_policy else {},
        "callbacks": TensorboardCallbacks,
        "lr": 1e-5,
        "num_workers": 0,  # parallelism
        "clip_actions": False,
        "framework": args.framework,
        "grad_clip": 1.0,
        "num_gpus": 0 if not args.gpuid else 1,
    }
    dqnconfig = {
        "env_config": env_config,
        "model": {
            "fcnet_hiddens": [256, 256]
            #     "custom_model": "my_model",
            #     "vf_share_layers": True,
        },
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": lambda agent_id: str(agent_id)
        } if not args.shared_policy else {},
        "env": "marketenv",
        "callbacks": TensorboardCallbacks,
        "lr": 1e-5,  # try different lrs
        "num_workers": 0,  # parallelism
        "framework": args.framework,
        "num_gpus": 0 if not args.gpuid else 4,
        # "num_cpus": 1,  # deactivate if running locally
    }

    # discrete or continuous
    if args.blind:
        register_env('marketenv', lambda env_config: MarketEnvDiscreteNoQuantityBlind(env_config))
    else:
        if args.no_quantity:
            register_env('marketenv', lambda env_config: MarketEnvDiscreteNoQuantity(env_config))
        else:
            register_env('marketenv', lambda env_config: MarketEnvDiscrete(env_config))

    if args.algorithm == "DQN":
        trainer = dqn.DQNTrainer(config=dqnconfig, logger_creator=None)

    if args.algorithm == "PPO":
        trainer = ppo.PPOTrainer(config=ppoconfig, logger_creator=None)

    # restart from checkpoint
    # analysis = ray.tune.analysis.ExperimentAnalysis('/') # TODO PATH TO RAY LOGDIR HERE
    # checkpoint_dir = analysis.get_best_config(metric=self._config['metric'])
    # checkpoint_path = # < your code to extract latest checkpoint file from the best logdir >
    # trainer.restore(checkpoint_path)


    policy = trainer.get_policy()
    # print(policy.model.base_model.summary())
    episode = 0

    while episode < 15000:
        results = trainer.train()
        episode = results["episodes_total"]

    ray.shutdown()
