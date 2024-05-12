# PriceOfAlgorithmicPricing
This project was built using [Ray's RLLIB](https://docs.ray.io/en/latest/rllib/index.html)

## launch
to start a run, please execute main.py

## arguments
| argument        | type  | default | choices                     |
|-----------------|-------|---------|-----------------------------|
| --algorithm     | str   | "PPO"   | "PPO", "DQN"                |
| --gpuid         | str   |         |                             |
| --num_agents    | int   | 3       |                             |
| --late_join_ep  | int   |         |                             |
| --framework     | str   | "tf2"   | "tf", "tf2", "tfe", "torch" |
| --manual-log    | bool  | False   |                             |
| --bias          | float | 0       |                             |
| --blind         | bool  | False   |                             |
| --no-quantity   | bool  | False   |                             |
| --shared-policy | bool  | False   |                             |
| --local-log     | bool  | False   |                             |
| --custom-filename| str  |         |                             |
| --supervision   | bool  | False   |                             |

## Schlechtinger et al. (2023) & Schlechtinger et al. (2024)
### example arguments to start scenario A 
--algorithm "PPO" --num-agents 3 --bias 0 --no-quantity

### arguments to start scenario B
--algorithm "PPO" --num-agents 3 --bias 0 --no-quantity --blind

## "Breaking the Cycle - Preventing Pricing AIs from Engagig in Collusion" (2024)
### example arguments to prevent collusion sparsely
--algorithm "PPO" --num-agents 3 --bias 1 --no-quantity --supervision
