# PriceOfAlgorithmicPricing
This project was build with [Ray's RLLIB](https://docs.ray.io/en/latest/rllib/index.html)

## launch
to start a run, please execute main.py

## arguments
| argument        | type  | default | choices                     |
|-----------------|-------|---------|-----------------------------|
| --algorithm     | str   | "PPO"   |                             |
| --gpuid         | str   |         |                             |
| --num_agents    | int   | 3       |                             |
| --late_join_ep  | int   |         |                             |
| --framework     | str   | "tf2"   | "tf", "tf2", "tfe", "torch" |
| --manual-log    | bool  | False   |                             |
| --bias          | float | 0       |                             |
| --blind         | bool  | False   |                             |
| --no-quantity   | bool  | False   |                             |
| --shared-policy | bool  | False   |                             |

### example arguments to start scenario A
--algorithm "PPO" --num-agents 3 --bias 0 --no-quantity

### arguments to start scenario B
--algorithm "PPO" --num-agents 3 --bias 0 --no-quantity --blind
