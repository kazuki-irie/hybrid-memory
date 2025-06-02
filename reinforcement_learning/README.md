## Reinforcement learning experiments

This folder contains the codebase used to produce the RL results.

This repository was originally forked from [twni2016/Memory-RL](https://github.com/twni2016/Memory-RL).

## Requirements

We refer our environment file in the root folder.

## Instructions

Please check out the `dropout` branch for these RL experiments.

Here is the command to run an experiment for a single model for a single seed:

```
export CUDA_VISIBLE_DEVICES=0

SEED=42

python main.py \
  --seed ${SEED} \
  --config_env configs/envs/visual_match.py \
  --config_env.env_name 750 \
  --config_env.final_reward 100 \
  --config_env.respawn_every 20 \
  --wandb_name 'RL--hybrid-memory' \
  --config_rl configs/rl/sacd_default.py \
  --shared_encoder \
  --freeze_critic \
  --train_episodes 2565 \
  --config_seq configs/seq_models/gpt_cnn_2lay_2head.py \
  --config_seq.sampled_seq_len -1
```

Modify the `--config_seq` argument above to change the sequence model as follows:
* Transformer: `configs/seq_models/gpt_cnn_2lay_2head.py`
* DeltaNet: `configs/seq_models/deltanet_cnn_2lay_2head.py`
* HQLT (synchronous): `configs/seq_models/hqlt_cnn_2lay_2head.py`

Training progress and evaluation can be monitored on "weights and biases" (check `return` and `suceess` for evaluation)
