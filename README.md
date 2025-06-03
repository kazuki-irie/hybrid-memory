# Hybrid Quadratic-Linear Transformers (HQLTs)

This is the official code repository for the paper:

[Blending Complementary Memory Systems in Hybrid Quadratic-Linear Transformers](https://arxiv.org/abs/2506.00744)

## Contents

* `fla` (contains common model implementations used in all the experiments)
* `language_modeling` (Table 1 and 3)
* `synthetic_algorithmic` (Table 2)
* `reinforcement_learning` (Figure 2)

## General instructions

Please refer to the README file in the corresponding directory for further instructions.

* Each of the experimental directories, `language_modeling`, `synthetic_algorithmic`, and `reinforcement_learning`, uses `fla`.
Please create a symbolic link (e.g., `ln -s ../fla/fla .` ; i.e., `fla` directory under `fla`) under each of these directories.

* We used the same conda environment for all the settings. The corresponding environment file is `environment.yml`. We used `Python 3.10.16`.

* [weights & biases](https://wandb.ai/site/) is used in all our experiments.

## Acknowledgement

This repository contains forks of code from the following publicly available resources.
LICENSE files are included in the corresponding directories.
We extend our heartfelt thanks to all the corresponding authors for making these very useful toolkits openly available.

* `language_modeling` is a fork of [fla-org/flame](https://github.com/fla-org/flame).
* `fla` is a fork of [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention).
* `synthetic_algorithmic` is a fork of [automl/unlocking_state_tracking](https://github.com/automl/unlocking_state_tracking) which itself is based on [NX-AI/xlstm](https://github.com/NX-AI/xlstm).
* `reinforcement_learning` is a fork of [twni2016/Memory-RL](https://github.com/twni2016/Memory-RL).
