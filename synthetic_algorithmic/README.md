# Synthetic algorithmic tasks

This folder contains the codebase used to produce the results on the parity and modular arithmetic tasks.

This repository was originally forked from [automl/unlocking_state_tracking](https://github.com/automl/unlocking_state_tracking) which itself is based on [NX-AI/xlstm](https://github.com/NX-AI/xlstm).

We provide the code used in our experiments as-is. For more up-to-date implementations of linear transformer models, we recommend to check [fla-org/flame](https://github.com/fla-org/flame).

## Requirements

We refer our environment file in the root folder.

## Script

An example script is as follows. 

```
MODEL_CONFIG=experiments/parity_sync_hqlt_net.yaml

LR=0.005
CLIP=1.0
BSZ=1024
LAY=2
INIT=0.006
STEP=20000
SEED=1
CHK=8

python experiments/main.py \
  --config=${MODEL_CONFIG} \
  --grad_clip_norm ${CLIP} \
  --lr ${LR} \
  --n_layers ${LAY} \
  --batch_size ${BSZ} \
  --initializer_range ${INIT} \
  --val_every_step 100 \
  --num_steps ${STEP} \
  --wand_name "synthetic-tasks" \
  --chunk_size ${CHK} \
  --seed ${SEED}
```

* Config files for other models and tasks can be found under `experiments/*yaml`.
* NB: certain hyper-parameters are set by default through flags in `main.py`. This means that these parameters in the config files are overwritten.
