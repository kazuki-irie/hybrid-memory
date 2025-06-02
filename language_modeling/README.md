## Language modeling experiments

This folder contains the codebase used to produce the language modeling results.

This repository was originally forked from [fla-org/flame](https://github.com/fla-org/flame). We provide the code used in our experiments as-is. For a more up-to-date training framework, we recommend checking the original repository.


## Requirements

We refer our environment file in the root folder. See also the requirements in [fla-org/flame](https://github.com/fla-org/flame)

## Training

An example training script is as follows. 

```
export NGPU=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

export HF_HOME=

MODEL_CONFIG=
DUMP=exp/mymodel

TOKENIZER=fla-hub/transformer-1.3B-100B

DATASET=HuggingFaceFW/fineweb-edu
DATASET_NAME=sample-100BT

STEPS=28672

bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder ${DUMP} \
  --model.config ${MODEL_CONFIG} \
  --model.tokenizer_path ${TOKENIZER} \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 1e-3 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 16 \
  --training.seq_len 2048 \
  --training.gradient_accumulation_steps 4 \
  --training.steps ${STEPS} \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset ${DATASET} \
  --training.dataset_name ${DATASET_NAME} \
  --training.dataset_split train \
  --training.num_workers 12 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
  --checkpoint.interval 2048 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 2 \
  --metrics.log_freq 1
```

The following variables need to be set:
* `HF_HOME`: Huggingface cache directory.
* `DUMP`: output directory (replace `exp/mymodel` above)
* `MODEL_CONFIG`: a config file (see below).

`configs` directory contains all the configs included in the paper. Their file name should be self-explanatory to find which model it corresponds to.

For 1.3B models, we used `--training.seq_len` of `2240` instead of `2048`

NB: We used four H100-80GB GPUs.
To train on GPUs with less memory, we recommend to reduce `--training.batch_size` and increase `--training.gradient_accumulation_steps` accordingly.

## Evaluation

The following evaluation script was used to test on all the language modeling tasks, except FDA:

```
export NGPU=1
export CUDA_VISIBLE_DEVICES=0

export HF_HOME=

CHECKPOINT=exp/mymodel

python -m evals.harness --model hf \
  --model_args pretrained=$CHECKPOINT,dtype=bfloat16 \
  --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,squad_completion,swde \
  --batch_size 1 \
  --num_fewshot 0 \
  --device cuda \
  --show_config
```

Note that for `squad_completion` and `swde`, we applied the white space stripping following: https://github.com/EleutherAI/lm-evaluation-harness/issues/2690

Regarding FDA, we used [HazyResearch/prefix-linear-attention](https://github.com/HazyResearch/prefix-linear-attention](https://github.com/HazyResearch/prefix-linear-attention/tree/main/lm-eval-harness/prompt_scripts)) following the recommendation from [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet?tab=readme-ov-file#5%EF%B8%8F%E2%83%A3-any-guidance-for-evaluating-the-models). As we only need `fda`, the corresponding script can be simply:

```
output_dir=
limit=-1

export PYTHONPATH=.

export CUDA_VISIBLE_DEVICES=0

export HF_HOME=

CHECKPOINT=exp/mymodel

python launch_hf.py \
  --batch-size 1 \
  -m ${CHECKPOINT} \
  -t based_fda \
  --output_dir ${output_dir} \
  --context_length 1000 \
  --answer_length 50 \
  --cutting_context \
  --limit ${limit} \
  -p
```
