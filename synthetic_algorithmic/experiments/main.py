# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbinian Poeppel, Maximilian Beck
import os
from argparse import ArgumentParser
from datetime import datetime
from typing import Type

import torch
import torch.optim as optim
from dacite import from_dict
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from experiments.data.formal_language.formal_language_dataset import (
    FormLangDatasetGenerator,
)
from experiments.data.utils import DataGen
from experiments.lr_scheduler import LinearWarmupCosineAnnealing

from hqlt_model import HybridQLTForCausalLMMod
from fla.models import HybridQLTConfig

dataset_registry: dict[str, Type[DataGen]] = {
    "form_language": FormLangDatasetGenerator
}

torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def create_save_directory(cfg, seed):
    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a directory name
    dir_name = f"results/saved_models_{cfg.model.name}_{timestamp}_seed_{seed}"

    # Create the directory
    os.makedirs(dir_name, exist_ok=True)

    return dir_name


def save_wandb_run_id(save_dir):
    run_id = wandb.run.id
    with open(os.path.join(save_dir, "wandb_run_id.txt"), "w") as f:
        f.write(run_id)


def check_nan_inf(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Warning: NaN or Inf detected in {name}")
        return True
    return False


def get_available_dtype(device):
    if device == 'cuda':
        # check that device supports bfloat16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            print('Warning: bfloat16 is not supported on this device. Using float32 instead.')
            return torch.float16
    elif device == 'mps':
        return torch.float32
    else:
        return torch.bfloat16


def load_dataset(name, kwargs):
    cls = dataset_registry[name]
    return cls(from_dict(cls.config_class, OmegaConf.to_container(kwargs)))


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(cfg: DictConfig, seed: int, lr: float, batch_size: int,
         grad_clip_norm: float, weight_decay: float,
         d_model: int, n_layers: int, n_heads: int, chunk_size: int,
         initializer_range: float, num_steps: int,
         wand_name: str, val_every_step: int):
    print(OmegaConf.to_yaml(cfg))
    save_dir = create_save_directory(cfg, seed)
    cfg.dataset.kwargs.seed = seed
    cfg.training.lr = lr
    cfg.training.batch_size = batch_size
    cfg.training.num_steps = num_steps
    cfg.training.val_every_step = val_every_step

    cfg.training.grad_clip_norm = grad_clip_norm
    cfg.training.weight_decay = weight_decay

    cfg.model.d_model = d_model
    cfg.model.n_layers = n_layers
    cfg.model.n_heads = n_heads
    cfg.model.chunk_size = chunk_size
    cfg.model.initializer_range = initializer_range

    # Initialize wandb
    wandb.init(project=wand_name, config=OmegaConf.to_container(cfg))
    seed_everything(cfg.dataset.kwargs.seed)
    # cfg.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = load_dataset(cfg.dataset.name, cfg.dataset.kwargs)

    train_loader = DataLoader(dataset.train_split, batch_size=cfg.training.batch_size)
    val_loaders = {
        key: DataLoader(val_ds, batch_size=cfg.training.batch_size) for key, val_ds in dataset.validation_split.items()
    }
    train_metrics = dataset.train_metrics.to(device=cfg.training.device)
    val_metrics = dataset.validation_metrics.to(device=cfg.training.device)
    if cfg.model.name == 'delayed_chunk_hqlt':
        config = HybridQLTConfig()
        config.hidden_size = cfg.model.d_model
        config.num_hidden_layers = cfg.model.n_layers
        config.num_heads = cfg.model.n_heads
        config.vocab_size = cfg.dataset.kwargs.vocab_size
        config.use_short_conv = cfg.model.use_short_conv
        config.allow_neg_eigval = cfg.model.allow_neg_eigval
        config.chunk_size = cfg.model.chunk_size
        config.use_memory_gate = cfg.model.use_memory_gate
        config.qk_activation = cfg.model.qk_activation
        config.qk_norm = cfg.model.qk_norm
        config.initializer_range = cfg.model.initializer_range
        config.attn_mode = 'chunk'
        model = HybridQLTForCausalLMMod(config).to(cfg.training.device)
    elif cfg.model.name == 'delayed_hqlt':
        config = HybridQLTConfig()
        config.hidden_size = cfg.model.d_model
        config.num_hidden_layers = cfg.model.n_layers
        config.num_heads = cfg.model.n_heads
        config.vocab_size = cfg.dataset.kwargs.vocab_size
        config.use_short_conv = cfg.model.use_short_conv
        config.allow_neg_eigval = cfg.model.allow_neg_eigval
        config.chunk_size = cfg.model.chunk_size
        config.use_memory_gate = cfg.model.use_memory_gate
        config.qk_activation = cfg.model.qk_activation
        config.qk_norm = cfg.model.qk_norm
        config.initializer_range = cfg.model.initializer_range
        config.attn_mode = 'delayed'
        model = HybridQLTForCausalLMMod(config).to(cfg.training.device)
    elif cfg.model.name == 'sync_hqlt':
        config = HybridQLTConfig()
        config.hidden_size = cfg.model.d_model
        config.num_hidden_layers = cfg.model.n_layers
        config.num_heads = cfg.model.n_heads
        config.vocab_size = cfg.dataset.kwargs.vocab_size
        config.use_short_conv = cfg.model.use_short_conv
        config.allow_neg_eigval = cfg.model.allow_neg_eigval
        config.chunk_size = cfg.model.chunk_size
        config.use_memory_gate = cfg.model.use_memory_gate
        config.qk_activation = cfg.model.qk_activation
        config.qk_norm = cfg.model.qk_norm
        config.initializer_range = cfg.model.initializer_range
        config.attn_mode = 'sync_chunk'
        model = HybridQLTForCausalLMMod(config).to(cfg.training.device)
    elif cfg.model.name == 'sync_linear_hqlt':
        config = HybridQLTConfig()
        config.hidden_size = cfg.model.d_model
        config.num_hidden_layers = cfg.model.n_layers
        config.num_heads = cfg.model.n_heads
        config.vocab_size = cfg.dataset.kwargs.vocab_size
        config.use_short_conv = cfg.model.use_short_conv
        config.allow_neg_eigval = cfg.model.allow_neg_eigval
        config.chunk_size = cfg.model.chunk_size
        config.use_memory_gate = cfg.model.use_memory_gate
        config.qk_activation = cfg.model.qk_activation
        config.qk_norm = cfg.model.qk_norm
        config.initializer_range = cfg.model.initializer_range
        config.attn_mode = 'sync_chunk_linear'
        model = HybridQLTForCausalLMMod(config).to(cfg.training.device)
    else:
        assert False, f'Unknown model {cfg.model.name}'
    if cfg.training.compile:
        print('Compiling model...')
        model = torch.compile(model, mode='default')
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()
    
    weights_dtype = torch_dtype_map[cfg.training.weight_precision]
    available_dtype = get_available_dtype(cfg.training.device)
    model = model.to(dtype=weights_dtype)

    wandb.config.update({"dtype_used": str(available_dtype), "weights_dtype": str(weights_dtype)})

    if hasattr(model, '_create_weight_decay_optim_groups'):
        optim_groups = model._create_weight_decay_optim_groups()
    else:
        optim_groups = []
        optim_groups.append(
            [param for name, param in model.named_parameters() if not hasattr(param, '_no_weight_decay')])
        optim_groups.append([param for name, param in model.named_parameters() if hasattr(param, '_no_weight_decay')])

    optimizer = optim.AdamW(
        (
            {"weight_decay": cfg.training.weight_decay, "params": optim_groups[0]},
            {"weight_decay": 0.0, "params": optim_groups[1]},
        ),
        lr=cfg.training.lr,
    )
    lr_scheduler = LinearWarmupCosineAnnealing(
        optimizer,
        cfg.training.lr_warmup_steps,
        cfg.training.lr_decay_until_steps,
        cfg.training.lr,
        cfg.training.lr_decay_factor * cfg.training.lr,
    )

    # Training loop
    step = 0
    epoch = 1
    running_loss = 0.0

    while step < cfg.training.num_steps:
        monitoring = tqdm(train_loader, total=0, initial=0)
        for inputs, labels in monitoring:
            monitoring.set_description_str(f"Steps {step + 1}/{cfg.training.num_steps} (Epoch: {epoch})")
            inputs = inputs.to(device=cfg.training.device)
            labels = labels.to(device=cfg.training.device)

            model.train()
            optimizer.zero_grad()
            with torch.autocast(
                    device_type=cfg.training.device,
                    dtype=available_dtype,
                    enabled=cfg.training.enable_mixed_precision,
            ):
                # Inside the training loop
                if check_nan_inf(inputs, "inputs") or check_nan_inf(labels, "labels"):
                    print(f"Warning: NaN or Inf in input data at step {step}. Skipping this batch.")
                    continue

                outputs = model(inputs.to(device=cfg.training.device))
                loss = nn.functional.cross_entropy(outputs.view(-1, cfg.model.vocab_size), labels.view(-1), ignore_index=-1)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN or Inf loss: {loss} encountered at step {step}. Skipping this batch.")
                    continue
                loss.backward()
                optimizer.step()
                # After optimizer.step()
                for name, param in model.named_parameters():
                    if check_nan_inf(param.data, f"parameter {name}"):
                        print(f"Warning: NaN or Inf in model parameters after update at step {step}")
                        break
                lr_scheduler.step()
                running_loss = loss
            step += 1
            train_metrics.update(outputs, labels)
            if step % cfg.training.val_every_step == 0:
                print(
                    f"\nStep [{step + 1}/{cfg.training.num_steps}] (Epoch: {epoch}), Loss: {running_loss:.4f},"
                    f" Metrics: {train_metrics.compute()}"
                )
                # Log training metrics to wandb
                wandb.log({
                    "step": step,
                    "epoch": epoch,
                    "train_loss": running_loss,
                    **{f"train_{k}": v for k, v in train_metrics.compute().items()}
                })
                train_metrics.reset()

                # Validation loop
                for vl_name, val_loader in val_loaders.items():
                    model.eval()
                    val_loss = 0.0
                    val_metrics.reset()
                    with torch.no_grad():
                        for val_inputs, val_labels in val_loader:
                            val_inputs = val_inputs.to(device=cfg.training.device)
                            val_labels = val_labels.to(device=cfg.training.device)
                            with torch.autocast(
                                    device_type=cfg.training.device,
                                    dtype=available_dtype,
                                    enabled=cfg.training.enable_mixed_precision,
                            ):
                                val_outputs = model(val_inputs)
                                loss = nn.functional.cross_entropy(
                                    val_outputs.view(-1, cfg.model.vocab_size),
                                    val_labels.view(-1),
                                    ignore_index=-1,
                                )
                                val_loss += loss.item()
                                val_metrics.update(val_outputs, val_labels)
                        val_loss /= len(val_loader)
                        print(
                            f"Validation[{vl_name}] Loss: {val_loss:.4f},"
                            f" Metrics: {val_metrics.compute()}"
                        )
                        metric_dict = {
                            "step": step,
                            f"val_{vl_name}_loss": val_loss,
                            **{f"val_{vl_name}_{k}": v for k, v in val_metrics.compute().items()}
                        }
                        '''
                        if cfg.model.name == "simple_recurrent" and cfg.model.layer_type == "diagonal":
                            sharpening_factors = {
                                f'sharpening_factor_{i}': layer.recurrent_layer.sharpening_factor.item() for
                                i, layer in zip(range(len(model.block_stack)), model.block_stack)}
                            metric_dict.update(sharpening_factors)

                            sharpening_factors_grad = {
                                f'sharpening_factor_{i}_grad': layer.recurrent_layer.sharpening_factor.grad.item() for
                                i, layer in zip(range(len(model.block_stack)), model.block_stack)}
                            metric_dict.update(sharpening_factors_grad)
                        '''
                        # Log validation metrics to wandb
                        wandb.log(metric_dict)

            if step >= cfg.training.num_steps:
                break
        epoch += 1

    # Save the model at the end of training
    model_save_path = os.path.join(save_dir, f"model_{cfg.model.name}_seed_{seed}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Log the saved model file to wandb
    wandb.save(model_save_path)

    # Save the configuration
    config_save_path = os.path.join(save_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)
    print(f"Configuration saved to {config_save_path}")
    save_wandb_run_id(save_dir)

    # Log the saved configuration file to wandb
    wandb.save(config_save_path)

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="parity_xlstm11")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=0.001, type=float)

    parser.add_argument("--grad_clip_norm", default=None)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--d_model", default=128, type=int)
    parser.add_argument("--n_layers", default=3, type=int)
    parser.add_argument("--n_heads", default=4, type=int)
    parser.add_argument("--chunk_size", default=8, type=int)
    parser.add_argument("--initializer_range", default=0.02, type=float)
    parser.add_argument("--num_steps", default=100000, type=int)
    parser.add_argument("--val_every_step", default=200, type=int)

    parser.add_argument("--wand_name", default="xlstm-synthetic")

    args = parser.parse_args()

    if args.grad_clip_norm is not None:
        try:
            args.grad_clip_norm = float(args.grad_clip_norm)
        except ValueError:
            raise ValueError("grad_clip_norm must be a float")

    with open(args.config, "r", encoding="utf8") as fp:
        config_yaml = fp.read()
    cfg = OmegaConf.create(config_yaml)
    OmegaConf.resolve(cfg)
    main(cfg, args.seed, args.lr, args.batch_size,
         args.grad_clip_norm, args.weight_decay,
         args.d_model, args.n_layers, args.n_heads,
         args.chunk_size, args.initializer_range,
         args.num_steps, args.wand_name, args.val_every_step)
