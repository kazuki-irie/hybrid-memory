import os
from argparse import ArgumentParser

import matplotlib.font_manager as font_manager  # noqa
import matplotlib.pyplot as plt
import scienceplots  # noqa
import seaborn as sns
import torch
from dacite import from_dict
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import wandb
from experiments.data.formal_language.formal_language_dataset import FormLangDatasetGenerator
from simple_recurrent.lm_model import SimpleRecurrentNet
from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig

# Set global plot style
plt.style.use(['science', 'no-latex', 'light'])
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def bootstrap_ci(data, num_bootstraps=1000, ci=95):
    bootstrapped_means = []
    for _ in range(num_bootstraps):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(boot_sample))
    return np.percentile(bootstrapped_means, [(100 - ci) / 2, 50 + ci / 2])


def create_accuracy_vs_length_plot(sequence_lengths, sequence_accuracies, train_sequence_length):
    # Calculate average accuracies and confidence intervals
    unique_lengths = sorted(set(sequence_lengths))
    avg_accuracies = []
    ci_lower = []
    ci_upper = []

    for length in unique_lengths:
        accuracies = [acc for len, acc in zip(sequence_lengths, sequence_accuracies) if len == length]
        avg_accuracies.append(np.mean(accuracies))
        ci = bootstrap_ci(accuracies)
        ci_lower.append(ci[0])
        ci_upper.append(ci[1])

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(unique_lengths, avg_accuracies, marker='o', linestyle='-', color='blue', label='Average Accuracy')
    plt.fill_between(unique_lengths, ci_lower, ci_upper, color='blue', alpha=0.2, label='95% CI')

    plt.axvline(x=train_sequence_length, color='red', linestyle='--', label='Train Sequence Length')

    plt.xlabel('Sequence Length')
    plt.ylabel('Accuracy')
    plt.title('Average Accuracy vs Sequence Length')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    return plt.gcf()


def load_dataset(name, kwargs):
    cls = FormLangDatasetGenerator
    return cls(from_dict(cls.config_class, OmegaConf.to_container(kwargs)))


def get_available_dtype(device):
    if device == 'cuda':
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device == 'mps':
        return torch.float32
    else:
        return torch.bfloat16


def evaluate_model(model, test_loader, device, vocab_size):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    sequence_lengths = []
    sequence_accuracies = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-1,
            )
            total_loss += loss.item()

            predictions = outputs.argmax(dim=-1)
            mask = labels != -1
            correct = (predictions[mask] == labels[mask]).float()

            # Calculate sequence lengths and accuracies
            for i in range(inputs.size(0)):
                seq_len = (inputs[i] != 0).sum().item()
                seq_correct = correct[i].sum().item()
                seq_total = mask[i].sum().item()

                sequence_lengths.append(seq_len)
                sequence_accuracies.append(seq_correct / seq_total if seq_total > 0 else 0)

            correct_predictions += correct.sum().item()
            total_predictions += mask.sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    return avg_loss, accuracy, sequence_lengths, sequence_accuracies


def create_improved_plots(sequence_lengths, sequence_accuracies, model_name):
    # Calculate average accuracies and confidence intervals
    avg_accuracies = {}
    for length, accuracy in zip(sequence_lengths, sequence_accuracies):
        if length not in avg_accuracies:
            avg_accuracies[length] = []
        avg_accuracies[length].append(accuracy)

    avg_lengths = list(avg_accuracies.keys())
    avg_accs = [np.mean(accs) for accs in avg_accuracies.values()]
    conf_intervals = [stats.t.interval(0.95, len(accs) - 1, loc=np.mean(accs), scale=stats.sem(accs))
                      for accs in avg_accuracies.values()]

    # Create main figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Model Performance Analysis: {model_name}', fontsize=16)

    # Scatter plot with trend line
    sns.regplot(x=sequence_lengths, y=sequence_accuracies, ax=ax1, scatter_kws={'alpha': 0.5},
                line_kws={'color': 'red'})
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Individual Sample Performance')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Line plot with confidence intervals
    ax2.plot(avg_lengths, avg_accs, marker='o', linestyle='-', linewidth=2, markersize=8)
    ax2.fill_between(avg_lengths,
                     [ci[0] for ci in conf_intervals],
                     [ci[1] for ci in conf_intervals],
                     alpha=0.3)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Average Accuracy')
    ax2.set_title('Average Performance with 95% Confidence Interval')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add annotations
    max_acc_idx = np.argmax(avg_accs)
    ax2.annotate(f'Peak: {avg_accs[max_acc_idx]:.3f}',
                 xy=(avg_lengths[max_acc_idx], avg_accs[max_acc_idx]),
                 xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    return fig


def main(directory_path):
    # Find the config file
    config_file = next((f for f in os.listdir(directory_path) if f.endswith('.yaml')), None)
    if not config_file:
        raise FileNotFoundError("No .yaml configuration file found in the directory.")
    cfg_path = os.path.join(directory_path, config_file)

    # Find the model file
    model_file = next((f for f in os.listdir(directory_path) if f.endswith('.pth')), None)
    if not model_file:
        raise FileNotFoundError("No .pth model file found in the directory.")
    model_path = os.path.join(directory_path, model_file)

    # Find the wandb ID file
    wandb_id_file = next((f for f in os.listdir(directory_path) if f.endswith('.txt')), None)
    if not wandb_id_file:
        raise FileNotFoundError("No .txt file containing wandb ID found in the directory.")

    # Read the wandb ID from the file
    with open(os.path.join(directory_path, wandb_id_file), 'r') as f:
        wandb_run_id = f.read().strip()

    # Load configuration
    with open(cfg_path, 'r') as f:
        cfg = OmegaConf.load(f)
    wandb.init(id=wandb_run_id, resume=True, entity='wandb_project', project='xlstm-training')

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    dataset = load_dataset(cfg.dataset.name, cfg.dataset.kwargs)
    test_dataset = dataset.test_split['test']
    print(f"Length of test dataset: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size)

    # Initialize model
    if cfg.model.name == "simple_recurrent":
        model = SimpleRecurrentNet(cfg.model).to(cfg.training.device)
    else:
        model = xLSTMLMModel(from_dict(xLSTMLMModelConfig, OmegaConf.to_container(cfg.model))).to(
            device=cfg.training.device
        )
    # Load model weights
    state_dict = torch.load(model_path, map_location=device)

    # Remove '_orig_mod.' prefix from keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v  # Remove first 10 characters ('_orig_mod.')
        else:
            new_state_dict[k] = v

    # Load the modified state dict
    model.load_state_dict(new_state_dict)
    # Set model to evaluation mode
    model.eval()

    # Set dtype
    available_dtype = get_available_dtype(device)
    model = model.to(dtype=available_dtype)

    # Evaluate model
    test_loss, test_accuracy, sequence_lengths, sequence_accuracies = evaluate_model(model, test_loader, device,
                                                                                     cfg.model.vocab_size)
    # Evaluate model
    test_loss, test_accuracy, sequence_lengths, sequence_accuracies = evaluate_model(model, test_loader, device,
                                                                                     cfg.model.vocab_size)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Create improved plots
    fig = create_improved_plots(sequence_lengths, sequence_accuracies, cfg.model.name)

    # Log metrics and plot to wandb
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "performance_analysis": wandb.Image(fig)
    })

    # Close the plot to free up memory
    plt.close(fig)

    # Additional heatmap for model size vs sequence length
    model_sizes = [1e6, 5e6, 10e6, 50e6]  # Example model sizes
    heatmap_data = np.random.rand(len(model_sizes), len(set(sequence_lengths)))  # Replace with actual data

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, xticklabels=sorted(set(sequence_lengths)), yticklabels=model_sizes,
                cmap='YlOrRd', annot=True, fmt='.2f')
    plt.xlabel('Sequence Length')
    plt.ylabel('Model Size (parameters)')
    plt.title('Accuracy Heatmap: Model Size vs Sequence Length')

    wandb.log({
        "model_size_vs_sequence_length": wandb.Image(plt)
    })
    plt.close()

    fig = create_accuracy_vs_length_plot(sequence_lengths, sequence_accuracies, cfg.dataset.kwargs.max_sequence_length)
    # Log metrics and plot to wandb
    wandb.log({
        "accuracy_vs_sequence_length": wandb.Image(fig)
    })

    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--directory", required=True,
                        help="Path to the directory containing config, model, and wandb ID files")

    args = parser.parse_args()

    main(args.directory)
