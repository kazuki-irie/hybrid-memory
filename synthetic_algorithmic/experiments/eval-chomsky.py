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
from mamba.mamba import MambaLM, MambaConfig
from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
try:
    from delta_net.delta_net import DeltaNetForCausalLMMod
    from fla.models import DeltaNetForCausalLM, DeltaNetConfig
except ImportError:
    pass

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


colors = {
    'xlstm10': 'blue',
    'xlstm01': 'orange',
    'mamba': 'blue',
    'mamba2': 'orange',
    'delta': 'blue',
    'delta2': 'orange',
}
labels = {
    'xlstm10': 'mLSTM',
    'xlstm01': 'sLSTM',
    'mamba': 'Mamba 1 [0, 1]',
    'mamba2': 'Mamba 1 [-1, 1]',
    'delta': 'DeltaNet [0, 1]',
    'delta2': 'DeltaNet [-1, 1]',
}
titles = {
    'parity': 'Parity',
    'mod': 'Mod. Arithmetic w/o Brackets',
    'modbra': 'Mod. Arithmetic w/ Brackets',
}

def bootstrap_ci(data, num_bootstraps=1000, ci=95):
    bootstrapped_means = []
    for _ in range(num_bootstraps):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(boot_sample))
    return np.percentile(bootstrapped_means, [(100 - ci) / 2, 50 + ci / 2])


def create_accuracy_vs_length_plot(ax, sequence_lengths, sequence_accuracies,
                                   train_sequence_length, method_name):
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

    task, method = method_name.split('-')

    ax.plot(unique_lengths, avg_accuracies, marker='o', linestyle='-',
            #color=colors[method], 
            label=labels[method])
    ax.fill_between(unique_lengths, ci_lower, ci_upper,
                    #color=colors[method], 
                    alpha=0.2)
    return ax


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


def evaluate_model(model, test_loader, device, vocab_size, task_name):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    sequence_lengths = []
    sequence_accuracies = []
    if task_name == 'modular_arithmetic':
        pad_token = 0
        scale = 0.2
    elif task_name == 'modular_arithmetic_with_brackets':
        pad_token = 11
        scale = 0.2
    elif task_name == 'parity':
        pad_token = 0
        scale = 0.5

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
            #from IPython import embed
            #embed()
            #import sys
            #sys.exit(0)
            for i in range(inputs.size(0)):
                seq_len = (inputs[i] != pad_token).sum().item()
                seq_correct = (correct[i].sum().item() - scale) / (1-scale)
                seq_total = mask[i].sum().item()

                sequence_lengths.append(seq_len)
                sequence_accuracies.append(seq_correct / seq_total if seq_total > 0 else 0)

            correct_predictions += correct.sum().item()
            total_predictions += mask.sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    accuracy = (accuracy - scale) / (1 - scale)

    return avg_loss, accuracy, sequence_lengths, sequence_accuracies

def get_data(directory_path):
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
        # change the min sequence length
        cfg.dataset.kwargs.subpar.test.min_sequence_length = 3
        cfg.dataset.kwargs.subpar.validation.min_sequence_length = 3
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
    elif cfg.model.name == 'delta_net':
        config = DeltaNetConfig()
        config.hidden_size = cfg.model.d_model
        config.sigmoid_scale = cfg.model.sigmoid_scale
        config.num_hidden_layers = cfg.model.n_layers
        config.num_heads = cfg.model.n_heads
        config.vocab_size = cfg.dataset.kwargs.vocab_size
        config.use_short_conv = cfg.model.use_short_conv
        model = DeltaNetForCausalLMMod(config).to(cfg.training.device)
    elif cfg.model.name == 'mamba':
        model = MambaLM(from_dict(MambaConfig, OmegaConf.to_container(cfg.model)), cfg.dataset.kwargs.vocab_size,
                        cfg.model.d_model, cfg.model.positive_and_negative).to(device=cfg.training.device)
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
    import pdb; pdb.set_trace()
    model.load_state_dict(new_state_dict)
    # Set model to evaluation mode
    model.eval()

    # Set dtype
    available_dtype = get_available_dtype(device)
    model = model.to(dtype=available_dtype)

    return model, test_loader, device, cfg.model.vocab_size, \
            cfg.dataset.kwargs.synth_lang_type, \
            cfg.dataset.kwargs.max_sequence_length

def plot_pairs(pair, dirs, results_path='results'):

    print(f'>>>> Evaluating {str(pair)}')
    fig, axis = plt.subplots(figsize=(3.2, 2))

    for config in pair:
        task = config.split('-')[0]
        directory_path = os.path.join(results_path, dirs[config])

        try:
            model, test_loader, device, vocab_size, synth_lang_type, max_seq_length = \
                    get_data(directory_path)
        except FileNotFoundError:
            print(f'File for {config} not found. Skipping.')
            return

        # Evaluate model
        test_loss, test_accuracy, sequence_lengths, sequence_accuracies = \
                evaluate_model(model, test_loader, device, vocab_size,
                               synth_lang_type)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        #from IPython import embed
        #embed()

        axis = create_accuracy_vs_length_plot(axis, sequence_lengths,
                                              sequence_accuracies,
                                              max_seq_length, config)

    axis.axvline(x=max_seq_length, color='red', linestyle='--')

    axis.set_xlabel('Sequence Length')
    axis.set_ylabel('Normalized Accuracy')
    axis.set_title(titles[task])
    axis.legend()
    #axis.set_scale('log')
    axis.grid(True, linestyle='--', alpha=0.7)
    # Log metrics and plot to wandb
    #wandb.log({
        #"accuracy_vs_sequence_length": wandb.Image(fig)
    #})

    save_path = 'plots'
    os.makedirs(save_path, exist_ok=True)

    fig.savefig(os.path.join(save_path, '_'.join(pair) + '.pdf'), dpi=300,
                             bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--directory", required=True,
                        help="Path to the directory containing config, model, and wandb ID files")

    args = parser.parse_args()

    dirs = {
        'parity-xlstm01': "saved_models_xlstm01_20240923_224552_seed_47",
        'parity-xlstm10': "saved_models_xlstm10_20240923_170728_seed_42",
        'parity-mamba': "saved_models_mamba_20240924_171815_seed_42",
        'parity-mamba2': "saved_models_mamba_20240924_174059_seed_48",
        'parity-delta': "saved_models_delta_net_20240928_122416_seed_48",
        'parity-delta2': "saved_models_delta_net_20240925_124617_seed_42",
        'mod-xlstm01': "saved_models_xlstm01_20240926_120310_seed_43",
        'mod-xlstm10': "saved_models_xlstm10_20240926_195706_seed_42",
        'mod-mamba': "saved_models_mamba_20240926_185022_seed_41",
        'mod-mamba2': "saved_models_mamba_20240927_202020_seed_44",
        'mod-delta': "saved_models_delta_net_20240926_184930_seed_41",
        'mod-delta2': "saved_models_delta_net_20240926_184932_seed_41",
        'modbra-xlstm01': "saved_models_xlstm01_20240928_120857_seed_47",
        'modbra-xlstm10': "saved_models_xlstm10_20240927_154733_seed_42",
        'modbra-mamba': "saved_models_mamba_20240927_154553_seed_43",
        'modbra-mamba2': "saved_models_mamba_20240929_133858_seed_51",
        'modbra-delta': "saved_models_delta_net_20240927_154152_seed_42",
        'modbra-delta2': "saved_models_delta_net_20240928_121009_seed_49"
    }
    #dirs = {
        #'mod-delta': "saved_models_delta_net_20240926_184930_seed_41",
        #'mod-delta2': "saved_models_delta_net_20240926_184932_seed_41",
    #}

    keys = list(dirs.keys())

    # Generate pairs of keys
    pairs = [
        [keys[i], keys[i+1]]
        for i in range(0, len(keys) - 1, 2)
    ]
    for pair in pairs:
        plot_pairs(pair, dirs, args.directory)

