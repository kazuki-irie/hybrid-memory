import time

import cola
import matplotlib.pyplot as plt
import numpy as np
import torch
from linear_operator.operators import DiagLinearOperator, LowRankRootLinearOperator


def householder_matrix_linear_operator(v):
    n = v.size(0)
    I = DiagLinearOperator(torch.ones(n, device=v.device))
    vvT = LowRankRootLinearOperator(v.unsqueeze(1))
    return I - 2 * vvT / (v @ v)


def householder_matrix_cola(v):
    n = v.size(0)
    I = cola.ops.Diagonal(torch.tensor([1.0] * n, device=v.device))
    v_cola = cola.lazify(v.unsqueeze(1))
    vvT = v_cola @ v_cola.T
    scalar = - 2 * (v @ v)
    return cola.ops.Sum(I, vvT / scalar)


def generate_householder_vector(n):
    v = torch.randn(n)
    return v / torch.norm(v)


def generate_householder_product(n, k, device, library='linear_operator'):
    start_time = time.time()
    vectors = [generate_householder_vector(n).to(device) for _ in range(k)]
    if library == 'linear_operator':
        product = householder_matrix_linear_operator(vectors[0])
        for v in vectors[1:]:
            product = product @ householder_matrix_linear_operator(v)
    elif library == 'cola':
        product = householder_matrix_cola(vectors[0])
        for v in vectors[1:]:
            product = product @ householder_matrix_cola(v)
    else:  # normal PyTorch
        product = torch.eye(n, device=device)
        for v in vectors:
            product = product @ (torch.eye(n, device=device) - 2 * torch.outer(v, v) / (v @ v))
    end_time = time.time()
    return product, end_time - start_time


def benchmark(n, k_range, device, library='linear_operator', num_runs=5):
    times = []
    for k in k_range:
        k_times = []
        for _ in range(num_runs):
            _, total_time = generate_householder_product(n, k, device, library)
            k_times.append(total_time)
        times.append(k_times)
    return np.array(times)


def run_benchmarks():
    n = 1000  # Fixed matrix size
    k_range = [1, 5, 10, 20, 50, 100]  # Number of matrices in the product
    num_runs = 5
    devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu', 'mps']

    results = {}
    for device in devices:
        results[device] = {
            'LinearOperator': benchmark(n, k_range, device, 'linear_operator', num_runs=num_runs),
            'CoLA': benchmark(n, k_range, device, 'cola', num_runs=num_runs),
            'Normal MatMul': benchmark(n, k_range, device, 'normal', num_runs=num_runs)
        }

    return results, k_range


def plot_results(results, k_range):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for i, (device, data) in enumerate(results.items()):
        ax = axes[i]
        for method, times in data.items():
            mean_times = np.mean(times, axis=1)
            std_times = np.std(times, axis=1)
            ax.errorbar(k_range, mean_times, yerr=std_times, capsize=5, marker='o', label=method)

        ax.set_xlabel('Number of Matrices in Product')
        ax.set_ylabel('Total Time (seconds)')
        ax.set_title(f'Total Computation Time on {device.upper()} (n=1000)')
        ax.legend()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('householder_product_comparison.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    results, k_range = run_benchmarks()
    plot_results(results, k_range)
