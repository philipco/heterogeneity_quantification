import torch
from matplotlib import pyplot as plt

from src.data.DataLoader import generate_client_models
from src.data.DatasetConstants import NB_CLIENTS
from src.data.SyntheticDataset import SyntheticLSRDataset
from src.utils.Utilities import get_project_root, create_folder_if_not_existing

def compute_effective_variance(datasets, lbda, epsilon):
    nb_clients = len(datasets)
    b, c = [[] for _ in range(nb_clients)], [[] for _ in range(nb_clients)]
    effective_cluster = [0 for _ in range(nb_clients)]
    for i in range(nb_clients):
        A, xi, mu  = datasets[i].covariance, datasets[i].variation, datasets[i].mu
        prod = A @ xi
        for j in range(nb_clients):
            A_prime = datasets[j].covariance
            xi_prime = datasets[j].variation
            bik = 2 * torch.linalg.norm(prod - A_prime @ xi_prime, ord=2)**2
            cik = 2 * torch.linalg.svd(torch.eye(A.shape[0]) - A_prime @ torch.linalg.pinv(A)).S[0]**2
            b[i].append(bik)
            c[i].append(cik)
            if lbda <= 1 - bik / (2 * mu * epsilon) - cik:
                effective_cluster[i] += 1
    return effective_cluster, b, c

COLORS = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:brown', 'tab:purple']
MARKERS = ['o', 's', 'D', '^', 'v', '<']

dataset_name = "synth"

# Initialize experiment
d = 2  # Input dimension  # Ground truth parameter
batch_size = 1
num_clients = NB_CLIENTS[dataset_name]

powers = [-5, -4,-3,-2,-1,0,1]
lambdas = [0.0] #[0, 10**-10, 10**-5, 10**-1, 0.5]
intra_var = [0.001, 0.01, 0.1, 1]

fig, axes = plt.subplots(1, 2, figsize=(2*8, 1.5))

idx = -1
for v in intra_var:
    idx += 1
    true_theta, variations = generate_client_models(num_clients, 1, d, cluster_variance=v)
    steps = 250

    # Create datasets and compute Lipschitz constant
    datasets = [SyntheticLSRDataset(m, v, batch_size) for (m, v) in zip(true_theta, variations)]

    # Compute the effective variance.
    for l in lambdas:
        len_clusters = []
        for power in powers:
            effective_cluster, b, c = compute_effective_variance(datasets, lbda=l, epsilon=10**power)
            len_clusters.append([effective_cluster[0], effective_cluster[1]])
        axes[0].plot(powers, [l[0] for l in len_clusters], label=rf"$v={v}$", linewidth=2, color=COLORS[idx], marker=MARKERS[idx])
        axes[1].plot(powers, [l[1] for l in len_clusters], label=rf"$v={v}$", linewidth=2, color=COLORS[idx], marker=MARKERS[idx])

axes[0].set_ylabel(r"$\mathcal{N}_1^*(\varepsilon, 10^{-2})$")
axes[1].set_ylabel(r"$\mathcal{N}_2^*(\varepsilon, 10^{-2})$")
axes[0].set_xlabel(r"Precision $\varepsilon$")
axes[1].set_xlabel(r"Precision $\varepsilon$")
axes[0].legend()

root = get_project_root()
folder = f'{root}/pictures/convergence/{dataset_name}'
create_folder_if_not_existing(folder)
plt.savefig(f"{folder}/effective_clusters.pdf", bbox_inches='tight', dpi=600)
