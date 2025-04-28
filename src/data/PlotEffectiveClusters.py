from matplotlib import pyplot as plt

from src.data.DataLoader import generate_client_models
from src.data.DatasetConstants import NB_CLIENTS
from src.data.Network import compute_effective_variance
from src.data.SyntheticDataset import SyntheticLSRDataset
from src.utils.Utilities import get_project_root, create_folder_if_not_existing

COLORS = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:brown', 'tab:purple']
MARKERS = ['o', 's', 'D', '^', 'v', '<']

dataset_name = "synth"

# Initialize experiment
d = 2  # Input dimension  # Ground truth parameter
batch_size = 1
num_clients = NB_CLIENTS[dataset_name]

powers = [-5, -4,-3,-2,-1,0,1]
lambdas = [0.01] #[0, 10**-10, 10**-5, 10**-1, 0.5]
intra_var = [0.001, 0.01, 0.1, 1]

fig, axes = plt.subplots(2, 1, figsize=(8, 2*2))

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

axes[0].set_ylabel(r"$\mathcal{N}_0^*(\varepsilon, 10^{-2})$")
axes[1].set_ylabel(r"$\mathcal{N}_1^*(\varepsilon, 10^{-2})$")
axes[1].set_xlabel(r"Precision $\varepsilon$")
axes[0].legend()

root = get_project_root()
folder = f'{root}/pictures/convergence/{dataset_name}'
create_folder_if_not_existing(folder)
plt.savefig(f"{folder}/effective_clusters.pdf", bbox_inches='tight', dpi=600)
