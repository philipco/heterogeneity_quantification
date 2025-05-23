import numpy as np
import scipy
from numpy import linalg as LA
import ot
from scipy.stats import ranksums
import torch
import torch.nn as nn

from src.outofdate.Data import Data,  DataCentralized
from src.data.Network import Network
from src.optim.PytorchUtilities import aggregate_models
from src.outofdate.AbstractTest import ProportionTest, compute_atomic_errors, Mannwhitneyu, RanksumsTest
from src.outofdate.Metrics import Metrics


def compute_KL_distance(distrib1: np.ndarray, distrib2: np.ndarray) -> float:
    """Kullback-Leibler."""
    assert len(distrib1) == len(distrib2), "KL: distributions are not of equal size."
    EPS = 10e-6
    N = len(distrib1)
    return np.sum([distrib1[i] * np.log((distrib1[i] / (distrib2[i]+ EPS)) + EPS) for i in range(N)])


def compute_TV_distance(distrib1: np.ndarray, distrib2: np.ndarray) -> float:
    """Total Variation."""
    assert len(distrib1) == len(distrib2), "TV: distributions are not of equal size."
    N = len(distrib1)
    return 0.5 * np.sum([np.abs(distrib1[i] - distrib2[i]) for i in range(N)])


def sub_sample(distrib, nb_sample):
    if nb_sample == distrib.shape[0]:
        return distrib
    idx = np.random.randint(distrib.shape[0], size=nb_sample)
    return distrib[idx, :]


def compute_EM_distance(distrib1: np.ndarray, distrib2: np.ndarray, stochastic: bool) -> float:
    """Earth's mover."""
    assert len(distrib1) >= 1 and len(distrib2) >= 1, "Distributions must not be empty."
    assert len(distrib1[0]) <= 20, "Dimension is bigger than 20."
    nb_sample1, nb_sample2 = distrib1.shape[0], distrib2.shape[0]

    percent_kept = 0.05 if stochastic else 1
    batch_size1, batch_size2 = int(nb_sample1 * percent_kept), int(percent_kept * nb_sample2)

    minibatch_emd = []

    for k in range(int(2 * percent_kept ** -1)):
        np.random.seed(k)
        sub_distrib1 = sub_sample(distrib1, batch_size1)
        np.random.seed(k)
        sub_distrib2 = sub_sample(distrib2, batch_size2)

        cost_matrix = ot.dist(sub_distrib1, sub_distrib2)
        a, b = ot.unif(len(sub_distrib1)), ot.unif(len(sub_distrib2))  # uniform distribution on samples
        # a = torch.ones(len(sub_distrib1)) / len(sub_distrib1)
        # b = torch.ones(len(sub_distrib2)) / len(sub_distrib2)
        minibatch_emd.append(ot.emd2(a, b, cost_matrix))  # Wasserstein distance / EMD value
    return np.average(minibatch_emd)
    # et si je n'averagais pas ?


def reconstruction_error(estimator, X, y=None):
    return LA.norm(X - estimator.inverse_transform(estimator.transform(X))) / len(X)


def compute_PCA_error(PCA_fitted, distrib: np.ndarray):
    return reconstruction_error(PCA_fitted, distrib)


def compute_LSR_distance(net, test_loss, features, labels):
    criterion = nn.MSELoss()
    with torch.no_grad():
        outputs = net(features)
        remote_loss = criterion(outputs, labels)
    return abs(remote_loss - test_loss) / test_loss


def compute_distance_based_on_ranksums(net, criterion, remote_atomic_test_loss, features, labels):
    """Return the pvalue of the ranksums test between the errors' distribution on current client and remote."""
    outputs = net(features)
    atomic_test_losses = criterion(outputs, labels)
    #  The  distribution of remote atomic losses is stochastically greater than the distribution of losses
    #  computed on current client using remote model.
    ranktest = ranksums(remote_atomic_test_loss.detach().numpy(), atomic_test_losses.detach().numpy(), alternative='two-sided')
    return ranktest.pvalue

def compute_distance_based_on_nn(net, criterion, test_loss, features, labels):
    """Returns the difference between local loss and loss computed using remote models. """
    outputs = net(features)
    remote_loss = criterion(outputs, labels)
    return remote_loss - test_loss # TODO : Flexibilité !


def compute_distance_based_on_cond_var_pvalue(local_loss, remote_loss, total_params, local_nb_points, remote_nb_points):
    """A global homogeneity test for high-dimensional linear regression, Charbonnier, Verzelen, Villers, 2013,
    Electronic Journal of Statistics."""
    def g_inverse(x):
        return  (x + 2 - np.sqrt(x * (x + 4))) / 2
    ratio = local_loss / remote_loss
    local_fisher_df, remote_fisher_df = local_nb_points - total_params, remote_nb_points - total_params
    if local_fisher_df == 0:
        print(local_nb_points)
    coef = local_nb_points * remote_fisher_df / (remote_nb_points * local_fisher_df)
    statistics = -2 + ratio.item() + 1/ ratio.item()
    return (scipy.stats.f.cdf(g_inverse(statistics) * coef, local_fisher_df, remote_fisher_df)
            + scipy.stats.f.cdf(g_inverse(statistics) / coef, remote_fisher_df, local_fisher_df))


def majoration_proba(spectre, N, u):
    if u < torch.sum(spectre):
        return 0 # After we compute the exponential
    else:
        norme_infini = torch.max(spectre)
        norme_1 = torch.linalg.norm(spectre, ord=1)
        norme_2 = torch.linalg.norm(spectre, ord=2)

        if torch.sum(spectre / norme_infini) == torch.sum((spectre / norme_infini) ** 2):
            lambda_val = (u - norme_1) / (2 * u * (norme_infini + norme_1 / N))
        else:
            b = norme_1 * u / (norme_infini * N) + u + norme_2 / norme_infini - norme_1
            delta = b ** 2 - 4 * u * (u - norme_1) * (norme_1 - norme_2 / norme_infini) / (N * norme_infini)
            lambda_val = (b - np.sqrt(delta)) / (4 * u * (norme_1 - norme_2 / norme_infini) / N)

        proba = torch.sum(np.log(1 - 2 * lambda_val * spectre)) / 2 + N / 2 * np.log(1 + 2 * lambda_val * u / N)
        return proba


def compute_distance_based_on_quantile_pvalue(criterion, dataloader, local_net, remote_net, n_loc, n_rem, lbda: int = 0.5):
    beta0 = 0.8

    atomic_errors = compute_atomic_errors(local_net, dataloader, criterion)
    q0 = np.quantile(atomic_errors.cpu(), beta0, method="higher")

    test = ProportionTest(beta0, delta=0, loss=criterion)

    # weights = [n_loc / (n_loc + n_rem), n_rem / (n_loc + n_rem)]
    weights = [lbda, 1 - lbda]
    avg_model = aggregate_models([local_net, remote_net], weights,
                                 next(local_net.parameters()).device)

    test.evaluate_test(q0, avg_model, dataloader)
    # plot_arrow_with_atomic_errors(atomic_errors.cpu().numpy(), test.atomic_errors.cpu().numpy(), test.beta0, test.pvalue, 1, "test")

    # print(f"pvalue={test.pvalue}")
    return test.pvalue


def function_to_compute_ranksums_pvalue(network: Network, i: int, j: int):
    d_iid = compute_distance_based_on_ranksums(network.clients[i].trained_model, network.criterion(reduction='none'),
                                               network.clients[i].atomic_test_losses, network.clients[j].X_test,
                                               network.clients[j].Y_test)
    d_heter = compute_distance_based_on_ranksums(network.clients[i].trained_model, network.criterion(reduction='none'),
                                                 network.clients[i].atomic_test_losses, network.clients[j].X_test,
                                                 network.clients[j].Y_test)
    return d_iid, d_heter


def function_to_compute_nn_distance(network: Network, i: int, j: int):
    d_iid = compute_distance_based_on_nn(network.clients[i].trained_model, network.criterion(), network.clients[i].test_loss,
                                         network.clients[j].X_test, network.clients[j].Y_test)
    d_heter = compute_distance_based_on_nn(network.clients[i].trained_model, network.criterion(), network.clients[i].test_loss,
                                           network.clients[j].X_test, network.clients[j].Y_test)
    return d_iid, d_heter


def function_to_compute_cond_var_pvalue(network: Network, i: int, j: int):

    # All clients have the same number of models.
    total_params = sum(p.numel() for p in network.clients[0].trained_model.parameters() if p.requires_grad)
    # d_iid = compute_distance_based_on_cond_var_pvalue(network.clients[i].train_loss, network.clients[j].train_loss,
    #                                                   total_params,
    #                                                   len(network.clients[i].train_labels),
    #                                                   len(network.clients[j].train_labels))
    d_heter = compute_distance_based_on_cond_var_pvalue(network.clients[i].train_loss, network.clients[j].train_loss,
                                                        total_params,
                                                        len(network.clients[i].Y_train),
                                                        len(network.clients[j].Y_train))
    return 0, d_heter


def acceptance_pvalue(network: Network, i: int, j: int, lbda=0.0):
    test = Mannwhitneyu(loss=network.criterion(reduction='none'))
    weights = [lbda, 1 - lbda]
    avg_model = aggregate_models([network.clients[i].trained_model, network.clients[j].trained_model], weights,
                     next(network.clients[i].trained_model.parameters()).device)
    p = test.evaluate_test(network.clients[i].trained_model, avg_model,
                           network.clients[i].val_loader, alternative="less")
    return p

def rejection_pvalue(network: Network, i: int, j: int):
    test = RanksumsTest(loss=network.criterion(reduction='none'))
    p1 = test.evaluate_test(network.clients[i].trained_model, network.clients[i].val_loader,
                            network.clients[j].val_loader)
    p2 = test.evaluate_test(network.clients[j].trained_model, network.clients[j].val_loader,
                            network.clients[i].val_loader)
    return min(p1, p2)


def function_to_compute_LSR_error(network: Network, i: int, j: int):
    d_iid = compute_LSR_distance(data.all_models.models_iid[i], data.all_models.test_loss_iid[i],
                                 data.features_iid[j], data.labels_iid[j])
    d_heter = compute_LSR_distance(data.all_models.models_heter[i], data.all_models.test_loss_heter[i],
                                   data.features_heter[j], data.labels_heter[j])
    return d_iid, d_heter


def function_to_compute_TV_error(network: Network, i: int, j: int):
    d_iid = compute_TV_distance(data.labels_iid_distrib[i], data.labels_iid_distrib[j])
    d_heter = compute_TV_distance(data.labels_heter_distrib[i], data.labels_heter_distrib[j])
    return d_iid, d_heter


def function_to_compute_PCA_error(network: Network, i: int, j: int):
    d_iid = compute_PCA_error(data.PCA_fit_iid[i], data.features_iid[j])
    d_heter = compute_PCA_error(data.PCA_fit_heter[i], data.features_heter[j])
    return d_iid, d_heter


def function_to_compute_EM(data: DataCentralized, i: int, j: int):
    stochastic_emd = False if max([distrib.shape[0] for distrib in data.features_heter]) <= 1000 else True
    d_iid = compute_EM_distance(data.features_iid[i], data.features_iid[j], stochastic=stochastic_emd)
    d_heter = compute_EM_distance(data.features_heter[i], data.features_heter[j], stochastic=stochastic_emd)
    return d_iid, d_heter


def compute_matrix_of_distances(fonction_to_compute_distance, data: Data, metrics: Metrics,
                                symetric_distance: bool = True) -> Metrics:
    distances_heter = np.zeros((data.nb_clients, data.nb_clients))
    for i in range(data.nb_clients):
        start = i if symetric_distance else 0
        for j in range(start, data.nb_clients):
            # Model trained on client i
            # Evaluated on test set from client j
            d_heter = fonction_to_compute_distance(data, i, j)
            distances_heter[i, j] = d_heter

            if symetric_distance:
                distances_heter[j, i] = d_heter

    metrics.add_distances(distances_heter)
    return metrics
