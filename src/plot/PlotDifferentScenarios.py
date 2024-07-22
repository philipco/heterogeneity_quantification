import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib

from src.plot.PlotArrowWithAtomicErrors import plot_arrow_with_atomic_errors
from src.quantif.AbstractTest import ProportionTest, compute_atomic_errors

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

DELTA = 0.05
DEG = 4


def f(scenario: str, n=100):

    noise1 = np.random.normal(0, 0.05, n)
    noise2 = np.random.normal(0, 0.05, n)
    X1 = np.random.uniform(-1, 1, n)
    X2 = np.random.uniform(-1, 1, n)

    if scenario == "partionned_support":
        X2 = np.random.uniform(0.75, 1, n)
        Y1 = [x**2 for x in X1] + noise1
        Y2 = [x**2 for x in X2] + noise2
    elif scenario == "Y_shift":
        Y1 = [x ** 2 for x in X1] + noise1
        Y2 = [x ** 2 + 1 for x in X2] + noise2
    elif scenario == "X_shift":
        X1 = np.random.uniform(0, 1., n)
        X2 = np.random.uniform(1., 2, n)
        Y1 = [x for x in X1] + noise1
        Y2 = [(x-1) for x in X2] + noise2
    elif scenario == "same":
        Y1 = [x ** 2 for x in X1] + noise1
        Y2 = [x ** 2 for x in X2] + noise2 # * 10 if heterogeneous noise.
    elif scenario == "same_partionned_support":
        X1 = np.random.uniform(-1, 0., n)
        X2 = np.random.uniform(0., 1, n)
        Y1 = [x ** 2 for x in X1] + noise1
        Y2 = [x ** 2 for x in X2] + noise2
    elif scenario == "same_partionned_support_different_Y":
        X1 = np.random.uniform(0, 1, n)
        X2 = np.random.uniform(1., 2, n)
        Y1 = [x ** 2 for x in X1] + 5 * noise1
        Y2 = [x ** 2 for x in X2] + 5 * noise2
    elif scenario == "same_with_variations":
        Y1 = [x ** 2 for x in X1] + noise1
        Y2 = [x ** 2 + np.sin(10 * x) / 2 for x in X2] + noise2 # or div by 2 ?
    elif scenario == "different":
        Y1 = [(x - 1) ** 2 for x in X1] + noise1
        Y2 = [(x + 1) ** 2 for x in X2] + noise2
    elif scenario == "totally_different":
        X1 = np.random.uniform(-1, 0., n)
        X2 = np.random.uniform(0., 1, n)
        Y1 = [(2*x + 0.5) ** 2 + 0.5 for x in X1] + noise1
        Y2 = [-(2*x - 0.5) ** 2 for x in X2] + noise2
    elif scenario == "classification":
        n=100
        X1 = np.vstack((np.random.triangular(-2, -1, 2, n), np.random.uniform(0, 1., n))).T
        X2 = np.vstack((np.random.triangular(-2, 1, 2, n), np.random.uniform(0, 1., n))).T
        Y1 = [-1 if x[0] < 0 else 1 for x in X1]
        Y2 = [-1 if x[0] < 0 else 1 for x in X2]
    elif scenario == "various_noise":
        X2 = np.random.uniform(0.65, 1, n)
        Y1 = [x ** 2  for x in X1] + 5 * noise1
        Y2 = [2 * x ** 2 - 0.65 for x in X2] + noise2
    elif scenario == "more_general":
        X2 = np.random.uniform(-0.4, 0.4, n)
        Y1 = [20 * x**6 - 40 * x**4 + 20 * x**2 for x in X1] + 2 * noise1
        Y2 = [20 * x ** 2 for x in X2] + 2 * noise2
    return X1, X2, Y1, Y2


def plot_classif_and_pvalue(scenario, X1, Y1, X2, Y2, p1, p2, beta1, beta2):
    fig, axs = plt.subplots(1, 1)

    # Plot the fit function on the graph.
    if p1 >= 0.01:
        beta1_sci, p1_sci = f'{beta1:.2f}', f'{p1:.2f}'
    else:
        beta1_sci, p1_sci = f'{beta1:.2f}', f'{p1:.2e}'
    if p2 >= 0.01:
        beta2_sci, p2_sci = f'{beta2:.2f}', f'{p2:.2f}'
    else:
        beta2_sci, p2_sci = f'{beta2:.2f}', f'{p2:.2e}'

    markers = {-1: 's', 1: 'v'}
    for cls in np.unique([Y1, Y2]):
        idx = np.where(Y1 == cls)
        axs.scatter(X1[idx, 0], X1[idx, 1], marker=markers[cls], color="tab:blue", s=100)
        idx = np.where(Y2 == cls)
        axs.scatter(X2[idx, 0], X2[idx, 1], marker=markers[cls], color="tab:orange", s=100)

    # Warning: p1 computed on client 1 corresponds to the desir of client 2 to collaborate with client 1.
    # I.e., the beta and pvalue computed on client 1 are relevant for client 2.
    init_legend = [Line2D([], [], marker="o", color="tab:blue",
                          label=fr'Client 1 - $\hat{{\beta}} = {beta1_sci}$, $\hat{{p}} = {p1_sci}$'),
                   Line2D([], [], marker="o", color="tab:orange",
                          label=fr'Client 1 - $\hat{{\beta}} = {beta2_sci}$, $\hat{{p}} = {p2_sci}$')
                   ]
    l2 = axs.legend(handles=init_legend, loc='upper right', fontsize=14)
    axs.add_artist(l2)

    plt.axis('off')  # Hide axes
    plt.savefig(f"../../pictures/{scenario}.pdf", dpi=600, bbox_inches='tight')
    plt.close()



def plot_reg_and_pvalue(scenario, poly_reg1, poly_reg2, X1, Y1, X2, Y2, p1, p2, beta1, beta2):

    fig, axs = plt.subplots(1, 1, figsize=(4, 4))

    # Plot the fit function on the graph.
    if scenario in ["totally_different", "partionned_support"]:
        abs1 = np.linspace(min(min(X1), min(X2)), max(max(X1), max(X2)), 100)
        abs2 = abs1
    else:
        abs1 = np.linspace(min(X1), max(X1), 100)
        abs2 = np.linspace(min(X2), max(X2), 100)
    poly = PolynomialFeatures(DEG)
    plt.plot(abs1, poly_reg1.predict(poly.fit_transform(abs1.reshape(-1, 1))))
    plt.plot(abs2, poly_reg2.predict(poly.fit_transform(abs2.reshape(-1, 1))))

    if p1 >= 0.01:
        beta1_sci, p1_sci = f'{beta1:.2f}', f'{p1:.2f}'
    else:
        beta1_sci, p1_sci = f'{beta1:.2f}', f'{p1:.2e}'
    if p2 >= 0.01:
        beta2_sci, p2_sci = f'{beta2:.2f}', f'{p2:.2f}'
    else:
        beta2_sci, p2_sci = f'{beta2:.2f}', f'{p2:.2e}'
    # Warning: p1 computed on client 1 corresponds to the desir of client 2 to collaborate with client 1.
    # I.e., the beta and pvalue computed on client 1 are relevant for client 2.
    axs.scatter(X1, Y1, label=fr'Client 1 - $\hat{{\beta}}^{{1,2}} = {beta1_sci}$, $\hat{{p}}^{{1,2}} = {p1_sci}$')
    axs.scatter(X2, Y2, label=fr'Client 2 - $\hat{{\beta}}^{{2,1}} = {beta2_sci}$, $\hat{{p}}^{{2,1}} = {p2_sci}$')
    # plt.scatter(X2, Y2, label=r'Client 2 - $\hat{\\beta} = {0}, \hat{p} = {1}$'.format(beta2, p2))
    if scenario in ["same_with_variations", "partionned_support"]:
        axs.legend(fontsize=12, loc="upper left")
    else:
        axs.legend(fontsize=12, loc="best")
    plt.axis('off')  # Hide axes
    plt.savefig(f"../../pictures/{scenario}.pdf", dpi=600, bbox_inches='tight')
    plt.close()



def polynomial_regression(X, Y, beta0, split_percent: int = 0.5):

    train_set_length = int(len(Y) * split_percent)

    poly = PolynomialFeatures(DEG)
    poly_features = poly.fit_transform(X.reshape(-1, 1))

    # fit polynomial regression model
    poly_reg = LinearRegression()
    poly_reg.fit(poly_features[:train_set_length], Y[:train_set_length])

    atomic_errors = compute_atomic_errors(poly_reg, poly_features[train_set_length:], Y[train_set_length:])
    q0 = np.quantile(atomic_errors, beta0, method="higher")
    return poly_reg, q0, atomic_errors


def logistic_regression(X, Y, beta0, split_percent: int = 0.5):
    train_set_length = int(len(Y) * split_percent)

    log_reg = LogisticRegression(penalty=None)

    log_reg.fit(X[:train_set_length], Y[:train_set_length])

    atomic_errors = compute_atomic_errors(log_reg, X[train_set_length:], Y[train_set_length:],
                                          logistic=True)
    q0 = np.quantile(atomic_errors, beta0, method="higher")
    return None, q0, atomic_errors


def compute_pvalue(remote_poly_reg, q0, X, Y, beta0, split_percent: int = 0.5, log: bool = False):

    train_set_length = int(len(Y) * split_percent)
    poly = PolynomialFeatures(DEG)
    poly_features = poly.fit_transform(X.reshape(-1, 1))

    test = ProportionTest(beta0=beta0)
    test.evaluate_test(q0, remote_poly_reg, poly_features[train_set_length:], Y[train_set_length:])

    if log:
        test.print()

    return test.beta_estimator, test.pvalue, test.atomic_errors


def quantile_test_on_two_datasets(scenario, X1, Y1, X2, Y2, beta0: int, split_percent: int):
    print("Client 1 share its model with client 2.")
    poly_reg1, q0_1, atomic_errors1 = polynomial_regression(X1, Y1, beta0, split_percent)
    beta_estimator1, pvalue1, atomic_errors2 = compute_pvalue(poly_reg1, q0_1, X2, Y2, beta0, split_percent)

    print("Client 2 share its model with client 1.")
    poly_reg2, q0_2, atomic_errors1 = polynomial_regression(X2, Y2, beta0, split_percent)
    beta_estimator2, pvalue2, atomic_errors2 = compute_pvalue(poly_reg2, q0_2, X1, Y1, beta0, split_percent)

    if scenario == "classification":
        plot_classif_and_pvalue(scenario, X1, Y1, X2, Y2, pvalue1, pvalue2, beta_estimator1, beta_estimator2)
    else:
        plot_reg_and_pvalue(scenario, poly_reg1, poly_reg2, X1, Y1, X2, Y2, pvalue1, pvalue2,
                            beta_estimator1, beta_estimator2)


def quantile_test_on_two_models(scenario, X1, Y1, X2, Y2, beta0: int, split_percent: int, plot=False):

    # We train the model on client 1.
    poly_reg1, q0_1, local_atomic_errors1 = polynomial_regression(X1, Y1, beta0, split_percent)
    # We train the model on client 2.
    poly_reg2, q0_2, local_atomic_errors2 = polynomial_regression(X2, Y2, beta0, split_percent)

    # We share model 2 with client 1.
    beta_estimator1, pvalue1, atomic_errors1 = compute_pvalue(poly_reg2, q0_1, X1, Y1, beta0, split_percent,
                                                              plot)

    # We share model 1 with client 2.
    beta_estimator2, pvalue2, atomic_errors2 = compute_pvalue(poly_reg1, q0_2, X2, Y2, beta0, split_percent,
                                                              plot)

    if plot:
        # q0_1 = quantile de local_atomic_errors1
        plot_arrow_with_atomic_errors(local_atomic_errors1, atomic_errors1, beta0,
                                      pvalue=pvalue1, main_client=1, name=f"{scenario}_quantiles1")
        plot_arrow_with_atomic_errors(local_atomic_errors2, atomic_errors2, beta0,
                                      pvalue=pvalue2, main_client=2, name=f"{scenario}_quantiles2")

        if scenario == "classification":
            plot_classif_and_pvalue(scenario, X1, Y1, X2, Y2, pvalue1, pvalue2, beta_estimator1, beta_estimator2)
        else:
            plot_reg_and_pvalue(scenario, poly_reg1, poly_reg2, X1, Y1, X2, Y2, pvalue1, pvalue2,
                                beta_estimator1, beta_estimator2)
    return pvalue1, pvalue2


# def plot_atomic_errors(atomic_errors1, atomic_errors2, q0):
#     # Showing the atomic errors on a line.
#     ones = np.ones(np.shape(atomic_errors1))  # Make all y values the same
#     plt.plot(np.log10(atomic_errors1), ones, marker="x", lw=2, ms=10,
#              label="Client 1")  # Plot a line at each location specified in a
#     plt.plot(np.log10(atomic_errors2), 2 * ones, marker="x", lw=2, ms=10,
#              label="Client 2")  # Plot a line at each location specified in a
#     plt.plot([np.log10(q0)], [1], marker="o", ms=10, color="black", label="q0")
#     plt.plot([np.log10(q0)], [2], marker="o", ms=10, color="black")
#     plt.legend()
#     plt.show()

def algo(scenario: str, beta0: int, split_percent: int, nb_run: int = 100, plot: bool = False):

    print(f"=== {scenario} ===")
    pvalue1, pvalue2 = [], []
    for k in range(nb_run):
        X1, X2, Y1, Y2 = f(scenario)
        p1, p2 = quantile_test_on_two_models(scenario, X1, Y1, X2, Y2, beta0, split_percent, plot=plot)
        pvalue1.append(p1)
        pvalue2.append(p2)

    # Replace pvalue equal to 0, by a small value in order to compute the log.
    if plot:
        pvalue1 = [x if x != 0 else 10**-196 for x in pvalue1]
        pvalue2 = [x if x != 0 else 10**-196 for x in pvalue2]
        plt.hist(np.log10(pvalue1), density=True, stacked=False, histtype='bar',
                 color="tab:blue", alpha=0.5, align="right", label="Client 1")
        plt.hist(np.log10(pvalue2), density=True, stacked=False, histtype='bar',
                 color="tab:orange", alpha=0.5, align="right", label="Client 2")
        # plt.hist(, color="tab:orange", alpha=0.5)
        plt.plot([np.log10(0.05), np.log10(0.05)], [0,0.5], color="tab:red", lw=3)
        plt.xlabel(r"$\mathrm{log}_{10}(\hat{p})$", fontsize=14)
        plt.ylabel("Density")
        plt.legend(fontsize=14)
        plt.savefig(f"../../pictures/{scenario}_pvalue.pdf", dpi=600, bbox_inches='tight')
        plt.close()

    return pvalue1, pvalue2

    ##### Comparing two datasets with one model. #####
    # quantile_test_on_two_datasets(scenario, X1, Y1, X2, Y2, beta0, split_percent)

    # ##### Comparing two models with one dataset. #####

def plot_violin_pvalue(pvalues, all_beta0, scenario):
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    pclient1 = [np.log10(p[0]) for p in pvalues[scenario]]
    pclient2 = [np.log10(p[1]) for p in pvalues[scenario]]
    ax1.violinplot(pclient1, positions=np.arange(1, len(all_beta0) + 1) - 0.15, widths=0.25)
    ax1.violinplot(pclient2, positions=np.arange(1, len(all_beta0) + 1) + 0.15, widths=0.25)
    ax1.plot([0.5, len(all_beta0) + 0.5], [np.log10(0.05), np.log10(0.05)], lw=2, color="tab:red")
    ax1.set_xticks(np.arange(1, len(all_beta0) + 1), labels=all_beta0)
    ax1.set_xlabel(r'Quantile level $\hat{\beta}$')
    ax1.set_ylabel(r'Distribution of $p$-values')

    fake_handles = [mpatches.Patch(color='tab:blue', alpha=0.5),
                    mpatches.Patch(color='tab:orange', alpha=0.5),
                    Line2D([0], [0], color="tab:red", lw=2)]
    ax1.legend(fake_handles, ["Client 1", "Client 2", r"$p=.05$"], loc='lower left', fontsize=14)
    plt.savefig(f"../../pictures/{scenario}_pvalue_violin.pdf", dpi=600, bbox_inches='tight')


if __name__ == "__main__":
    BETA_0 = 0.8
    split_percent = 0.8

    np.random.seed(2024)

    all_beta0 = [0.5, 0.75, 0.8, 0.9, 0.95]
    scenarios = ["same", "different", "Y_shift", "same_partionned_support", "X_shift",
                 "same_partionned_support_different_Y", "partionned_support",
                 "totally_different", "various_noise", "more_general", "same_with_variations"]

    # scenarios = ["same_with_variations"]
    pvalues = {l: [] for l in scenarios}

    for scenario in scenarios:
        algo(scenario, BETA_0, split_percent, nb_run=1, plot=True)
        for beta0 in all_beta0:
            pvalues[scenario].append(algo(scenario, beta0, split_percent))

        plot_violin_pvalue(pvalues, all_beta0, scenario)