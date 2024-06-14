import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib

from src.plot.PlotArrowWithAtomicErrors import plot_arrow_with_atomic_errors

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def MSE(y, ypred):
    return (y - ypred)**2 / 2


def sigmoid_loss(y, y_pred):
    return np.log(1 + np.exp(-y * y_pred))


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
        Y2 = [x ** 2 for x in X2] + noise2
    elif scenario == "same_partionned_support":
        X1 = np.random.uniform(-1, 0., n)
        X2 = np.random.uniform(0., 1, n)
        Y1 = [x ** 2 for x in X1] + noise1
        Y2 = [x ** 2 for x in X2] + noise2
    elif scenario == "same_partionned_support_different_Y":
        X1 = np.random.uniform(0, 1, n)
        X2 = np.random.uniform(1., 2, n)
        Y1 = [x ** 2 for x in X1] + noise1
        Y2 = [x ** 2 for x in X2] + noise2
    elif scenario == "same_with_variations":
        Y1 = [x ** 2 for x in X1] + noise1
        Y2 = [x ** 2 + np.sin(10 * x) / 10 for x in X2] + noise2
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
                          label=fr'Client 1 - $\hat{{\beta}} = {beta2_sci}$, $\hat{{p}} = {p2_sci}$'),
                   Line2D([], [], marker="o", color="tab:orange",
                          label=fr'Client 1 - $\hat{{\beta}} = {beta1_sci}$, $\hat{{p}} = {p1_sci}$')
                   ]

    # plt.scatter(X1, Y1, label=fr'Client 1 - $\hat{{\beta}} = {beta2_sci}$, $\hat{{p}} = {p2_sci}$')
    # plt.scatter(X2, Y2, label=fr'Client 2 - $\hat{{\beta}} = {beta1_sci}$, $\hat{{p}} = {p1_sci}$')
    # plt.scatter(X2, Y2, label=r'Client 2 - $\hat{\\beta} = {0}, \hat{p} = {1}$'.format(beta2, p2))
    # plt.legend(fontsize=14)

    l2 = axs.legend(handles=init_legend, loc='upper right', fontsize=14)
    axs.add_artist(l2)

    plt.axis('off')  # Hide axes
    plt.savefig(f"../../pictures/{scenario}.pdf", dpi=600, bbox_inches='tight')
    plt.close()



def plot_reg_and_pvalue(scenario, poly_reg1, poly_reg2, X1, Y1, X2, Y2, p1, p2, beta1, beta2):

    fig, axs = plt.subplots(1, 1, figsize=(4, 4))

    # Plot the fit function on the graph.
    abs = np.linspace(min(min(X1), min(X2)), max(max(X1), max(X2)), 100)
    poly = PolynomialFeatures(2)
    plt.plot(abs, poly_reg1.predict(poly.fit_transform(abs.reshape(-1, 1))))
    plt.plot(abs, poly_reg2.predict(poly.fit_transform(abs.reshape(-1, 1))))

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
    axs.scatter(X1, Y1, label=fr'Client 1 - $\hat{{\beta}} = {beta2_sci}$, $\hat{{p}} = {p2_sci}$')
    axs.scatter(X2, Y2, label=fr'Client 2 - $\hat{{\beta}} = {beta1_sci}$, $\hat{{p}} = {p1_sci}$')
    # plt.scatter(X2, Y2, label=r'Client 2 - $\hat{\\beta} = {0}, \hat{p} = {1}$'.format(beta2, p2))
    axs.legend(fontsize=14)
    plt.axis('off')  # Hide axes
    plt.savefig(f"../../pictures/{scenario}.pdf", dpi=600, bbox_inches='tight')
    plt.close()


def compute_atomic_errors(reg, features, Y, logistic=False):
    prediction = reg.predict(features)
    if not logistic:
        atomic_errors = [MSE(y, ypred) for (y, ypred) in zip(Y, prediction)]
    else:
        atomic_errors = [sigmoid_loss(y, ypred) for (y, ypred) in zip(Y, prediction)]
    return atomic_errors


def polynomial_regression(X, Y, beta0, split_percent: int = 0.5):

    train_set_length = int(len(Y) * split_percent)

    poly = PolynomialFeatures(2)
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
    test_set_length = len(Y) - train_set_length
    poly = PolynomialFeatures(2)
    poly_features = poly.fit_transform(X.reshape(-1, 1))

    atomic_errors = compute_atomic_errors(remote_poly_reg, poly_features[train_set_length:], Y[train_set_length:])

    beta_estimator = np.sum([e <= q0 for e in atomic_errors]) / len(atomic_errors)


    pvalue = norm.cdf(np.sqrt(test_set_length) * (beta_estimator - beta0) / np.sqrt(beta0 * (1 - beta0)))


    beta_critique = norm.ppf(0.05) * np.sqrt(beta0 * (1 - beta0))/ np.sqrt(test_set_length) + beta0


    if log:
        if pvalue < 0.05:
            print("=> H0 is rejected.")
        else:
            print("=> H0 can not be rejected.")

        print(f"\tEstimation de beta: {beta_estimator}")
        print(f"\tP-value: {pvalue}")
        print(f"\tBeta critique: {beta_critique}")

    return beta_estimator, pvalue, atomic_errors


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
    # print("Client 1 and 2 train their models.")
    poly_reg1, q0_1, local_atomic_errors1 = polynomial_regression(X1, Y1, beta0, split_percent)
    poly_reg2, q0_2, local_atomic_errors2 = polynomial_regression(X2, Y2, beta0, split_percent)


    # print("Client 1 receives the trained model from client 2 and evaluates it on its dataset.")
    beta_estimator1, pvalue1, atomic_errors1 = compute_pvalue(poly_reg2, q0_1, X1, Y1, beta0, split_percent,
                                                              plot)



    # print("Client 2 share its trained model with client 1 which will evaluate it on its dataset.")
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

def algo(scenario: str, beta0: int, split_percent: int, nb_run: int = 200):

    print(f"=== {scenario} ===")
    pvalue1, pvalue2 = [], []
    for k in range(nb_run):
        X1, X2, Y1, Y2 = f(scenario)
        p1, p2 = quantile_test_on_two_models(scenario, X1, Y1, X2, Y2, beta0, split_percent, plot=(k==0))
        pvalue1.append(p1)
        pvalue2.append(p2)

    plt.hist(np.log10(pvalue2), density=True, stacked=False, histtype='bar',
             color="tab:blue", alpha=0.5, align="right", label="Client 1")
    plt.hist(np.log10(pvalue1), density=True, stacked=False, histtype='bar',
             color="tab:orange", alpha=0.5, align="right", label="Client 2")
    # plt.hist(, color="tab:orange", alpha=0.5)
    plt.plot([np.log10(0.05), np.log10(0.05)], [0,0.5], color="tab:red", lw=3)
    plt.xlabel(r"$\mathrm{log}_{10}(\hat{p})$", fontsize=14)
    plt.ylabel("Density")
    plt.legend(fontsize=14)
    plt.savefig(f"../../pictures/{scenario}_pvalue.pdf", dpi=600, bbox_inches='tight')
    plt.close()


    ##### Comparing two datasets with one model. #####
    # quantile_test_on_two_datasets(scenario, X1, Y1, X2, Y2, beta0, split_percent)

    # ##### Comparing two models with one dataset. #####


if __name__ == "__main__":
    beta0 = 0.8
    split_percent = 0.8

    # np.random.seed(100)

    algo("same", beta0, split_percent)
    algo("different", beta0, split_percent)
    algo("classification", beta0, split_percent)
    algo("Y_shift", beta0, split_percent)
    algo("same_partionned_support", beta0, split_percent)
    algo("X_shift", beta0, split_percent)
    algo("same_partionned_support_different_Y", beta0, split_percent)
    algo("totally_different", beta0, split_percent)

    algo("partionned_support", beta0, split_percent)
    algo("same_with_variations", beta0, split_percent)
    algo("various_noise", beta0, split_percent)