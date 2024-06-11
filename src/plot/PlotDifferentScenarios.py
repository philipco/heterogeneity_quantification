import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def MSE(y, ypred):
    return np.abs(y - ypred)


def f(scenario: str, n=100):

    noise1 = np.random.normal(0, 0.05, n)
    noise2 = np.random.normal(0, 0.05, n)
    X1 = np.random.uniform(-1, 1, n)
    X2 = np.random.uniform(-1, 1, n)

    if scenario == "partionned_support":
        X2 = np.random.uniform(0.75, 1, n)
        Y1 = [x**2 for x in X1] + noise1
        Y2 = [x**2 for x in X2] + noise2
    elif scenario == "shift":
        Y1 = [x ** 2 for x in X1] + noise1
        Y2 = [x ** 2 + 1 for x in X2] + noise2
    elif scenario == "same":
        Y1 = [x ** 2 for x in X1] + noise1
        Y2 = [x ** 2 + np.sin(10 * x) / 10 for x in X2] + noise2
    elif scenario == "different":
        Y1 = [x ** 2 for x in X1] + noise1
        Y2 = [(x + 1) ** 2 for x in X2] + noise2
    elif scenario == "various_noise":
        X2 = np.random.uniform(0.65, 1, n)
        Y1 = [x ** 2  for x in X1] + 5 * noise1
        Y2 = [2 * x ** 2 - 0.65 for x in X2] + noise2
    return X1, X2, Y1, Y2


def compute_atomic_errors(reg, features, Y, train_set_length):
    prediction = reg.predict(features[train_set_length:])
    atomic_errors = [MSE(y, ypred) for (y, ypred) in zip(Y[train_set_length:], prediction)]
    return atomic_errors


def polynomial_regression(X, Y, beta0, split_percent: int = 0.5):

    train_set_length = int(len(Y) * split_percent)

    poly = PolynomialFeatures(2)
    poly_features = poly.fit_transform(X.reshape(-1, 1))

    # fit polynomial regression model
    poly_reg = LinearRegression()
    poly_reg.fit(poly_features[:train_set_length], Y[:train_set_length])

    atomic_errors = compute_atomic_errors(poly_reg, poly_features, Y, train_set_length)
    q0 = np.quantile(atomic_errors, beta0, method="inverted_cdf")

    return poly_reg, q0, atomic_errors


def compute_pvalue(remote_poly_reg, q0, X, Y, beta0, split_percent: int = 0.5):

    train_set_length = int(len(Y) * split_percent)
    test_set_length =  len(Y) - train_set_length
    poly = PolynomialFeatures(2)
    poly_features = poly.fit_transform(X.reshape(-1, 1))

    atomic_errors = compute_atomic_errors(remote_poly_reg, poly_features, Y, train_set_length)

    beta_estimator = np.sum([e <= q0 for e in atomic_errors]) / test_set_length
    print(f"\tEstimation de beta: {beta_estimator}")

    pvalue = norm.cdf(np.sqrt(test_set_length) * (beta_estimator - beta0) / np.sqrt(beta0 * (1 - beta0)))
    print(f"\tP-value: {pvalue}")

    beta_critique = norm.ppf(0.05) * np.sqrt(beta0 * (1 - beta0))/ np.sqrt(test_set_length) + beta0
    print(f"\tBeta critique: {beta_critique}")

    if pvalue < 0.05:
        print("=> H0 is rejected.")
    else:
        print("=> H0 can not be rejected.")

    return pvalue, atomic_errors


def quantile_test_on_two_datasets(scenario, X1, Y1, X2, Y2, beta0: int, split_percent: int):
    print("Client 1 share its model with client 2.")
    poly_reg1, q0_1, atomic_errors1 = polynomial_regression(X1, Y1, beta0, split_percent)
    pvalue, atomic_errors2 = compute_pvalue(poly_reg1, q0_1, X2, Y2, beta0, split_percent)
    plot_atomic_errors(atomic_errors1, atomic_errors2, q0_1)

    print("Client 2 share its model with client 1.")
    poly_reg2, q0_2, atomic_errors1 = polynomial_regression(X2, Y2, beta0, split_percent)
    pvalue, atomic_errors2 = compute_pvalue(poly_reg2, q0_2, X1, Y1, beta0, split_percent)
    plot_atomic_errors(atomic_errors1, atomic_errors2, q0_2)

    # Plot the fit function on the graph.
    abs = np.linspace(-1, 1, 100)
    poly = PolynomialFeatures(2)
    plt.plot(abs, poly_reg1.predict(poly.fit_transform(abs.reshape(-1, 1))))
    plt.plot(abs, poly_reg2.predict(poly.fit_transform(abs.reshape(-1, 1))))

    plt.scatter(X1, Y1, label="Client 1")
    plt.scatter(X2, Y2, label="Client 2")
    plt.legend()
    plt.savefig(f"../../pictures/{scenario}.pdf", dpi=600, bbox_inches='tight')
    plt.close()


def quantile_test_on_two_models(scenario, X1, Y1, X2, Y2, beta0: int, split_percent: int):
    print("Client 1 and 2 train their models.")
    poly_reg1, q0_1, local_atomic_errors1 = polynomial_regression(X1, Y1, beta0, split_percent)
    poly_reg2, q0_2, local_atomic_errors2 = polynomial_regression(X2, Y2, beta0, split_percent)

    print("Client 1 receives the trained model from client 2 and evaluates it on its dataset.")
    pvalue, atomic_errors1 = compute_pvalue(poly_reg2, q0_1, X1, Y1, beta0, split_percent)
    plot_atomic_errors(local_atomic_errors1, atomic_errors1, q0_1)

    print("Client 2 share its trained model with client 1 which will evaluate it on its dataset.")
    pvalue, atomic_errors2 = compute_pvalue(poly_reg1, q0_2, X2, Y2, beta0, split_percent)
    plot_atomic_errors(local_atomic_errors2, atomic_errors2, q0_2)


def plot_atomic_errors(atomic_errors1, atomic_errors2, q0):
    # Showing the atomic errors on a line.
    ones = np.ones(np.shape(atomic_errors1))  # Make all y values the same
    plt.plot(np.log10(atomic_errors1), ones, marker="x", lw=2, ms=10,
             label="Client 1")  # Plot a line at each location specified in a
    plt.plot(np.log10(atomic_errors2), 2 * ones, marker="x", lw=2, ms=10,
             label="Client 2")  # Plot a line at each location specified in a
    plt.plot([np.log10(q0)], [1], marker="o", ms=10, color="black", label="q0")
    plt.plot([np.log10(q0)], [2], marker="o", ms=10, color="black")
    plt.legend()
    plt.show()

def algo(scenario: str, beta0: int, split_percent: int):

    print(f"=== {scenario} ===")
    X1, X2, Y1, Y2 = f(scenario)

    ##### Comparing two datasets with one model. #####
    quantile_test_on_two_datasets(scenario, X1, Y1, X2, Y2, beta0, split_percent)

    # ##### Comparing two models with one dataset. #####
    quantile_test_on_two_models(scenario, X1, Y1, X2, Y2, beta0, split_percent)

if __name__ == "__main__":
    beta0 = 0.95
    split_percent = 0.8


    algo("same", beta0, split_percent)
    algo("various_noise", beta0, split_percent)
    algo("partionned_support", beta0, split_percent)
    algo("shift", beta0, split_percent)
    algo("different", beta0, split_percent)