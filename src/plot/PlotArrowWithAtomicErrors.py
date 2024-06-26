import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def plot_arrow_with_atomic_errors(local_loss, loss_with_remote_model, beta, pvalue, main_client, name):

    # Create figure and axes
    fig, ax = plt.subplots()

    pvalue = f'{pvalue:.2f}' if pvalue >= 0.01 else 0.0

    # m = min(min(x_points1), min(x_points2))
    # x_points1 = np.array(x_points1) - m
    reference_quantile = np.quantile(local_loss, beta, method="higher")
    # x_points2 = np.array(x_points2) - m
    quantile_on_remote_loss = np.quantile(loss_with_remote_model, beta, method="higher")

    # if main_client == 1:
    estimated_beta = np.sum([e <= reference_quantile for e in loss_with_remote_model]) / len(loss_with_remote_model)
    # else:
    #     estimated_beta = np.sum([e <= q2 for e in x_points1]) / len(x_points1)

    line_length = max(max(local_loss), max(loss_with_remote_model))
    y_sep = line_length / 4

    # Draw the first line
    ax.annotate(
        '', xy=(line_length * 1.25, 0),
        xytext=(-line_length * 0.1, 0),
        arrowprops=dict(facecolor='tab:grey', edgecolor='tab:grey', arrowstyle='-|>',
                        mutation_scale=20)
    )

    # Draw the second line
    ax.annotate(
        '', xy=(line_length * 1.25, y_sep),
        xytext=(-line_length * 0.1, y_sep),
        arrowprops=dict(facecolor='tab:grey', edgecolor='tab:grey', arrowstyle='-|>',
                        mutation_scale=20)
    )

    # Plotting the atomic errors on the two lines.
    # Client 1 is at the lower line.
    # Client 2 at the upper line.

    lower = np.zeros(len(local_loss))
    upper = np.full(len(loss_with_remote_model), y_sep)
    if main_client == 1:
        ax.scatter(local_loss, lower, color='tab:blue', label=r"$\hat{w}_1$", s=50)
        ax.scatter(loss_with_remote_model, upper, color='tab:orange', label=r"$\hat{w}_2$", s=50)
    else:
        ax.scatter(loss_with_remote_model, lower, color='tab:blue', label=r"$\hat{w}_1$", s=50)
        ax.scatter(local_loss, upper, color='tab:orange', label=r"$\hat{w}_2$", s=50)


    if main_client == 1:
        ax.plot([reference_quantile, reference_quantile], [-y_sep*0.25, y_sep*0.25], color='tab:blue', lw=2)
        ax.plot([reference_quantile, reference_quantile], [y_sep*0.35, y_sep*1.5], color='tab:red', lw=2, linestyle="dotted")
        ax.plot([quantile_on_remote_loss, quantile_on_remote_loss], [y_sep*0.75, y_sep*1.25],
                color='tab:orange', lw=2)

        ax.text(reference_quantile, - y_sep * .5, r'$q_{' + str(beta) + '}(\hat{{w}}_1, \mathcal{{D}}_1)$', fontsize=14, ha='left',
                va='center', color="tab:blue")
        ax.text(quantile_on_remote_loss, y_sep * .5, r'$q_{' + str(beta) + '}(\hat{{w}}_2, \mathcal{{D}}_1)$',
                fontsize=14, ha='center',
                va='center', color="tab:orange")
        ax.text(reference_quantile * 1.05, y_sep * 1.5, r'$q_{\hat{\beta}}(\hat{w}_2, \mathcal{D}_1)$', fontsize=14,
                ha='left', va='center', color="tab:red")

        # Plot the estimated beta and the pvalue.
        ax.text(line_length * 1.25, y_sep*1.25, rf'$\hat{{\beta}} = {estimated_beta}$', fontsize=14, ha='left', va='center',
                color="tab:orange")
        ax.text(line_length * 1.25, y_sep*0.75, rf'$\hat{{p}} = {pvalue}$', fontsize=14, ha='left', va='center',
                color="tab:orange")

    else:
        ax.plot([quantile_on_remote_loss, quantile_on_remote_loss], [-y_sep * 0.25, y_sep * 0.25],
                color='tab:blue', lw=2)
        ax.plot([reference_quantile, reference_quantile], [y_sep * 0.65, -y_sep * 0.5],
                color='tab:red', lw=2, linestyle="dotted")
        ax.plot([reference_quantile, reference_quantile], [y_sep * 0.75, y_sep * 1.25],
                color='tab:orange', lw=2)

        ax.text(quantile_on_remote_loss, y_sep * .5, r'$q_{' + str(beta) + '}(\hat{{w}}_1, \mathcal{{D}}_2)$',
                fontsize=14, ha='left', va='center', color="tab:blue")
        ax.text(reference_quantile, y_sep * 1.5, r'$q_{' + str(beta) + '}(\hat{{w}}_2, \mathcal{{D}}_2)$',
                fontsize=14, ha='center', va='center', color="tab:orange")
        ax.text(reference_quantile * 1.05, -y_sep * .5, r'$q_{\hat{\beta}}(\hat{w}_1, \mathcal{D}_2)$', fontsize=14, ha='left',
                va='center', color="tab:red")

        # Plot the estimated beta and the pvalue.
        ax.text(line_length * 1.25, -y_sep * 0.25, rf'$\hat{{\beta}} = {estimated_beta}$', fontsize=14, ha='left', va='center',
                color="tab:blue")
        ax.text(line_length*1.25, y_sep * 0.25, rf'$\hat{{p}} = {pvalue}$', fontsize=14, ha='left', va='center',
                color="tab:blue")

    ax.text(-line_length*0.15, 0, '$\hat{w}_1$', fontsize=14, ha='center', va='center', color="tab:blue")
    ax.text(-line_length*0.15, y_sep, '$\hat{w}_2$', fontsize=14, ha='center', va='center', color="tab:orange")

    # plt.show()

    # Set limits and show the plot
    ax.set_xlim(-line_length*0.3, line_length * 1.5)
    ax.set_ylim(-y_sep*0.75, y_sep * 1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')  # Hide axes

    plt.savefig(f"../../pictures/{name}.pdf", dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    beta = 0.8

    x_points1 = [1, 2, 2.25, 3, 4, 5.1, 6.9, 7.9, 9.2, 9.5]
    x_points2 = [1.5, 1.75, 2.75, 4.25, 4.5, 5, 6.25, 7, 8, 9.8]
    plot_arrow_with_atomic_errors(x_points1, x_points2, beta, pvalue=0.78, main_client=1, name="iid")

    x_points1 = [1, 1.2, 1.35, 2, 2.3, 2.7, 2.9, 3.5, 4.7, 5]
    x_points2 = [4.5, 5.1, 5.5, 6, 6.45, 6.9, 7.1, 7.65, 8, 9.8]
    plot_arrow_with_atomic_errors(x_points1, x_points2, beta, pvalue=10**-10, main_client=1, name="non_iid")