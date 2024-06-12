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

# Parameters
line_length = 10
y_sep = 1
num_points = 5

def plot_arrow_with_atomic_errors(x_points1, q1, x_points2, q2, q_critique, name):

    # Create figure and axes
    fig, ax = plt.subplots()

    # Draw the first line
    ax.annotate(
        '', xy=(line_length+1, 0), xytext=(0, 0),
        arrowprops=dict(facecolor='tab:grey', edgecolor='tab:grey', arrowstyle='-|>',
                        mutation_scale=20)
    )

    # Draw the second line
    ax.annotate(
        '', xy=(line_length+1, y_sep), xytext=(0, y_sep),
        arrowprops=dict(facecolor='tab:grey', edgecolor='tab:grey', arrowstyle='-|>',
                        mutation_scale=20)
    )
     #Plotting the atomic errors on the two lines.
    y_points1 = np.zeros(len(x_points1))
    ax.scatter(x_points1, y_points1, color='tab:blue', label=r"$\hat{w}_1$", alpha=1, s=50)

    y_points2 = np.full(len(x_points2), y_sep)
    ax.scatter(x_points2, y_points2, color='tab:orange', label=r"$\hat{w}_2$", s=50)

    ax.plot([q1, q1], [-0.25, 0.25], color='tab:blue', lw=2)
    ax.plot([q1, q1], [0.35, 1.5], color='tab:red', lw=2, linestyle="dotted")
    ax.plot([q2, q2], [1-0.25, 1+0.25], color='tab:orange', lw=2)

    # Plot q_critical.
    # ax.plot([q_critique, q_critique], [1-0.25, 1+0.25], color='tab:red', lw=2)
    # ax.plot([q_critique, q_critique+0.25], [1 - 0.25, 1 - 0.25], color='tab:red', lw=2)
    # ax.plot([q_critique, q_critique+0.25], [1 + 0.25, 1 + 0.25], color='tab:red', lw=2)

    ax.text(q1, -0.5, r'$q_{0.8}(\hat{w}_1, \mathcal{D}_1)$', fontsize=14, ha='center', va='center',
            color="tab:blue")
    ax.text(q2, 0.5, r'$q_{0.8}(\hat{w}_2, \mathcal{D}_1)$', fontsize=14, ha='center', va='center',
            color="tab:orange")
    ax.text(q1+0.1, 1.5, r'$q_{\hat{\beta}}(\hat{w}_2, \mathcal{D}_1)$', fontsize=14, ha='left', va='center',
            color="tab:red")
    # ax.text(q_critique-0.25, 1.5, r'$q_\alpha$', fontsize=14, ha='center', va='center',
    #         color="tab:red")

    ax.text(-.5, 0, '$\hat{w}_1$', fontsize=14, ha='center', va='center', color="tab:blue")
    ax.text(-.5, 1, '$\hat{w}_2$', fontsize=14, ha='center', va='center', color="tab:orange")

    # Set limits and show the plot
    ax.set_xlim(-1, line_length + 1)
    ax.set_ylim(-1, y_sep + 1)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')  # Hide axes
    plt.savefig(f"../../../pictures/{name}.pdf", dpi=600, bbox_inches='tight')


if __name__ == "__main__":

    x_points1 = [1, 2, 2.25, 3, 4, 5.1, 6.9, 7.9, 9.2, 9.5]
    x_points2 = [1.5, 1.75, 2.75, 4.25, 4.5, 5, 6.25, 7, 8, 9.8]
    q1, q2 = 9.2, 8
    plot_arrow_with_atomic_errors(x_points1, q1, x_points2, q2, q1-1, name="iid")

    x_points1 = [1, 1.2, 1.35, 2, 2.3, 2.7, 2.9, 3.5, 4.7, 5]
    x_points2 = [4.5, 5.1, 5.5, 6, 6.45, 6.9, 7.1, 7.65, 8, 9.8]
    q1, q2 = 4.7, 8
    plot_arrow_with_atomic_errors(x_points1, q1, x_points2, q2, q1-0.5, name="non_iid")