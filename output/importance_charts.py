######################################################################################
######################################################################################
#
# Description: creates chart for variable importance figure
#
######################################################################################
######################################################################################

import pandas as pd

# replace the paths with the relative paths of your variable importance csv files
data_primal = pd.read_csv("results/Colab/rep/rep_importance/importance_primal_v1_colab_env5_rep.csv")
data_dual = pd.read_csv("results/Colab/rep/rep_importance/importance_dual_v1_colab_env5_rep.csv")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

#######################################################
# Labels of inputs
#######################################################

labels_primal = {
    1: r"$S(X_{\mathrm{primal}})_{t_n}^{1}$: $[S]_{t_n}$",
    2: r"$S(X_{\mathrm{primal}})_{t_n}^{2}$: $v_{t_n}-v_0$",
    3: r"$S(X_{\mathrm{primal}})_{t_n}^{12}$: $\int [S]\,dv$",
    4: r"$S(X_{\mathrm{primal}})_{t_n}^{112}$: $\int [S]\,d[S]\,dv$",
    5: r"$S(X_{\mathrm{primal}})_{t_n}^{122}$: $\int [S]\,dv\,dv$",
    6: r"State log price $S_{t_n}$",
    7: r"State average $A_{t_n}$"
}

labels_dual = {
    1: r"$S(X_{\mathrm{dual}})_{s_j}^{1}$: $s_j$",
    2: r"$S(X_{\mathrm{dual}})_{s_j}^{2}$: $v_{s_j}-v_0$",
    3: r"$S(X_{\mathrm{dual}})_{s_j}^{12}$: $\int t\,dv$",
    4: r"$S(X_{\mathrm{dual}})_{s_j}^{112}$: $\int t\,dt\,dv$",
    5: r"$S(X_{\mathrm{dual}})_{s_j}^{122}$: $\int t\,dv\,dv$",
    6: r"State log price $S_{s_j}$",
    7: r"State average $A_{s_j}$"
}


#######################################################
# Data preparation
#######################################################

# remove constant and standard result
primal_plot = data_primal[data_primal["Dimension"] > 0].copy()
dual_plot = data_dual[data_dual["Dimension"] > 0].copy()

# rename rows
primal_plot["Label"] = primal_plot["Dimension"].map(labels_primal)
dual_plot["Label"] = dual_plot["Dimension"].map(labels_dual)

# Keep natural order:
# signature components first, additional variables second
primal_plot = primal_plot.sort_values("Dimension")
dual_plot = dual_plot.sort_values("Dimension")

#######################################################
# Create the plot
#######################################################

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
})

fig, axes = plt.subplots(
    2, 1,
    figsize=(9, 7.5),
    sharex=True
)

def plot_importance(ax, df, title):

    # Dimensions 1-5 = signature components
    # Dimensions 6-7 = additional variables
    #
    # Extra gap inserted before dimensions 6-7
    y_positions = [0, 1, 2, 3, 4, 5.4, 6.4]

    bars = ax.barh(
        y_positions,
        df["Error pct"],
        height=0.7
    )

    # Variable labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["Label"])

    # Put dimension 1 at the top
    ax.invert_yaxis()

    # Percentage labels at end of bars
    for bar, value in zip(bars, df["Error pct"]):
        ax.text(
            value + 0.006,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            ha="left",
            fontsize=9
        )

    # Panel title
    ax.set_title(
        title,
        loc="center",
        fontweight="bold",
        pad=10
    )

    # Percentage axis
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))

    # Grid
    ax.xaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    # Clean appearance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="y", length=0)


plot_importance(
    axes[0],
    primal_plot,
    "Panel A: Primal algorithm"
)

plot_importance(
    axes[1],
    dual_plot,
    "Panel B: Dual algorithm"
)

# Common scale for direct comparison
axes[0].set_xlim(0, 0.52)

axes[1].set_xlabel(
    "Relative change in bound after permutation"
)

plt.tight_layout()
fig.subplots_adjust(hspace=0.25)

plt.savefig(
    "variable_importance.pdf",
    bbox_inches="tight"
)

fig.savefig(
    "results/variable_importance.pdf",
    bbox_inches="tight"
)

plt.show()