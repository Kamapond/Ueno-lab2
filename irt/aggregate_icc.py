"""
aggregate_icc.py — aggregate item characteristic curve and latent-ability plot
(paper: Appendix A; repository module `irt/`; produces the curves shown as Figs. 5, 7 and 9).

An individual item characteristic curve (ICC) describes one item. To characterize a whole
category, the aggregate ICC averages the estimated response probability over every item in that
category, giving the expected probability of a correct response at each level of latent ability
theta. Plotting each respondent's posterior-mean theta as a vertical line against that curve
shows directly where a model's ability falls relative to the difficulty and discrimination of
the item pool.

The theta grid spans -4 to +4, the range over which the curves are reported in the article.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit as sigmoid

THETA_MIN, THETA_MAX, THETA_POINTS = -4.0, 4.0, 301


def theta_grid(n: int = THETA_POINTS) -> np.ndarray:
    """Ability grid over which the aggregate ICC is evaluated."""
    return np.linspace(THETA_MIN, THETA_MAX, n)


def aggregate_icc(posterior_samples, grid: np.ndarray | None = None):
    """Return (grid, p_mean): the aggregate ICC averaged over all items.

    Posterior-mean item parameters are used, so `p_mean[i]` is the expected probability of a
    correct response at `grid[i]`, averaged across the item pool.
    """
    if grid is None:
        grid = theta_grid()

    a_mean_items = np.asarray(posterior_samples["a"]).mean(axis=0)
    b_mean_items = np.asarray(posterior_samples["b"]).mean(axis=0)

    # sigmoid(a * (theta - b)), vectorised over the grid and the item axis.
    p_items = sigmoid(
        np.outer(grid, a_mean_items)
        - np.outer(np.ones_like(grid), a_mean_items * b_mean_items)
    )

    return grid, p_items.mean(axis=1)


def icc_curve(theta, a, b):
    """Single-item ICC — the 2PL item response function stated in Section III-D."""
    return 1.0 / (1.0 + np.exp(-a * (theta - b)))


def plot_aggregate_icc(
    posterior_samples,
    model_names,
    model_colors,
    out_path: str = "aggregate_icc_with_models.png",
    grid: np.ndarray | None = None,
):
    """Draw the aggregate ICC with one vertical line per respondent at its posterior-mean theta.

    Args:
        posterior_samples: output of `irt.sampling.run_mcmc`.
        model_names:  respondent labels, in the row order used to build the response arrays.
        model_colors: mapping from label to any matplotlib colour.
        out_path:     destination file; 150 dpi is adequate for review, and the article's
                      figures are regenerated as vector output for submission.
    """
    grid, p_mean = aggregate_icc(posterior_samples, grid)

    plt.figure(figsize=(10, 6))
    plt.plot(
        grid, p_mean,
        label="Aggregate ICC (Mean)",
        color="black",
        linewidth=2,
    )

    theta_means = np.asarray(posterior_samples["theta"]).mean(axis=0)
    for model_name, theta_val in zip(model_names, theta_means):
        plt.axvline(
            theta_val,
            color=model_colors[model_name],
            linestyle="--",
            alpha=0.6,
        )
        plt.text(theta_val + 0.05, 0.02, model_name, rotation=90, fontsize=8)

    plt.xlabel("Ability (θ)", fontsize=12)
    plt.ylabel("Probability of Correct Response", fontsize=12)
    plt.xlim(THETA_MIN, THETA_MAX)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path
