"""
sampling.py — posterior sampling for the 2PL-binomial model
(paper: Appendix A, TABLE V; repository module `irt/`).

The posterior is sampled with NUTS (No-U-Turn Sampler), a form of Markov chain Monte Carlo.
Each model carries many parameters and the posterior is correspondingly high-dimensional, so a
long warm-up is used to stabilize convergence before samples are retained.

Settings below are the ones reported in TABLE V of the article. Convergence is judged on the
standard diagnostics printed by `mcmc.print_summary()` — the Gelman-Rubin statistic and the
effective sample size, together with the Monte-Carlo standard error.
"""

from __future__ import annotations

import jax.random as random
from numpyro.infer import MCMC, NUTS

from .model import irt_2pl_binomial

# Sampler settings (paper: Appendix A, TABLE V)
TARGET_ACCEPT_PROB = 0.8  # default value
NUM_WARMUP = 2000
NUM_SAMPLES = 2000
NUM_CHAINS = 4
SEED = 0


def run_mcmc(
    successes_array,
    trials_array,
    N: int,
    K: int,
    seed: int = SEED,
    progress_bar: bool = True,
):
    """Sample the posterior and return (posterior_samples, mcmc).

    The `mcmc` object is returned as well so that `mcmc.print_summary()` can be called for the
    convergence diagnostics.
    """
    nuts_kernel = NUTS(irt_2pl_binomial, target_accept_prob=TARGET_ACCEPT_PROB)

    mcmc = MCMC(
        nuts_kernel,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        chain_method="parallel",
        progress_bar=progress_bar,
    )

    mcmc.run(
        random.PRNGKey(seed),
        successes=successes_array,
        trials=trials_array,
        N=N,
        K=K,
    )

    return mcmc.get_samples(), mcmc
