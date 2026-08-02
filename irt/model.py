"""
model.py — two-parameter logistic (2PL) item response model with a binomial likelihood
(paper: Appendix A; repository module `irt/`).

Ability is estimated jointly for every respondent (each large language model under each
condition, plus the expert benchmark) over the 904-item certification benchmark. Each item is
presented under five option-order patterns, so the observation for one respondent on one item
is a count of successes out of `trials` rather than a single binary response; a binomial
likelihood models those repeated presentations directly instead of expanding them into
independent Bernoulli rows.

Parameterization
    theta   latent ability, one per respondent, re-centred to mean 0 at every estimation
    a       discrimination, one per item, truncated at 0 so the curve never decreases
    b       difficulty, one per item, non-centred through LocScaleReparam
    beta    likelihood tempering weight (1.0 = untempered, the value used in the paper)

The hyperprior scales — HalfNormal(0.35) on the discrimination spread and HalfNormal(1.0) on
the difficulty spread — are the values used for every estimation reported in the article.

Because theta is re-centred to mean 0 within each estimation, only differences in theta
*inside one estimation* are comparable; absolute values must not be compared across runs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import LocScaleReparam


def irt_2pl_binomial(successes, trials, N, K, beta: float = 1.0, **hyper_params):
    """2PL-binomial model definition.

    Args:
        successes: (N, K) integer array — correct responses per respondent per item.
        trials:    (N, K) integer array — presentations per respondent per item (5 in the paper).
        N:         number of respondents.
        K:         number of items.
        beta:      likelihood tempering weight.
        **hyper_params: 'a' and 'b' override the hyperprior scales (defaults 0.35 and 1.0).
    """
    hyper_a_sd = hyper_params.get("a", 0.35)
    hyper_b_sd = hyper_params.get("b", 1.0)

    theta1 = numpyro.sample(
        "theta1",
        dist.Normal(0, 1).expand([N]).to_event(1),
    )
    # Re-centre to mean 0. Only within-estimation differences in theta are comparable.
    theta = numpyro.deterministic(
        "theta",
        theta1 - jnp.mean(theta1, axis=0, keepdims=True),
    )

    log_a_sigma = numpyro.sample("log_a_sigma", dist.HalfNormal(hyper_a_sd))
    a = numpyro.sample(
        "a",
        dist.TruncatedNormal(1.0, log_a_sigma, low=0).expand([K]).to_event(1),
    )

    b_sd = numpyro.sample("b_sd", dist.HalfNormal(hyper_b_sd))
    with numpyro.handlers.reparam(config={"b": LocScaleReparam()}):
        b = numpyro.sample(
            "b",
            dist.Normal(0.0, b_sd).expand([K]).to_event(1),
        )

    logit = a * (theta[:, None] - b)
    p = jax.nn.sigmoid(logit)

    with numpyro.handlers.scale(scale=beta):
        with numpyro.plate("person", N):
            numpyro.sample(
                "obs",
                dist.Binomial(total_count=trials, probs=p).to_event(1),
                obs=successes,
            )
