/// Laplace-approximation Bradley-Terry estimator with per-judge positional bias.
///
/// A deterministic alternative to the MCMC sampler (`gaussian_bt`). It fits the
/// **same** posterior — the model is identical:
///
/// ```text
/// LL = Σ_c [ p_c·log σ(d_c) + (1−p_c)·log σ(−d_c) ],   d_c = θ_i − θ_j + β_k
/// ```
///
/// with a Gaussian prior on the item log-strengths `θ` (precision
/// `1/prior_tau2 + regularization_strength`, matching `gaussian_bt`'s prior plus
/// quadratic shrinkage) and a Gaussian prior on the per-judge biases `β`
/// (mean `bias_prior_mu`, precision `1/bias_prior_tau2`).
///
/// Instead of sampling, it finds the posterior mode (MAP) by Newton's method and
/// reads the curvature there: the negative Hessian `A = −H` is the observed
/// information, and `A⁻¹` is the Laplace covariance. So one fit yields both the
/// means (the mode) and the standard deviations (`√diag(A⁻¹)`) — the same
/// `(mean, std)` summary `gaussian_bt` produces from samples.
///
/// The log-posterior is concave (logistic log-likelihood + Gaussian priors), so
/// the mode is unique and Newton converges quickly. The prior makes `A`
/// positive-definite even for items with no comparisons, so the Cholesky solve
/// below never hits a singular matrix.
///
/// Internal module — operates on pre-mapped `usize` indices, not caller IDs.
use crate::types::IndexedComparison;

/// Result of a Laplace fit. Vectors are indexed the same as the inputs:
/// `means[i]`/`stds[i]` for item `i`, `bias_means[k]`/`bias_stds[k]` for judge `k`.
pub struct LaplaceFit {
    /// MAP log-strengths, mean-centered (matching `gaussian_bt`'s convention).
    pub means: Vec<f64>,
    /// Posterior standard deviation per item, `√diag(A⁻¹)`.
    pub stds: Vec<f64>,
    /// MAP per-judge positional bias (logit space, positive = favors item1).
    pub bias_means: Vec<f64>,
    /// Posterior standard deviation per judge bias.
    pub bias_stds: Vec<f64>,
}

/// Numerically stable sigmoid σ(x) = 1/(1+e^−x).
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Numerically stable log σ(x).
fn log_sigmoid(x: f64) -> f64 {
    if x > 0.0 {
        -((-x).exp().ln_1p())
    } else {
        x - x.exp().ln_1p()
    }
}

/// Linear-time Laplace fit of the model.
///
/// Finds the posterior mode (MAP) by **Newton-CG** — conjugate gradient using
/// Hessian-vector products, so the dense Hessian is never formed — and reads a
/// **diagonal-Fisher** std (`1/√A_ii`) at the mode.
///
/// Cost is O(#comparisons) per inner CG step and O(#comparisons) for the std, so
/// it scales to large item counts. The std is approximate — it ignores
/// between-item correlations and so tends to underestimate uncertainty — but the
/// std only feeds selection weighting, not the ranking.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn fit_linear(
    num_items: usize,
    num_judges: usize,
    comparisons: &[IndexedComparison],
    prior_tau2: f64,
    regularization_strength: f64,
    bias_prior_mu: f64,
    bias_prior_tau2: f64,
    max_iterations: usize,
    tol: f64,
) -> LaplaceFit {
    let dim = num_items + num_judges;
    let theta_precision = 1.0 / prior_tau2 + regularization_strength;
    let bias_precision = 1.0 / bias_prior_tau2;

    let mut phi = vec![0.0; dim];
    for k in 0..num_judges {
        phi[num_items + k] = bias_prior_mu;
    }

    let log_posterior = |phi: &[f64]| -> f64 {
        let mut lp = 0.0;
        for &(i, j, p, k) in comparisons {
            let d = phi[i] - phi[j] + phi[num_items + k];
            lp += p * log_sigmoid(d) + (1.0 - p) * log_sigmoid(-d);
        }
        for &t in phi.iter().take(num_items) {
            lp -= 0.5 * theta_precision * t * t;
        }
        for k in 0..num_judges {
            let diff = phi[num_items + k] - bias_prior_mu;
            lp -= 0.5 * bias_precision * diff * diff;
        }
        lp
    };

    // Per-comparison Newton weight w_c = σ(1−σ), refreshed each Newton step.
    let mut w = vec![0.0; comparisons.len()];

    for _ in 0..max_iterations {
        // Gradient g and the weights w_c at the current point.
        let mut g = vec![0.0; dim];
        for (c, &(i, j, p, k)) in comparisons.iter().enumerate() {
            let kk = num_items + k;
            let d = phi[i] - phi[j] + phi[kk];
            let s = sigmoid(d);
            g[i] += p - s;
            g[j] -= p - s;
            g[kk] += p - s;
            w[c] = s * (1.0 - s);
        }
        for i in 0..num_items {
            g[i] -= theta_precision * phi[i];
        }
        for k in 0..num_judges {
            let kk = num_items + k;
            g[kk] -= bias_precision * (phi[kk] - bias_prior_mu);
        }

        let grad_inf = g.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        if grad_inf < tol {
            break;
        }

        // Hessian-vector product A·v = (priors)·v + Σ_c w_c (u_c·v) u_c, where
        // u_c = e_i − e_j + e_{kk}. Never materializes A.
        let hessvec = |v: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0; dim];
            for i in 0..num_items {
                out[i] = theta_precision * v[i];
            }
            for k in 0..num_judges {
                let kk = num_items + k;
                out[kk] = bias_precision * v[kk];
            }
            for (c, &(i, j, _p, k)) in comparisons.iter().enumerate() {
                let kk = num_items + k;
                let s = w[c] * (v[i] - v[j] + v[kk]);
                out[i] += s;
                out[j] -= s;
                out[kk] += s;
            }
            out
        };

        let delta = conjugate_gradient(&hessvec, &g, dim, dim, 1e-10);

        // Damped step: full Newton unless it lowers the log-posterior.
        let lp_current = log_posterior(&phi);
        let mut step = 1.0;
        let mut accepted = false;
        for _ in 0..30 {
            let candidate: Vec<f64> = (0..dim).map(|m| phi[m] + step * delta[m]).collect();
            if log_posterior(&candidate) >= lp_current {
                phi = candidate;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }
    }

    // Diagonal-Fisher std: A_ii = prior precision + Σ_{c touching i} w_c at the
    // mode. std = 1/√A_ii (ignores off-diagonal → underestimates uncertainty).
    let mut info = vec![0.0; dim];
    for i in 0..num_items {
        info[i] = theta_precision;
    }
    for k in 0..num_judges {
        info[num_items + k] = bias_precision;
    }
    for &(i, j, _p, k) in comparisons {
        let kk = num_items + k;
        let d = phi[i] - phi[j] + phi[kk];
        let s = sigmoid(d);
        let wc = s * (1.0 - s);
        info[i] += wc;
        info[j] += wc;
        info[kk] += wc;
    }

    let theta_mean: f64 = (0..num_items).map(|i| phi[i]).sum::<f64>() / num_items.max(1) as f64;
    let means: Vec<f64> = (0..num_items).map(|i| phi[i] - theta_mean).collect();
    let stds: Vec<f64> = (0..num_items).map(|i| (1.0 / info[i]).sqrt()).collect();
    let bias_means: Vec<f64> = (0..num_judges).map(|k| phi[num_items + k]).collect();
    let bias_stds: Vec<f64> = (0..num_judges)
        .map(|k| (1.0 / info[num_items + k]).sqrt())
        .collect();

    LaplaceFit { means, stds, bias_means, bias_stds }
}

/// Conjugate gradient solve of `A x = b` for a symmetric positive-definite `A`
/// supplied only as a matrix-vector product `hessvec`. Returns `x`.
fn conjugate_gradient(
    hessvec: &impl Fn(&[f64]) -> Vec<f64>,
    b: &[f64],
    dim: usize,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };

    let mut x = vec![0.0; dim];
    let mut r = b.to_vec(); // r = b − A·0 = b
    let mut p = r.clone();
    let mut rs_old = dot(&r, &r);
    if rs_old.sqrt() < tol {
        return x;
    }

    for _ in 0..max_iter {
        let ap = hessvec(&p);
        let denom = dot(&p, &ap);
        if denom <= 0.0 {
            break; // not positive-definite along p (shouldn't happen with priors)
        }
        let alpha = rs_old / denom;
        for i in 0..dim {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new = dot(&r, &r);
        if rs_new.sqrt() < tol {
            break;
        }
        let beta = rs_new / rs_old;
        for i in 0..dim {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    // Defaults mirroring the CLI/MCMC.
    const PRIOR_TAU2: f64 = 10.0;
    const REG: f64 = 0.01;
    const BIAS_PRIOR_MU: f64 = 0.0;
    const BIAS_PRIOR_TAU2: f64 = 2.0;

    fn fit_default(num_items: usize, num_judges: usize, comps: &[IndexedComparison]) -> LaplaceFit {
        fit_linear(num_items, num_judges, comps, PRIOR_TAU2, REG, BIAS_PRIOR_MU, BIAS_PRIOR_TAU2, 100, 1e-10)
    }

    #[test]
    fn test_orders_clear_winner() {
        // Item 0 beats item 1 every time → θ_0 > θ_1, both stds positive.
        let comps: Vec<IndexedComparison> =
            (0..20).map(|_| (0usize, 1usize, 1.0, 0usize)).collect();
        let f = fit_default(2, 1, &comps);
        assert!(f.means[0] > f.means[1], "winner should rank higher: {:?}", f.means);
        assert!(f.stds[0] > 0.0 && f.stds[1] > 0.0);
        // Mean-centered → means sum to ~0.
        assert!((f.means[0] + f.means[1]).abs() < 1e-9);
    }

    #[test]
    fn test_more_data_shrinks_std() {
        let few: Vec<IndexedComparison> = (0..4).map(|n| (0usize, 1usize, (n % 2) as f64, 0usize)).collect();
        let many: Vec<IndexedComparison> = (0..200).map(|n| (0usize, 1usize, (n % 2) as f64, 0usize)).collect();
        let f_few = fit_default(2, 1, &few);
        let f_many = fit_default(2, 1, &many);
        assert!(f_many.stds[0] < f_few.stds[0], "more data should tighten std: {} vs {}", f_many.stds[0], f_few.stds[0]);
    }

    #[test]
    fn test_unplayed_item_falls_back_to_prior_std() {
        // Item 2 plays nothing; its std should be the prior std √(1/precision).
        let comps: Vec<IndexedComparison> = (0..10).map(|_| (0usize, 1usize, 1.0, 0usize)).collect();
        let f = fit_default(3, 1, &comps);
        let prior_std = (1.0 / (1.0 / PRIOR_TAU2 + REG)).sqrt();
        assert!((f.stds[2] - prior_std).abs() < 1e-6, "unplayed std {} vs prior {}", f.stds[2], prior_std);
    }

    #[test]
    fn test_recovers_positional_bias() {
        // Symmetric strengths (θ_0 = θ_1) but item1 wins 90% — the asymmetry must
        // be absorbed by a positive bias, not by strength differences. We feed
        // both orderings equally so strengths stay tied and β carries the signal.
        let mut comps: Vec<IndexedComparison> = Vec::new();
        for _ in 0..100 {
            comps.push((0, 1, 0.9, 0)); // item1=0 wins 90%
            comps.push((1, 0, 0.9, 0)); // item1=1 wins 90% (same first-position edge)
        }
        let f = fit_default(2, 1, &comps);
        assert!(f.bias_means[0] > 0.5, "first-position edge → positive bias, got {}", f.bias_means[0]);
        // With the position edge explained by bias, strengths stay ~equal.
        assert!((f.means[0] - f.means[1]).abs() < 0.2, "strengths should stay close: {:?}", f.means);
    }
}
