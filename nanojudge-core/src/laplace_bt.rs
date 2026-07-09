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
    /// MAP per-judge positional bias (logit space, positive = favors item1). This
    /// is the slot-0 (first-position) advantage — the pairwise β, and the summary
    /// for multi-slot runs.
    pub bias_means: Vec<f64>,
    /// Posterior standard deviation per judge bias (slot-0).
    pub bias_stds: Vec<f64>,
    /// Full MAP free per-slot advantages per judge (length `num_slots − 1` each;
    /// the reference slot is omitted). Slot 0 equals `bias_means`. Used to carry
    /// the complete bias state into the next round's warm start.
    pub bias_free: Vec<Vec<f64>>,
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
    // Number of presentation slots: one past the highest slot index seen, at
    // least 2. Each judge gets `num_slots − 1` free advantage params; the highest
    // slot is the pinned zero reference.
    let mut num_slots = 2usize;
    for &(_, _, _, _, s1, s2) in comparisons {
        num_slots = num_slots.max(s1 as usize + 1).max(s2 as usize + 1);
    }
    let free_per_judge = num_slots - 1;
    let dim = num_items + num_judges * free_per_judge;
    let theta_precision = 1.0 / prior_tau2 + regularization_strength;
    let bias_precision = 1.0 / bias_prior_tau2;

    // Parameter index of judge `k`'s slot `s` advantage, or None if `s` is the
    // reference slot (contributes 0, no parameter).
    let bias_idx = |k: usize, s: usize| -> Option<usize> {
        if s + 1 >= num_slots {
            None
        } else {
            Some(num_items + k * free_per_judge + s)
        }
    };

    let mut phi = vec![0.0; dim];
    for idx in num_items..dim {
        phi[idx] = bias_prior_mu;
    }

    let log_posterior = |phi: &[f64]| -> f64 {
        let mut lp = 0.0;
        for &(i, j, p, k, s1, s2) in comparisons {
            let g1 = bias_idx(k, s1 as usize).map(|b| phi[b]).unwrap_or(0.0);
            let g2 = bias_idx(k, s2 as usize).map(|b| phi[b]).unwrap_or(0.0);
            let d = phi[i] - phi[j] + g1 - g2;
            lp += p * log_sigmoid(d) + (1.0 - p) * log_sigmoid(-d);
        }
        for &t in phi.iter().take(num_items) {
            lp -= 0.5 * theta_precision * t * t;
        }
        for idx in num_items..dim {
            let diff = phi[idx] - bias_prior_mu;
            lp -= 0.5 * bias_precision * diff * diff;
        }
        lp
    };

    // Per-comparison Newton weight w_c = σ(1−σ), refreshed each Newton step.
    let mut w = vec![0.0; comparisons.len()];

    for _ in 0..max_iterations {
        // Gradient g and the weights w_c at the current point.
        let mut g = vec![0.0; dim];
        for (c, &(i, j, p, k, s1, s2)) in comparisons.iter().enumerate() {
            let b1 = bias_idx(k, s1 as usize);
            let b2 = bias_idx(k, s2 as usize);
            let g1 = b1.map(|b| phi[b]).unwrap_or(0.0);
            let g2 = b2.map(|b| phi[b]).unwrap_or(0.0);
            let d = phi[i] - phi[j] + g1 - g2;
            let s = sigmoid(d);
            let resid = p - s;
            g[i] += resid;
            g[j] -= resid;
            if let Some(b) = b1 { g[b] += resid; }
            if let Some(b) = b2 { g[b] -= resid; }
            w[c] = s * (1.0 - s);
        }
        for i in 0..num_items {
            g[i] -= theta_precision * phi[i];
        }
        for idx in num_items..dim {
            g[idx] -= bias_precision * (phi[idx] - bias_prior_mu);
        }

        let grad_inf = g.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        if grad_inf < tol {
            break;
        }

        // Hessian-vector product A·v = (priors)·v + Σ_c w_c (u_c·v) u_c, where
        // u_c = e_i − e_j + e_{b1} − e_{b2} (bias terms present only for free
        // slots). Never materializes A.
        let hessvec = |v: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0; dim];
            for i in 0..num_items {
                out[i] = theta_precision * v[i];
            }
            for idx in num_items..dim {
                out[idx] = bias_precision * v[idx];
            }
            for (c, &(i, j, _p, k, s1, s2)) in comparisons.iter().enumerate() {
                let b1 = bias_idx(k, s1 as usize);
                let b2 = bias_idx(k, s2 as usize);
                let mut uv = v[i] - v[j];
                if let Some(b) = b1 { uv += v[b]; }
                if let Some(b) = b2 { uv -= v[b]; }
                let s = w[c] * uv;
                out[i] += s;
                out[j] -= s;
                if let Some(b) = b1 { out[b] += s; }
                if let Some(b) = b2 { out[b] -= s; }
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
    for idx in num_items..dim {
        info[idx] = bias_precision;
    }
    for &(i, j, _p, k, s1, s2) in comparisons {
        let b1 = bias_idx(k, s1 as usize);
        let b2 = bias_idx(k, s2 as usize);
        let g1 = b1.map(|b| phi[b]).unwrap_or(0.0);
        let g2 = b2.map(|b| phi[b]).unwrap_or(0.0);
        let d = phi[i] - phi[j] + g1 - g2;
        let s = sigmoid(d);
        let wc = s * (1.0 - s);
        info[i] += wc;
        info[j] += wc;
        if let Some(b) = b1 { info[b] += wc; }
        if let Some(b) = b2 { info[b] += wc; }
    }

    let theta_mean: f64 = (0..num_items).map(|i| phi[i]).sum::<f64>() / num_items.max(1) as f64;
    let means: Vec<f64> = (0..num_items).map(|i| phi[i] - theta_mean).collect();
    let stds: Vec<f64> = (0..num_items).map(|i| (1.0 / info[i]).sqrt()).collect();
    // Report slot 0 (first-position advantage) as the judge bias summary.
    let bias_means: Vec<f64> = (0..num_judges)
        .map(|k| phi[bias_idx(k, 0).expect("slot 0 is always free")])
        .collect();
    let bias_stds: Vec<f64> = (0..num_judges)
        .map(|k| (1.0 / info[bias_idx(k, 0).expect("slot 0 is always free")]).sqrt())
        .collect();
    // Full free advantage vector per judge, for warm-starting the next round.
    let bias_free: Vec<Vec<f64>> = (0..num_judges)
        .map(|k| (0..free_per_judge).map(|s| phi[num_items + k * free_per_judge + s]).collect())
        .collect();

    LaplaceFit { means, stds, bias_means, bias_stds, bias_free }
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
            (0..20).map(|_| (0usize, 1usize, 1.0, 0usize, 0, 1)).collect();
        let f = fit_default(2, 1, &comps);
        assert!(f.means[0] > f.means[1], "winner should rank higher: {:?}", f.means);
        assert!(f.stds[0] > 0.0 && f.stds[1] > 0.0);
        // Mean-centered → means sum to ~0.
        assert!((f.means[0] + f.means[1]).abs() < 1e-9);
    }

    #[test]
    fn test_more_data_shrinks_std() {
        let few: Vec<IndexedComparison> = (0..4).map(|n| (0usize, 1usize, (n % 2) as f64, 0usize, 0, 1)).collect();
        let many: Vec<IndexedComparison> = (0..200).map(|n| (0usize, 1usize, (n % 2) as f64, 0usize, 0, 1)).collect();
        let f_few = fit_default(2, 1, &few);
        let f_many = fit_default(2, 1, &many);
        assert!(f_many.stds[0] < f_few.stds[0], "more data should tighten std: {} vs {}", f_many.stds[0], f_few.stds[0]);
    }

    #[test]
    fn test_unplayed_item_falls_back_to_prior_std() {
        // Item 2 plays nothing; its std should be the prior std √(1/precision).
        let comps: Vec<IndexedComparison> = (0..10).map(|_| (0usize, 1usize, 1.0, 0usize, 0, 1)).collect();
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
            comps.push((0, 1, 0.9, 0, 0, 1)); // item1=0 wins 90%
            comps.push((1, 0, 0.9, 0, 0, 1)); // item1=1 wins 90% (same first-position edge)
        }
        let f = fit_default(2, 1, &comps);
        assert!(f.bias_means[0] > 0.5, "first-position edge → positive bias, got {}", f.bias_means[0]);
        // With the position edge explained by bias, strengths stay ~equal.
        assert!((f.means[0] - f.means[1]).abs() < 0.2, "strengths should stay close: {:?}", f.means);
    }

    #[test]
    fn test_recovers_three_slot_bias() {
        // Three slots (A=0, B=1, C=2 reference). Symmetric strengths, but whoever
        // sits in slot A beats whoever sits in slot C 90% of the time — the edge
        // must be absorbed by slot A's advantage, not by strengths. Feed both
        // items through slot A equally so strengths stay tied.
        let mut comps: Vec<IndexedComparison> = Vec::new();
        for _ in 0..100 {
            comps.push((0, 1, 0.9, 0, 0, 2)); // item0 in slot A beats item1 in slot C
            comps.push((1, 0, 0.9, 0, 0, 2)); // item1 in slot A beats item0 in slot C
        }
        let f = fit_default(2, 1, &comps);
        // num_slots = 3 → two free advantages per judge [γ_A, γ_B]; slot C is the
        // reference. γ_A (bias_means / bias_free[_][0]) is the recovered slot-A edge.
        assert_eq!(f.bias_free[0].len(), 2, "two free slot advantages expected");
        assert!(f.bias_means[0] > 0.5, "slot-A advantage should be positive, got {}", f.bias_means[0]);
        assert!((f.means[0] - f.means[1]).abs() < 0.2, "strengths should stay close: {:?}", f.means);
    }
}
