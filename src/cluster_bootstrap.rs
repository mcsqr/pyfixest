use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Solve OLS via Cholesky: beta = (X'X)^{-1} X'y
/// For small k (typical in bootstrap), this is faster than LAPACK dispatch.
fn solve_ols_cholesky(xtx: &Array2<f64>, xty: &Array1<f64>) -> Option<Array1<f64>> {
    let k = xtx.nrows();
    // Cholesky decomposition: xtx = L * L^T
    let mut l = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..=i {
            let mut sum = xtx[[i, j]];
            for p in 0..j {
                sum -= l[[i, p]] * l[[j, p]];
            }
            if i == j {
                if sum <= 0.0 {
                    return None; // Not positive definite
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    // Solve L * z = xty (forward substitution)
    let mut z = Array1::<f64>::zeros(k);
    for i in 0..k {
        let mut sum = xty[i];
        for j in 0..i {
            sum -= l[[i, j]] * z[j];
        }
        z[i] = sum / l[[i, i]];
    }
    // Solve L^T * beta = z (back substitution)
    let mut beta = Array1::<f64>::zeros(k);
    for i in (0..k).rev() {
        let mut sum = z[i];
        for j in (i + 1)..k {
            sum -= l[[j, i]] * beta[j];
        }
        beta[i] = sum / l[[i, i]];
    }
    Some(beta)
}

/// Core cluster bootstrap implementation.
///
/// For each bootstrap iteration:
/// 1. Resample clusters with replacement (stratified by group)
/// 2. Expand to observation indices
/// 3. Re-encode FEs to contiguous indices for the subsample
/// 4. Demean Y and X
/// 5. Solve OLS via Cholesky
/// 6. Compute group means of extra columns
///
/// Returns (coefs, group_means) where:
/// - coefs: (n_bootstrap, k) — OLS coefficients per iteration
/// - group_means: (n_bootstrap, n_groups, n_extra) — mean of extra_cols per group per iteration
fn cluster_bootstrap_impl(
    y: &ArrayView1<f64>,                // (n_obs,)
    x: &ArrayView2<f64>,                // (n_obs, k)
    fe: &ArrayView2<usize>,             // (n_obs, n_fe) — integer-encoded FE columns
    obs_to_cluster: &ArrayView1<usize>, // (n_obs,) — cluster assignment per obs
    obs_to_group: &ArrayView1<usize>,   // (n_obs,) — treatment group (0/1)
    weights: &ArrayView1<f64>,          // (n_obs,)
    extra_cols: &ArrayView2<f64>,       // (n_obs, n_extra) — additional columns for group means
    n_bootstrap: usize,
    seed: u64,
    demean_tol: f64,
    demean_maxiter: usize,
) -> (Array2<f64>, Array2<f64>) {
    let n_obs = y.len();
    let k = x.ncols();
    let n_fe = fe.ncols();
    let n_extra = extra_cols.ncols();

    // Determine unique groups and build cluster lists per group
    let n_groups = obs_to_group.iter().cloned().max().unwrap_or(0) + 1;
    let n_clusters = obs_to_cluster.iter().cloned().max().unwrap_or(0) + 1;

    // Build obs indices per cluster
    let mut cluster_obs: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    for i in 0..n_obs {
        cluster_obs[obs_to_cluster[i]].push(i);
    }

    // Build cluster lists per group (for stratified resampling)
    let mut clusters_by_group: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    // Determine group of each cluster (from first obs)
    let mut cluster_group = vec![0usize; n_clusters];
    for i in 0..n_obs {
        let cid = obs_to_cluster[i];
        cluster_group[cid] = obs_to_group[i];
    }
    for cid in 0..n_clusters {
        if !cluster_obs[cid].is_empty() {
            clusters_by_group[cluster_group[cid]].push(cid);
        }
    }

    let has_fe = n_fe > 0;

    // Parallel bootstrap iterations
    let results: Vec<(Array1<f64>, Vec<f64>)> = (0..n_bootstrap)
        .into_par_iter()
        .filter_map(|b| {
            // Simple LCG-based RNG per iteration (deterministic, no shared state)
            let iter_seed = seed.wrapping_add(b as u64);
            let mut rng_state = iter_seed;

            let mut obs_indices: Vec<usize> = Vec::with_capacity(n_obs + n_obs / 4);

            // Stratified cluster resampling
            for g in 0..n_groups {
                let group_clusters = &clusters_by_group[g];
                let n_gc = group_clusters.len();
                if n_gc == 0 {
                    continue;
                }
                for _ in 0..n_gc {
                    // LCG random index
                    rng_state = rng_state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let idx = ((rng_state >> 33) as usize) % n_gc;
                    let chosen_cluster = group_clusters[idx];
                    obs_indices.extend_from_slice(&cluster_obs[chosen_cluster]);
                }
            }

            let n_sub = obs_indices.len();
            if n_sub == 0 {
                return None;
            }

            // Build subsample arrays
            let mut yx_data = Array2::<f64>::zeros((n_sub, 1 + k));
            let mut sub_weights = vec![1.0f64; n_sub];
            let mut sub_fe: Vec<Vec<usize>> = vec![Vec::with_capacity(n_sub); n_fe];
            let mut sub_extra = Array2::<f64>::zeros((n_sub, n_extra));
            let mut sub_group = vec![0usize; n_sub];

            for (si, &oi) in obs_indices.iter().enumerate() {
                yx_data[[si, 0]] = y[oi];
                for j in 0..k {
                    yx_data[[si, 1 + j]] = x[[oi, j]];
                }
                sub_weights[si] = weights[oi];
                for f_idx in 0..n_fe {
                    sub_fe[f_idx].push(fe[[oi, f_idx]]);
                }
                for e_idx in 0..n_extra {
                    sub_extra[[si, e_idx]] = extra_cols[[oi, e_idx]];
                }
                sub_group[si] = obs_to_group[oi];
            }

            // Re-encode FE to contiguous indices for subsample
            if has_fe {
                for f_idx in 0..n_fe {
                    let col = &mut sub_fe[f_idx];
                    let max_val = col.iter().cloned().max().unwrap_or(0);
                    let mut mapping = vec![usize::MAX; max_val + 1];
                    let mut next_id = 0usize;
                    for val in col.iter_mut() {
                        if mapping[*val] == usize::MAX {
                            mapping[*val] = next_id;
                            next_id += 1;
                        }
                        *val = mapping[*val];
                    }
                }
            }

            // Demean if fixed effects present, reusing the shared accelerated MAP kernel.
            if has_fe {
                let mut flist = Array2::<usize>::zeros((n_sub, n_fe));
                for f_idx in 0..n_fe {
                    for i in 0..n_sub {
                        flist[[i, f_idx]] = sub_fe[f_idx][i];
                    }
                }
                let sub_weights_arr = Array1::from(sub_weights.clone());
                let (demeaned, _converged) = crate::demean::demean_impl(
                    &yx_data.view(),
                    &flist.view(),
                    &sub_weights_arr.view(),
                    demean_tol,
                    demean_maxiter,
                );
                yx_data = demeaned;
            }

            // Extract demeaned Y and X
            let y_d: Vec<f64> = (0..n_sub).map(|i| yx_data[[i, 0]]).collect();
            let x_d = yx_data.slice(s![.., 1..]);

            // Apply WLS transform if weights aren't all 1
            let has_weights = sub_weights.iter().any(|&w| (w - 1.0).abs() > 1e-12);
            let (y_final, x_final) = if has_weights {
                let sqrt_w: Vec<f64> = sub_weights.iter().map(|&w| w.sqrt()).collect();
                let y_w: Vec<f64> = y_d
                    .iter()
                    .zip(sqrt_w.iter())
                    .map(|(&yi, &wi)| yi * wi)
                    .collect();
                let mut x_w = Array2::<f64>::zeros((n_sub, k));
                for i in 0..n_sub {
                    for j in 0..k {
                        x_w[[i, j]] = x_d[[i, j]] * sqrt_w[i];
                    }
                }
                (y_w, x_w)
            } else {
                let x_owned = x_d.to_owned();
                (y_d, x_owned)
            };

            // Compute X'X and X'y
            let mut xtx = Array2::<f64>::zeros((k, k));
            let mut xty = Array1::<f64>::zeros(k);
            for i in 0..n_sub {
                for j in 0..k {
                    xty[j] += x_final[[i, j]] * y_final[i];
                    for l in j..k {
                        let val = x_final[[i, j]] * x_final[[i, l]];
                        xtx[[j, l]] += val;
                        if l != j {
                            xtx[[l, j]] += val;
                        }
                    }
                }
            }

            // Solve OLS
            let beta = solve_ols_cholesky(&xtx, &xty)?;

            // Compute group means of extra columns
            let mut group_counts = vec![0.0f64; n_groups];
            let mut group_sums = vec![0.0f64; n_groups * n_extra];
            for i in 0..n_sub {
                let g = sub_group[i];
                group_counts[g] += 1.0;
                for e in 0..n_extra {
                    group_sums[g * n_extra + e] += sub_extra[[i, e]];
                }
            }
            let mut means = vec![0.0f64; n_groups * n_extra];
            for g in 0..n_groups {
                if group_counts[g] > 0.0 {
                    for e in 0..n_extra {
                        means[g * n_extra + e] = group_sums[g * n_extra + e] / group_counts[g];
                    }
                }
            }

            Some((beta, means))
        })
        .collect();

    let n_valid = results.len();
    let mut coefs = Array2::<f64>::zeros((n_valid, k));
    // group_means shape: (n_valid, n_groups * n_extra) — flattened, reshape in Python
    let mut group_means = Array2::<f64>::zeros((n_valid, n_groups * n_extra));

    for (b, (beta, means)) in results.iter().enumerate() {
        for j in 0..k {
            coefs[[b, j]] = beta[j];
        }
        for idx in 0..n_groups * n_extra {
            group_means[[b, idx]] = means[idx];
        }
    }

    (coefs, group_means)
}

/// Perform stratified cluster bootstrap OLS estimation entirely in Rust.
///
/// This function runs `n_bootstrap` iterations of:
/// 1. Stratified cluster resampling (by treatment group)
/// 2. Fixed-effect demeaning (Irons-Tuck accelerated MAP)
/// 3. OLS coefficient estimation (Cholesky)
/// 4. Group-mean computation for auxiliary columns
///
/// All iterations run in parallel via rayon with no Python GIL interaction.
///
/// Parameters
/// ----------
/// y : np.ndarray[float64], shape (n_obs,)
///     Dependent variable (raw, pre-demeaning).
/// x : np.ndarray[float64], shape (n_obs, k)
///     Independent variables (raw, pre-demeaning).
/// fe : np.ndarray[uint64], shape (n_obs, n_fe)
///     Integer-encoded fixed effects. Pass empty (n_obs, 0) if no FEs.
/// obs_to_cluster : np.ndarray[uint64], shape (n_obs,)
///     Cluster assignment for each observation.
/// obs_to_group : np.ndarray[uint64], shape (n_obs,)
///     Treatment group (0, 1, ...) for each observation.
/// weights : np.ndarray[float64], shape (n_obs,)
///     Observation weights (ones if unweighted).
/// extra_cols : np.ndarray[float64], shape (n_obs, n_extra)
///     Additional columns to compute per-group means on each iteration.
/// n_bootstrap : int
///     Number of bootstrap iterations.
/// seed : int
///     Random seed for reproducibility.
/// demean_tol : float
///     Convergence tolerance for demeaning.
/// demean_maxiter : int
///     Maximum iterations for demeaning.
///
/// Returns
/// -------
/// tuple[np.ndarray, np.ndarray]
///     - coefs: shape (n_valid, k) — OLS coefficients for each successful iteration
///     - group_means: shape (n_valid, n_groups * n_extra) — flattened group means
///       Reshape to (n_valid, n_groups, n_extra) in Python.
#[pyfunction]
#[pyo3(signature = (y, x, fe, obs_to_cluster, obs_to_group, weights, extra_cols, n_bootstrap, seed, demean_tol=1e-8, demean_maxiter=100_000))]
pub fn _cluster_bootstrap_rs(
    py: Python<'_>,
    y: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    fe: PyReadonlyArray2<usize>,
    obs_to_cluster: PyReadonlyArray1<usize>,
    obs_to_group: PyReadonlyArray1<usize>,
    weights: PyReadonlyArray1<f64>,
    extra_cols: PyReadonlyArray2<f64>,
    n_bootstrap: usize,
    seed: u64,
    demean_tol: f64,
    demean_maxiter: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let y_arr = y.as_array();
    let x_arr = x.as_array();
    let fe_arr = fe.as_array();
    let cluster_arr = obs_to_cluster.as_array();
    let group_arr = obs_to_group.as_array();
    let weights_arr = weights.as_array();
    let extra_arr = extra_cols.as_array();

    let (coefs, group_means) = py.detach(|| {
        cluster_bootstrap_impl(
            &y_arr,
            &x_arr,
            &fe_arr,
            &cluster_arr,
            &group_arr,
            &weights_arr,
            &extra_arr,
            n_bootstrap,
            seed,
            demean_tol,
            demean_maxiter,
        )
    });

    let coefs_py = coefs.into_pyarray(py).to_owned().into();
    let means_py = group_means.into_pyarray(py).to_owned().into();
    Ok((coefs_py, means_py))
}
