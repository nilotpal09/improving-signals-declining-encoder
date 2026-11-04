"""WP1 identifiability metrics — the numbers F1 is made of.

Fixed in advance (M1-M4; M4 = upgrade U2,
approved 2026-08-16). No CCA / Procrustes / orthogonality code existed
anywhere in the stack (verified 2026-08-16), so everything here is written
fresh and pinned against exact synthetic ground truth in
test_wp1_metrics.py before any real checkpoint is scored.

Frozen conventions (these ARE the definitions fixed in advance; changing any
of them after the pilot freeze is a logged protocol change):

  * All fits happen on the TRAIN split; all reported numbers come from the
    HELD-OUT split. float64 throughout.
  * M1  ridge_r2 — the house canonical linear probe construction:
    features and targets z-scored on the train split, W = solve(X'X + 1.0*I, X'Y)
    with alpha FIXED at 1.0 (no CV), R^2 per target on eval.
  * M2  mean_cca — CCA fitted on train (regularized whitening of both
    blocks, SVD of the cross-covariance); the report is the mean over the
    n latent components of the PEARSON CORRELATION OF THE HELD-OUT
    canonical variates. Invariant under invertible linear maps of either
    block — the theorem-facing "linearly related" number.
  * M3  procrustes_mse — embeddings reduced to n dims via the train-fitted
    CCA projection (unit-variance canonical variates), best orthogonal map
    to z fitted on train (SVD Procrustes), mean squared error per
    coordinate on eval.
  * M4  orth_err — the U2 metric, computed in the ORIGINAL embedding space
    (CCA coordinates would absorb any shear — the invariance M4 exists to
    break): W = OLS map z -> centered h fitted on train (n x K); G = W W';
    err = ||G/s - I||_F / sqrt(n) with s = trace(G)/n. Scale-normalized on
    purpose: it measures whether the recovered map is orthogonal UP TO ONE
    GLOBAL SCALE (shape), while M2 measures strength. A degenerate map
    (s ~ 0, e.g. total collapse) reports err = 1.0 by convention.

Feature extraction and probe-set generation live in the driver
(wp1_probe_driver.py), not here: this module is pure metric math on
(features, targets) splits.
"""

import torch


def _f64(*ts):
    return [torch.as_tensor(t).double() for t in ts]


def ridge_r2(h_tr, z_tr, h_ev, z_ev, alpha=1.0):
    """M1: held-out R^2 per target, the house ridge construction."""
    X, Y, Xe, Ye = _f64(h_tr, z_tr, h_ev, z_ev)
    Xm, Xs = X.mean(0), X.std(0) + 1e-8
    Ym, Ys = Y.mean(0), Y.std(0) + 1e-8
    Xn, Yn = (X - Xm) / Xs, (Y - Ym) / Ys
    K = Xn.shape[1]
    W = torch.linalg.solve(Xn.T @ Xn + alpha * torch.eye(K, dtype=Xn.dtype),
                           Xn.T @ Yn)
    P = ((Xe - Xm) / Xs) @ W
    Yne = (Ye - Ym) / Ys
    ss_res = ((Yne - P) ** 2).sum(0)
    ss_tot = ((Yne - Yne.mean(0)) ** 2).sum(0) + 1e-12
    return (1.0 - ss_res / ss_tot).tolist()


def _inv_sqrt(S, eps_rel=1e-9):
    """Symmetric inverse square root with relative regularization."""
    w, V = torch.linalg.eigh(S)
    floor = eps_rel * float(w.max().clamp(min=1e-300)) + 1e-30
    w = w.clamp(min=floor)
    return V @ torch.diag(w.rsqrt()) @ V.T


def cca_metrics(h_tr, z_tr, h_ev, z_ev):
    """M2 + M3 + M4 in one pass (they share the train-fitted CCA basis).

    Returns dict: mean_cca, cca_per_component (list, len n), procrustes_mse,
    orth_err, orth_scale."""
    X, Y, Xe, Ye = _f64(h_tr, z_tr, h_ev, z_ev)
    n = Y.shape[1]
    n_comp = min(n, X.shape[1])   # only min(dim_h, dim_z) canonical pairs exist
    N = X.shape[0]

    Xm, Ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - Xm, Y - Ym
    Xec, Yec = Xe - Xm, Ye - Ym

    Sxx = Xc.T @ Xc / N
    Syy = Yc.T @ Yc / N
    Sxy = Xc.T @ Yc / N
    Wx = _inv_sqrt(Sxx)
    Wy = _inv_sqrt(Syy)
    U, S, Vh = torch.linalg.svd(Wx @ Sxy @ Wy)
    A = Wx @ U[:, :n_comp]     # embedding-side canonical directions
    B = Wy @ Vh.T[:, :n_comp]  # latent-side canonical directions

    # M2: Pearson correlation of the HELD-OUT canonical variates
    u, v = Xec @ A, Yec @ B
    per = []
    for i in range(n_comp):
        ui, vi = u[:, i], v[:, i]
        su = float(ui.std())
        sv = float(vi.std())
        if su < 1e-12 or sv < 1e-12:
            per.append(0.0)
        else:
            per.append(float(
                ((ui - ui.mean()) * (vi - vi.mean())).mean() / (su * sv)))
    mean_cca = sum(per) / n_comp

    # M3: Procrustes on the train variates, error on eval
    u_tr = Xc @ A
    Uq, _, Vqh = torch.linalg.svd(u_tr.T @ Yc, full_matrices=False)
    Q = Uq @ Vqh               # best (semi-)orthogonal map, rectangular-safe
    procrustes_mse = float(((u @ Q - Yec) ** 2).mean())

    # M4: orthogonality of the OLS map z -> h, original embedding space
    G_zz = Yc.T @ Yc + 1e-9 * torch.eye(n, dtype=Yc.dtype)
    W = torch.linalg.solve(G_zz, Yc.T @ Xc)      # (n, K): z @ W ~ h
    G = W @ W.T
    s = float(G.trace()) / n
    if s < 1e-12:
        orth_err = 1.0
    else:
        I = torch.eye(n, dtype=G.dtype)
        orth_err = float(torch.linalg.matrix_norm(G / s - I, ord='fro')) / n ** 0.5

    return {
        'mean_cca': mean_cca,
        'cca_per_component': per,
        'procrustes_mse': procrustes_mse,
        'orth_err': orth_err,
        'orth_scale': s,
    }


def wp1_metric_row(h_tr, z_tr, h_ev, z_ev, target_names=('z1', 'z2', 'E')):
    """All four metrics as one flat dict — one CSV row per (run, epoch)."""
    row = cca_metrics(h_tr, z_tr, h_ev, z_ev)
    r2 = ridge_r2(h_tr, z_tr, h_ev, z_ev)
    for i, v in enumerate(r2):
        name = target_names[i] if i < len(target_names) else f'z{i + 1}'
        row[f'ridge_r2_{name}'] = v
    return row
