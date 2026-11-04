"""LiDAR (Thilak et al., arXiv:2312.04000), the implementation the paper uses.

Copied verbatim from the companion study's monitor code, which the
analysis previously loaded from there."""
import torch


def lidar(feats_MKD, delta=1e-4):
    """LiDAR (arXiv:2312.04000): entropy-exponent of eigenvalues of
    Sw^{-1/2} Sb Sw^{-1/2}; surrogate classes = the M resim truths."""
    Mc, Kc, D = feats_MKD.shape
    mu_c = feats_MKD.mean(1)
    mu = mu_c.mean(0)
    Sb = (mu_c - mu).T @ (mu_c - mu) / (Mc - 1)
    Xw = (feats_MKD - mu_c[:, None, :]).reshape(-1, D)
    Sw = Xw.T @ Xw / (Mc * (Kc - 1)) + delta * torch.eye(D, device=Xw.device)
    ew, U = torch.linalg.eigh(Sw)
    W = U @ torch.diag(ew.clamp(min=1e-12).rsqrt()) @ U.T
    lam = torch.linalg.eigvalsh(W @ Sb @ W).clamp(min=0)
    p = lam / (lam.sum() + 1e-12) + 1e-12
    return float(torch.exp(-(p * torch.log(p)).sum()))
