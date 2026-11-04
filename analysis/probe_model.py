"""The truth-query probe (v1).

One query per TRUE particle, built from that particle's known position (and,
in the ablation, its energy). The queries cross-attend over the frozen
encoder's UNPOOLED token grid and each query emits (ln alpha_hat, ln T_hat)
for its own particle. There is deliberately NO query-to-query attention:
queries are independent given the tokens, which is what makes them
truth-anchored (permuting the queries permutes the predictions exactly) and
what makes the permuted-truth floor meaningful.

Capacity rungs (the probe-capacity ladder):
    small : 1 cross-attention block, d_model 32
    big   : 2 cross-attention blocks, d_model 64
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CAPACITY = {
    'small': dict(layers=1, d_model=32, heads=4),
    'big':   dict(layers=2, d_model=64, heads=4),
}
# Positional Fourier octaves in the query. The token grid is 16x16, so octave k
# has period (image width)/2^k; k=4 already resolves one token. Octaves far above
# that add nothing the probe can use to FIND a token and only give it a unique
# fingerprint per particle to memorize (seen in an earlier stability run).
N_FOURIER = 5


def query_features(x, y, lnE, half_w, half_h, lnE_mu, lnE_sd, use_energy,
                   n_fourier=None):
    """(B, P, F) query features from truth. Position always; energy only in the
    `pose` ablation."""
    n_fourier = N_FOURIER if n_fourier is None else n_fourier
    u = (x / half_w).clamp(-2, 2)
    v = (y / half_h).clamp(-2, 2)
    feats = [u, v]
    for k in range(n_fourier):
        f = (2.0 ** k) * math.pi
        feats += [torch.sin(f * u), torch.cos(f * u),
                  torch.sin(f * v), torch.cos(f * v)]
    if use_energy:
        feats.append((lnE - lnE_mu) / lnE_sd)
    return torch.stack(feats, dim=-1)


class CrossBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.nq, self.nk = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.nm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        # learnable strength of the "look near your own particle" prior; the probe
        # can switch it off by driving this to 0
        self.loc = nn.Parameter(torch.tensor(1.0))

    def forward(self, q, kv, bias=None):
        mask = None
        if bias is not None:
            B, Q, T = bias.shape
            mask = (self.loc * bias).unsqueeze(1).expand(B, self.heads, Q, T)
            mask = mask.reshape(B * self.heads, Q, T)
        h, _ = self.attn(self.nq(q), self.nk(kv), self.nk(kv),
                         attn_mask=mask, need_weights=False)
        q = q + h
        return q + self.mlp(self.nm(q))


def token_grid_positions(n_tokens, device):
    """Normalized (u, v) in [-1, 1] of each token's centre, in the row-major order
    PatchEmbed produces (row = image x axis, col = image y axis)."""
    g = int(round(n_tokens ** 0.5))
    c = (torch.arange(g, device=device, dtype=torch.float32) + 0.5) / g * 2 - 1
    uu, vv = torch.meshgrid(c, c, indexing='ij')
    return torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)      # (T, 2)


class QueryProbe(nn.Module):
    """Truth-query probe.

    Two ingredients let a query actually reach its own particle's tokens:
      * each token carries its own grid position, embedded with the same Fourier
        basis the query uses;
      * the attention logits get an additive, learnable-strength locality prior
        -‖pos_query − pos_token‖² / 2σ².
    Without them the probe sits on a plateau where the attention stays diffuse
    and it only ever reproduces the event mean, i.e. its own floor (measured
    in an earlier stability run). Both depend ONLY on the query's own position, so
    truth-anchoring is preserved exactly — permuting queries still permutes
    predictions (unit check U3).
    """

    def __init__(self, token_dim, n_feat, capacity='small', use_tokens=True,
                 n_fourier=None, pos_bias=True, sigma=0.25):
        super().__init__()
        cf = CAPACITY[capacity]
        d = cf['d_model']
        self.use_tokens = use_tokens
        self.pos_bias = pos_bias
        self.sigma = sigma
        self.n_fourier = N_FOURIER if n_fourier is None else n_fourier
        n_tok_feat = 2 + 4 * self.n_fourier
        self.tok_norm = nn.LayerNorm(token_dim)
        self.tok_proj = nn.Linear(token_dim, d)
        self.tok_pos = nn.Sequential(nn.Linear(n_tok_feat, d), nn.GELU(), nn.Linear(d, d))
        self.q_mlp = nn.Sequential(nn.Linear(n_feat, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([CrossBlock(d, cf['heads']) for _ in range(cf['layers'])])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
        self._pos_cache = None

    def _token_pos(self, tokens):
        if self._pos_cache is None or self._pos_cache.shape[0] != tokens.shape[1]:
            self._pos_cache = token_grid_positions(tokens.shape[1], tokens.device)
        return self._pos_cache

    def forward(self, tokens, qfeat, qpos=None):
        q = self.q_mlp(qfeat)
        kv = self.tok_proj(self.tok_norm(tokens.float()))
        if not self.use_tokens:
            kv = torch.zeros_like(kv)
        bias = None
        if self.use_tokens:
            tp = self._token_pos(tokens)                                   # (T, 2)
            tf = query_features(tp[:, 0], tp[:, 1], tp[:, 0], 1.0, 1.0, 0.0, 1.0,
                                False, self.n_fourier)                     # (T, F)
            kv = kv + self.tok_pos(tf).unsqueeze(0)
            if self.pos_bias and qpos is not None:
                d2 = ((qpos.unsqueeze(2) - tp.view(1, 1, -1, 2)) ** 2).sum(-1)
                bias = -d2 / (2 * self.sigma ** 2)
        for blk in self.blocks:
            q = blk(q, kv, bias)
        return self.head(q)


# ------------------------------------------------------- floor construction

def cyclic_shift_perm(mask, seed):
    """Permuted-truth floor: within each event, cyclically shift which particle's
    (alpha, T) each query must predict. A cyclic shift by 1..n-1 is a derangement,
    so NO query keeps its own target (a plain random permutation would leave ~1
    fixed point per event and leak real signal into the floor).

    Returns an index tensor `idx` (B, P) with idx[b, p] = slot whose target query
    p must predict. Pad slots map to themselves.
    """
    B, P = mask.shape
    rng = np.random.RandomState(seed & 0xFFFFFFFF)
    idx = np.tile(np.arange(P)[None, :], (B, 1))
    m = mask.cpu().numpy()
    for b in range(B):
        real = np.where(m[b])[0]
        n = len(real)
        if n < 2:
            continue
        k = 1 + rng.randint(n - 1)
        idx[b, real] = real[(np.arange(n) + k) % n]
    return torch.from_numpy(idx).long()


# --------------------------------------------------------------- training

def _r2(pred, targ):
    """Per-target R^2 over the pooled eval particles."""
    ss_res = ((pred - targ) ** 2).sum(0)
    ss_tot = ((targ - targ.mean(0, keepdim=True)) ** 2).sum(0)
    return (1.0 - ss_res / ss_tot).cpu().numpy()


def fit_probe(tok_tr, data_tr, tok_ev, data_ev, *, capacity, probe_seed,
              use_energy=False, use_tokens=True, permute=False,
              steps=2000, batch=128, lr=3e-3, wd=1e-4, half_w=5.0, half_h=5.0,
              eval_every=200, device=None, verbose=False, n_fourier=None,
              pos_bias=True):
    """Train one probe and return a result dict.

    tok_*  : (N, T, D) token grids (fp16 on device) or None when use_tokens=False
    data_* : dict of CPU tensors from gen_probe_events
    permute: True -> the permuted-truth FLOOR for this same cell
    """
    device = device or torch.device('cuda:0')
    g = torch.Generator(device='cpu').manual_seed(int(probe_seed))
    torch.manual_seed(int(probe_seed))

    def prep(data, tok, split_id):
        m = data['mask'].to(device)
        x, y = data['x'].to(device), data['y'].to(device)
        lnE = data['lnE'].to(device)
        la, lt = data['ln_alpha'].to(device), data['ln_T'].to(device)
        if permute:
            idx = cyclic_shift_perm(data['mask'], probe_seed * 1000 + split_id).to(device)
            la = torch.gather(la, 1, idx)
            lt = torch.gather(lt, 1, idx)
        targ = torch.stack([la, lt], dim=-1)
        return dict(mask=m, x=x, y=y, lnE=lnE, targ=targ, tok=tok)

    tr = prep(data_tr, tok_tr, 0)
    ev = prep(data_ev, tok_ev, 1)

    # normalization constants come from the TRAIN split only
    mtr = tr['mask']
    lnE_mu = tr['lnE'][mtr].mean()
    lnE_sd = tr['lnE'][mtr].std() + 1e-6
    y_mu = tr['targ'][mtr].mean(0)
    y_sd = tr['targ'][mtr].std(0) + 1e-6

    for d in (tr, ev):
        d['qf'] = query_features(d['x'], d['y'], d['lnE'], half_w, half_h,
                                 lnE_mu, lnE_sd, use_energy, n_fourier)
        d['qpos'] = torch.stack([(d['x'] / half_w).clamp(-2, 2),
                                 (d['y'] / half_h).clamp(-2, 2)], dim=-1)
        d['yz'] = (d['targ'] - y_mu) / y_sd

    n_feat = tr['qf'].shape[-1]
    token_dim = tok_tr.shape[-1] if tok_tr is not None else 96
    if tok_tr is None:                      # no-token references: 1 zero token
        tr['tok'] = torch.zeros(tr['mask'].shape[0], 1, token_dim, device=device)
        ev['tok'] = torch.zeros(ev['mask'].shape[0], 1, token_dim, device=device)

    model = QueryProbe(token_dim, n_feat, capacity, use_tokens=use_tokens,
                       n_fourier=n_fourier, pos_bias=pos_bias).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    N = tr['mask'].shape[0]
    losses = []
    track = []

    def evaluate():
        model.eval()
        with torch.no_grad():
            preds, targs = [], []
            for i in range(0, ev['mask'].shape[0], 512):
                sl = slice(i, i + 512)
                p = model(ev['tok'][sl], ev['qf'][sl], ev['qpos'][sl])
                m = ev['mask'][sl]
                preds.append(p[m]); targs.append(ev['yz'][sl][m])
        model.train()
        return _r2(torch.cat(preds), torch.cat(targs))

    model.train()
    perm_pool = torch.randperm(N, generator=g)
    ptr = 0
    for s in range(steps):
        if ptr + batch > N:
            perm_pool = torch.randperm(N, generator=g)
            ptr = 0
        sel = perm_pool[ptr:ptr + batch].to(device)
        ptr += batch
        tok = tr['tok'][sel] if use_tokens else tr['tok'][sel] * 0
        pred = model(tok, tr['qf'][sel], tr['qpos'][sel])
        m = tr['mask'][sel]
        loss = F.mse_loss(pred[m], tr['yz'][sel][m])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        if (s + 1) % eval_every == 0 or s == steps - 1:
            track.append((s + 1, evaluate()))
            if verbose:
                r = track[-1][1]
                print(f'      step {s+1:5d}  loss {np.mean(losses[-100:]):.4f}  '
                      f'R2a {r[0]:+.4f}  R2T {r[1]:+.4f}', flush=True)

    r2_final = track[-1][1]
    r2_best = np.max(np.stack([t[1] for t in track]), axis=0)
    L = np.array(losses)
    d1 = L[int(0.6 * steps):int(0.8 * steps)].mean()
    d2 = L[int(0.8 * steps):].mean()
    rel = float(abs(d1 - d2) / (abs(d1) + 1e-12))
    return dict(
        r2_alpha=float(r2_final[0]), r2_T=float(r2_final[1]),
        r2_alpha_best=float(r2_best[0]), r2_T_best=float(r2_best[1]),
        train_loss_final=float(d2), loss_tail_rel_change=rel,
        gap_best_final=float(max(r2_best[0] - r2_final[0], r2_best[1] - r2_final[1])),
        n_params=int(n_par), track=track,
    )
