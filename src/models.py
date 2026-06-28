import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- MODULES HỖ TRỢ ---
def _logit(p, eps=1e-4): 
    p = p.clamp(eps, 1-eps)
    return torch.log(p/(1-p))

class XCorrLagExogenous(nn.Module):
    def __init__(self, exo_idx, tau_star, R, n_vars, use_lag=True, use_gate=True, learnable_gate=True, gate_init='prior_centered', center_spread=2.0):
        super().__init__()
        self.n_vars, self.use_lag, self.use_gate = n_vars, use_lag, use_gate
        self.register_buffer('exo_idx', torch.as_tensor(exo_idx, dtype=torch.long))
        self.register_buffer('tau_star', torch.as_tensor(tau_star, dtype=torch.long))
        R = torch.as_tensor(R, dtype=torch.float32)
        self.register_buffer('R', R)
        
        if gate_init == 'relevance': init = _logit(R)
        elif gate_init == 'ones': init = _logit(torch.ones_like(R)*0.999)
        elif gate_init == 'prior_centered':
            z = (R-R.mean())/(R.std()+1e-6)
            g0 = (0.5+0.5*torch.tanh(z/center_spread)).clamp(0.1,0.9)
            init = _logit(g0)
        else: raise ValueError(gate_init)
        
        if use_gate:
            if learnable_gate: self.gate_logit = nn.Parameter(init.clone())
            else: self.register_buffer('gate_logit', init.clone())

    @staticmethod
    def from_profile(p, idx_map, n_vars, **kw):
        return XCorrLagExogenous([idx_map[c] for c in p['names']], [int(t) for t in p['tau_star']], [float(r) for r in p['R_star']], n_vars, **kw)

    def gates(self): 
        return torch.sigmoid(self.gate_logit) if self.use_gate else None

    def forward(self, x):
        B, T, V = x.shape
        cols = [x[:,:,v] for v in range(V)]
        if self.use_lag:
            for k, c in enumerate(self.exo_idx.tolist()):
                tau = int(self.tau_star[k])
                if tau > 0:
                    col = cols[c]
                    cols[c] = torch.cat([col[:,:1].expand(B,tau), col[:,:T-tau]], 1)
        out = torch.stack(cols, -1)
        if self.use_gate:
            g = torch.sigmoid(self.gate_logit)
            gf = out.new_ones(V).scatter(0, self.exo_idx, g)
            out = out * gf.view(1,1,V)
        return out

# --- TIMEXER BLOCKS ---
class PositionalEmbedding(nn.Module):
    def __init__(self, d, ml=5000):
        super().__init__()
        pe = torch.zeros(ml, d).float()
        pos = torch.arange(0, ml).float().unsqueeze(1)
        dt = (torch.arange(0, d, 2).float()*-(math.log(10000.)/d)).exp()
        pe[:, 0::2] = torch.sin(pos*dt)
        pe[:, 1::2] = torch.cos(pos*dt)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return self.pe[:, :x.size(1)]

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c, d, dr=0.1): 
        super().__init__()
        self.ve = nn.Linear(c, d)
        self.dr = nn.Dropout(dr)
    def forward(self, x, xm):
        if xm is not None: x = torch.cat([x, xm], -1)
        return self.dr(self.ve(x.permute(0, 2, 1)))

class EnEmbedding(nn.Module):
    def __init__(self, nv, d, pl, dr):
        super().__init__()
        self.pl = pl
        self.ve = nn.Linear(pl, d, bias=False)
        self.glb = nn.Parameter(torch.randn(1, nv, 1, d))
        self.pe = PositionalEmbedding(d)
        self.dr = nn.Dropout(dr)
    def forward(self, x):
        B, nv = x.shape[0], x.shape[1]
        glb = self.glb.repeat(B, 1, 1, 1)
        x = x.unfold(-1, self.pl, self.pl)
        x = x.reshape(B*nv, x.shape[2], x.shape[3])
        x = self.ve(x) + self.pe(x)
        x = x.reshape(B, nv, x.shape[-2], x.shape[-1])
        x = torch.cat([x, glb], 2)
        x = x.reshape(B*nv, x.shape[2], x.shape[3])
        return self.dr(x), nv

class FullAttention(nn.Module):
    def __init__(self, ad=0.1): 
        super().__init__()
        self.dr = nn.Dropout(ad)
    def forward(self, q, k, v, m=None):
        B, L, H, E = q.shape
        sc = 1. / math.sqrt(E)
        s = torch.einsum('blhe,bshe->bhls', q, k) * sc
        A = self.dr(torch.softmax(s, -1))
        return torch.einsum('bhls,bshd->blhd', A, v).contiguous(), A

class AttentionLayer(nn.Module):
    def __init__(self, att, d, h):
        super().__init__()
        dk = d // h
        self.att = att
        self.q = nn.Linear(d, dk*h); self.k = nn.Linear(d, dk*h); self.v = nn.Linear(d, dk*h); self.o = nn.Linear(dk*h, d)
        self.h = h
    def forward(self, q, k, v, m=None):
        B, L, _ = q.shape; _, S, _ = k.shape; H = self.h
        Q = self.q(q).view(B, L, H, -1); K = self.k(k).view(B, S, H, -1); V = self.v(v).view(B, S, H, -1)
        o, a = self.att(Q, K, V, m)
        return self.o(o.view(B, L, -1)), a

class EncoderLayer(nn.Module):
    def __init__(self, sa, ca, d, dff=None, dr=0.1, act='gelu'):
        super().__init__()
        dff = dff or 4*d
        self.sa = sa; self.ca = ca
        self.c1 = nn.Conv1d(d, dff, 1); self.c2 = nn.Conv1d(dff, d, 1)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d); self.n3 = nn.LayerNorm(d)
        self.dr = nn.Dropout(dr)
        self.act = F.gelu if act == 'gelu' else F.relu
    def forward(self, x, cross):
        nx, _ = self.sa(x, x, x)
        x = self.n1(x + self.dr(nx))
        nx, _ = self.ca(x, cross, cross)
        y = self.n2(x + self.dr(nx))
        y2 = self.dr(self.act(self.c1(y.transpose(-1, 1))))
        y2 = self.dr(self.c2(y2).transpose(-1, 1))
        return self.n3(y + y2)

class Encoder(nn.Module):
    def __init__(self, layers, norm=None): 
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm
    def forward(self, x, cross):
        for l in self.layers: x = l(x, cross)
        return self.norm(x) if self.norm is not None else x

class FlattenHead(nn.Module):
    def __init__(self, nv, nf, tw, dr=0):
        super().__init__()
        self.fl = nn.Flatten(start_dim=-2)
        self.lin = nn.Linear(nf, tw)
        self.dr = nn.Dropout(dr)
    def forward(self, x): 
        return self.dr(self.lin(self.fl(x)))

# --- TIMEXER MODEL ---
class TimeXer(nn.Module):
    def __init__(self, profile, idx_map, n_vars, seq_len, pred_len, patch_len, d_model, n_heads, e_layers, d_ff, dropout, activation, use_norm, use_xcorr_lag, use_xcorr_gate, gate_init):
        super().__init__()
        self.n_vars = n_vars
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_norm = use_norm
        
        self.xcorr_exo = XCorrLagExogenous.from_profile(profile, idx_map, n_vars, use_lag=use_xcorr_lag, use_gate=use_xcorr_gate, gate_init=gate_init)
        self.en_embedding = EnEmbedding(1, d_model, patch_len, dropout)
        self.ex_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)
        
        npn = seq_len // patch_len + 1
        self.encoder = Encoder([
            EncoderLayer(AttentionLayer(FullAttention(dropout), d_model, n_heads), AttentionLayer(FullAttention(dropout), d_model, n_heads), d_model, d_ff, dropout, activation) 
            for _ in range(e_layers)
        ], nn.LayerNorm(d_model))
        
        self.head = FlattenHead(1, npn * d_model, pred_len, dropout)

    def forward(self, x_enc, x_mark_enc):
        B = x_enc.shape[0]
        if self.use_norm:
            mean = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - mean
            std = torch.sqrt(torch.var(x_enc, 1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc = x_enc / std
            
        x_exo = self.xcorr_exo(x_enc)
        ex_out = self.ex_embedding(x_exo, x_mark_enc)
        x_endo = x_enc[:, :, -1:].permute(0, 2, 1)
        en_out, nv = self.en_embedding(x_endo)
        
        enc = self.encoder(en_out, ex_out.repeat_interleave(nv, 0)).reshape(B, nv, -1, ex_out.shape[-1])
        out = self.head(enc[:, 0, :, :])
        
        if self.use_norm: 
            out = out * std[:, 0, -1:] + mean[:, 0, -1:]
        return out.unsqueeze(-1)

# --- LOSS FUNCTIONS ---
class WeightedMSELoss(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.register_buffer('w', w / w.mean())
    def forward(self, pred, true):
        return ((pred - true).pow(2).squeeze(-1) * self.w.view(1, -1)).mean()

class CorrStructLoss(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.register_buffer('W', W)
    def forward(self, pred, true):
        e = (pred - true).squeeze(-1)
        return torch.einsum('bi,ij,bj->b', e, self.W, e).mean() / e.shape[1]

def _load_prior(key, loss_weights_path, loss_prior, pred_len):
    if os.path.exists(loss_weights_path):
        arr = np.load(loss_weights_path)
        if key in arr: return torch.tensor(arr[key], dtype=torch.float32)
    return torch.tensor(loss_prior[key], dtype=torch.float32)

def make_loss(name, loss_weights_path, loss_weights_key, loss_prior, pred_len):
    if name == 'mse': return nn.MSELoss()
    if name == 'weighted':
        w = _load_prior(loss_weights_key, loss_weights_path, loss_prior, pred_len)
        return WeightedMSELoss(w)
    if name == 'corr':
        W = _load_prior('W_invvar', loss_weights_path, loss_prior, pred_len)
        return CorrStructLoss(W)
    raise ValueError(name)
