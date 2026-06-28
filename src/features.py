import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# --- TIME FEATURES ---
def time_features(df_stamp):
    d = pd.DatetimeIndex(pd.to_datetime(df_stamp['date'].values))
    f = [d.hour/23.0-0.5, d.dayofweek/6.0-0.5, (d.day-1)/30.0-0.5, (d.dayofyear-1)/365.0-0.5]
    return np.vstack(f).T

# --- PHA 1: HỒ SƠ LIÊN QUAN PHI TUYẾN (dCor) ---
def _how(ts):
    d = pd.DatetimeIndex(ts)
    return (d.dayofweek.values*24 + d.hour.values).astype(int)

def _deseason(v, sidx):
    nb = int(sidx.max())+1
    clim = np.array([v[sidx==b].mean() if (sidx==b).any() else 0. for b in range(nb)])
    return v - clim[sidx]

def _dc(d): 
    return d - d.mean(0,keepdims=True) - d.mean(1,keepdims=True) + d.mean()

def _dcor_fromB(x, B, dvy):
    A = _dc(np.abs(x[:,None]-x[None,:]))
    dcov2 = (A*B).mean()
    dvx = np.sqrt((A*A).mean())
    return float(np.sqrt(max(dcov2,0))/(np.sqrt(dvx*dvy)+1e-12))

def _mi(x, y):
    return float(mutual_info_regression(x[:,None], y, random_state=0)[0])

def build_exo_profile(clean_csv, train_end, target, exo_cols, period=24, lag_mult=1.0, measure='dcor', subsample=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(clean_csv, parse_dates=['time']).sort_values('time')
    df = df[df['time'] <= pd.Timestamp(train_end)].reset_index(drop=True)
    sidx = _how(df['time'])
    L = int(round(lag_mult*period))
    
    y = _deseason(df[target].values.astype(float), sidx)
    xr = {c: _deseason(df[c].values.astype(float), sidx) for c in exo_cols}
    
    n = len(y)
    m = min(subsample, n-L)
    idx = rng.choice(np.arange(L,n), m, replace=False)
    ys = y[idx]
    
    if measure == 'dcor':
        B = _dc(np.abs(ys[:,None]-ys[None,:]))
        dvy = np.sqrt((B*B).mean())
        
    lags = np.arange(L+1)
    R = np.zeros((len(exo_cols), L+1))
    
    for j, c in enumerate(exo_cols):
        for tau in lags:
            xs = xr[c][idx-tau]
            R[j,tau] = _dcor_fromB(xs, B, dvy) if measure=='dcor' else _mi(xs, ys)
            
    ts = R.argmax(1).astype(int)
    return dict(names=list(exo_cols), lags=lags, R=R, tau_star=ts, R_star=R[np.arange(len(exo_cols)), ts], measure=measure)

def col_indices(csv, target, exo_cols):
    cols = [c for c in pd.read_csv(csv, nrows=0).columns if c not in ('time','date')]
    order = [c for c in cols if c != target] + [target]
    return {c: order.index(c) for c in exo_cols}, order

# --- PHA 0: TRỌNG SỐ LOSS ---
def _autocov(x, maxlag):
    x = x - x.mean(); n = len(x)
    return np.array([ (x[:n-k]*x[k:]).sum()/n for k in range(maxlag+1) ])

def _ar_yule_walker(x, p):
    r = _autocov(x, p)
    R = np.array([[r[abs(i-j)] for j in range(p)] for i in range(p)])
    phi = np.linalg.solve(R + 1e-8*np.eye(p), r[1:p+1])
    sig2 = r[0] - phi @ r[1:p+1]
    return phi, max(sig2, 1e-8)

def _ma_psi(phi, H):
    p = len(phi); psi = np.zeros(H); psi[0] = 1.0
    for k in range(1, H):
        psi[k] = sum(phi[i]*psi[k-1-i] for i in range(min(p, k)))
    return psi

def _err_cov(psi, sig2, H):
    S = np.zeros((H, H))
    for a in range(H):
        for b in range(H):
            d = abs(a-b); m = min(a, b)
            S[a, b] = sig2 * sum(psi[k]*psi[k+d] for k in range(m+1))
    return S

def _shrink_to_cond(S, cond_max):
    tgt = (np.trace(S)/S.shape[0]) * np.eye(S.shape[0])
    for a in np.linspace(0, 1, 101):
        Sh = (1-a)*S + a*tgt
        if np.linalg.cond(Sh) <= cond_max:
            return Sh, float(a)
    return tgt, 1.0

def build_loss_prior(clean_csv, train_end, target, H, ar_order, cond_max):
    df = pd.read_csv(clean_csv, parse_dates=['time']).sort_values('time')
    df = df[df['time'] <= pd.Timestamp(train_end)].reset_index(drop=True)
    res = _deseason(df[target].values.astype(float), _how(df['time']))
    
    phi, sig2 = _ar_yule_walker(res, ar_order)
    psi = _ma_psi(phi, H)
    Sig = _err_cov(psi, sig2, H)
    
    var_h = np.diag(Sig).copy()
    w_invvar = (1.0/var_h)
    w_invvar = w_invvar / w_invvar.mean() 
    
    Sig_sh, alpha = _shrink_to_cond(Sig, cond_max)
    W = np.linalg.inv(Sig_sh)
    W = W/(np.trace(W)/H)
    
    return dict(w_invvar=w_invvar.astype(np.float32), W_invvar=W.astype(np.float32), 
                phi=phi, sig2=sig2, alpha=alpha, cond_raw=float(np.linalg.cond(Sig)), cond_sh=float(np.linalg.cond(Sig_sh)))
