import time
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_loader import build_loaders
from models import TimeXer, make_loss

def evaluate(model, loader, device, pred_len):
    model.eval()
    P, Tr = [], []
    with torch.no_grad():
        for x, y, xm, _ in loader:
            p = model(x.to(device), xm.to(device))
            P.append(p.cpu().numpy())
            Tr.append(y[:, -pred_len:, :].numpy())
            
    P = np.concatenate(P).flatten()
    Tr = np.concatenate(Tr).flatten()
    mse = mean_squared_error(Tr, P)
    return dict(MAE=mean_absolute_error(Tr, P), MSE=mse, RMSE=math.sqrt(mse))

def train_one(model, loaders, loss_name, seed, config, loss_prior):
    device = config['DEVICE']
    opt = Adam(model.parameters(), lr=config['LR'])
    sch = ReduceLROnPlateau(opt, 'min', factor=0.5, patience=2)
    crit = make_loss(loss_name, config['LOSS_WEIGHTS_PATH'], config['LOSS_WEIGHTS_KEY'], loss_prior, config['PRED_LEN']).to(device)
    
    best = float('inf'); bw = None; noimp = 0
    
    for ep in range(1, config['TRAIN_EPOCHS'] + 1):
        model.train()
        tl = []
        for x, y, xm, _ in loaders['train']:
            x, y, xm = x.to(device), y.to(device), xm.to(device)
            opt.zero_grad()
            loss = crit(model(x, xm), y[:, -config['PRED_LEN']:, :])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(loss.item())
            
        model.eval()
        vl = []
        with torch.no_grad():
            for x, y, xm, _ in loaders['val']:
                vl.append(crit(model(x.to(device), xm.to(device)), y[:, -config['PRED_LEN']:, :].to(device)).item())
                
        v = float(np.mean(vl))
        sch.step(v)
        
        if config.get('VERBOSE', False): 
            print(f'    ep{ep:2d} train={np.mean(tl):.4f} val={v:.4f}')
            
        if v < best: 
            best = v; bw = {k: val.clone() for k, val in model.state_dict().items()}; noimp = 0
        else:
            noimp += 1
            if noimp >= config['PATIENCE']: break
            
    if bw: model.load_state_dict(bw)
    return evaluate(model, loaders['test'], device, config['PRED_LEN'])

def run_ablation(profile, idx_map, n_vars, datasets, config, loss_prior):
    EXO_CFGS = {
        'baseline':  dict(use_xcorr_lag=False, use_xcorr_gate=False),
        'xcorr_lag': dict(use_xcorr_lag=True,  use_xcorr_gate=True)
    }
    rows = []
    t0 = time.time()
    device = config['DEVICE']
    
    for exo_name, exo_cfg in EXO_CFGS.items():
        for loss_name in config['LOSS_VARIANTS']:
            per = []
            for sd in config['SEEDS']:
                loaders = build_loaders(datasets, config['BATCH_SIZE'], sd)
                torch.manual_seed(sd)
                np.random.seed(sd)
                
                model = TimeXer(
                    profile=profile, idx_map=idx_map, n_vars=n_vars, 
                    seq_len=config['SEQ_LEN'], pred_len=config['PRED_LEN'], patch_len=config['PATCH_LEN'],
                    d_model=config['D_MODEL'], n_heads=config['N_HEADS'], e_layers=config['E_LAYERS'], 
                    d_ff=config['D_FF'], dropout=config['DROPOUT'], activation=config['ACTIVATION'], 
                    use_norm=config['USE_NORM'], gate_init=config['GATE_INIT'], **exo_cfg
                ).to(device)
                
                m = train_one(model, loaders, loss_name, sd, config, loss_prior)
                per.append(m)
                print(f'  [{exo_name:9s}|{loss_name:8s}|seed {sd}] '
                      f'MAE={m["MAE"]:.4f} MSE={m["MSE"]:.4f} RMSE={m["RMSE"]:.4f} '
                      f'({time.time()-t0:.0f}s)')
                
            agg = {k: (np.mean([p[k] for p in per]), np.std([p[k] for p in per])) for k in per[0]}
            rows.append(dict(exo=exo_name, loss=loss_name,
                             MAE=agg['MAE'][0], MAE_std=agg['MAE'][1], 
                             MSE=agg['MSE'][0], MSE_std=agg['MSE'][1],
                             RMSE=agg['RMSE'][0], RMSE_std=agg['RMSE'][1]))
            
    return pd.DataFrame(rows)
