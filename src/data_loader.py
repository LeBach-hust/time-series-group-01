import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from features import time_features

class Dataset_Electricity(Dataset):
    def __init__(self, root, flag, target='load', seq_len=168, label_len=48, pred_len=24, scaler=None):
        self.flag = flag
        self.scaler = scaler
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.target = target
        
        fmap = {'train': 'train.csv', 'val': 'val.csv', 'test': 'test.csv'}
        df = pd.read_csv(os.path.join(root, fmap[flag]))
        
        if 'time' in df.columns: 
            df = df.rename(columns={'time': 'date'})
            
        for c in df.columns:
            if c != 'date':
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',',''), errors='coerce')
                
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df.sort_values('date').reset_index(drop=True)
        
        feat = [c for c in df.columns if c not in ['date', self.target]]
        df = df[['date'] + feat + [self.target]]
        dd = df.iloc[:, 1:]
        
        if flag == 'train':
            self.scaler = StandardScaler().fit(dd.values)
        assert self.scaler is not None, 'val/test cần scaler từ train'
        
        data = self.scaler.transform(dd.values)
        self.data_stamp = time_features(df[['date']])
        self.data_x = data
        self.data_y = data[:, -1:]

    def __len__(self): 
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, i):
        s, e = i, i + self.seq_len
        rb = e - self.label_len
        re = rb + self.label_len + self.pred_len
        return (torch.FloatTensor(self.data_x[s:e]),  
                torch.FloatTensor(self.data_y[rb:re]),
                torch.FloatTensor(self.data_stamp[s:e]), 
                torch.FloatTensor(self.data_stamp[rb:re]))

def build_datasets(data_root, target, seq_len, label_len, pred_len):
    tr = Dataset_Electricity(data_root, 'train', target, seq_len, label_len, pred_len)
    va = Dataset_Electricity(data_root, 'val', target, seq_len, label_len, pred_len, tr.scaler)
    te = Dataset_Electricity(data_root, 'test', target, seq_len, label_len, pred_len, tr.scaler)
    return tr, va, te

def build_loaders(dss, batch_size, seed):
    tr, va, te = dss
    g = torch.Generator()
    g.manual_seed(seed)
    return {
        'train': DataLoader(tr, batch_size, shuffle=True, drop_last=True, generator=g, num_workers=2),
        'val':   DataLoader(va, batch_size, shuffle=False, num_workers=2),
        'test':  DataLoader(te, batch_size, shuffle=False, num_workers=2)
    }
