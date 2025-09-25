# %%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch as tn
import torchtt as tntt
import teneva as ten
import numpy as np
from time import time
import json

from src.qtt_interpolation.utils import *
from src.qtt_interpolation.int_tools import *
import pandas as pd
import seaborn as sns
from metrics.mutils import*

# %% Configuration

alpha = 1e2
R = 0.2
l = 1

ftn = lambda X,Y : symmetric_wing(X,Y,rad=R,new='wing1',alpha=alpha)
def ften(I):
    X = ind_to_r_ten(I,a=-l,b=l,d=2,pdp=1)
    x,y = X[:,1], X[:,0]
    return symmetric_wing( tn.tensor(x,dtype=tn.float64), tn.tensor(y,dtype=tn.float64),rad=R,new='wing1',alpha=alpha )
# %% Coarse grid 

p_coarse = 10
num_points = 2**p_coarse
pmax=17

# %% Data preparation
# prepare output directory & JSONL file
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
jsonl_path = os.path.join(BASE_DIR, 'metrics', 'data', 'wing_results_lr.jsonl')


# %% Gather Data for Cross

mode = []
scale  = []
error = []
T= []
rmax, ernk = [], []



reps = 10
for p in range(p_coarse+1, pmax+1):
    print(p,pmax, flush=True)
    pfinal = p
    #get statistics for cross
    for i in range(reps):

        # SVD coarse
        t = time()
        x = tn.linspace(-l, l, num_points + 1, dtype=tn.float64)[:-1]
        y = tn.linspace(-l, l, num_points + 1, dtype=tn.float64)[:-1]
        X, Y = tn.meshgrid(x, y, indexing='ij')

        mask_grid = ftn(X, Y)
        mask_svd = tntt.TT(zM(mask_grid,p_coarse),[2,2]*p_coarse,eps=1e-14)
        t_mask = time() - t


        ncoarse = [ 2,2]*p_coarse
        m         = None  # Number of calls to target function
        e         =  1e-6   # Desired accuracy
        nswp      = 4   # Sweep number
        r         = 20      # TT-rank of the initial tensor
        dr_min    = 2      # Cross parameter (minimum number of added rows)
        dr_max    = 5      # Cross parameter (maximum number of added rows)

        
        Yc = ten.rand(ncoarse, r)
        cache,info_coarse = {}, {}
        mask_ttc = ten.cross(ften, Yc, m, e, nswp, dr_min=dr_min, dr_max=dr_max,log=True,cache=cache,m_cache_scale=10,info=info_coarse)
        mask_ttc = ten.truncate(mask_ttc,1e-14)

        # do the interpolation
        t = time()
        mask_svd_i = qtt_skcubic2d_p(mask_svd, pfinal, eps=1e-10, order=1).round(1e-3)
        t_svd_i = time() - t
        mask_svd_i = [c.numpy() for c in mask_svd_i.cores]

        t = time()
        mask_ttc_i = qtt_skcubic2d_p( tntt.TT([tn.tensor(c,dtype=tn.float64) for c in mask_ttc]) ,pfinal, eps=1e-10, order=1).round(1e-3)
        t_ttc_i = time() - t
        mask_ttc_i = [c.numpy() for c in mask_ttc_i.cores]

        #cross over the fine grid

        n = [ 2,2]*pfinal   # Shape of the tensor
        m         = None  # Number of calls to target function
        e         =  1e-6   # Desired accuracy
        nswp      = 4   # Sweep number
        r         = 20 + (p-p_coarse)*2      # TT-rank of the initial tensor
        dr_min    = 2      # Cross parameter (minimum number of added rows)
        dr_max    = 5      # Cross parameter (maximum number of added rows)

        Y = ten.rand(n, r)
        info_fine,cache = {}, {}
        Y = ten.cross(ften, Y, m, e, nswp, dr_min=dr_min, dr_max=dr_max, cache=cache,m_cache_scale=5,log=True,info=info_fine)
        Yr = ten.truncate(Y, 1e-3) 

        ''' 
        #validate over 1e5 points
        m_tst = int(1e5)
        # Random multi-indices for the test points:
        I_tst = np.vstack([np.random.choice(k, m_tst) for k in n]).T
        y_tst = f(I_tst)
        norm = np.linalg.norm(y_tst)

        y_tt = ten.get_many(Yr, I_tst)
        e_tst = np.linalg.norm(y_tt - y_tst) / norm

        y_tti = ten.get_many(mask_ttc_i, I_tst)
        e_tstti = np.linalg.norm(y_tti - y_tst) / norm
    
        y_tti_svd = ten.get_many(mask_svd_i, I_tst)
        e_tstti_svd = np.linalg.norm(y_tti - y_tst) / norm
        print('errors', e_tst,e_tstti,e_tstti_svd)
        '''
        print('doing errors...',flush=True)
        N = 2**pfinal
        x = tn.linspace(-l, l, N + 1, dtype=tn.float64)[:-1]
        y = tn.linspace(-l, l, N + 1, dtype=tn.float64)[:-1]
        X, Y = tn.meshgrid(x, y, indexing='ij')
        mask_grid = ftn(X, Y)
        mask_full = zM(mask_grid,pfinal)
        norm = tn.linalg.norm(mask_grid)


        y_tt = tntt.TT( [tn.tensor(c,dtype=tn.float64) for c in Yr] ).full().reshape(N,N)
        e_tst = tn.linalg.norm(mask_full - y_tt) / norm

        y_tti = tntt.TT( [tn.tensor(c,dtype=tn.float64) for c in mask_ttc_i] ).full().reshape(N,N)
        e_tstti = tn.linalg.norm(mask_full - y_tti) / norm
    
        y_tti_svd = tntt.TT( [tn.tensor(c,dtype=tn.float64) for c in mask_svd_i] ).full().reshape(N,N)
        e_tstti_svd = tn.linalg.norm(mask_full - y_tti_svd) / norm

        # Save only the last 3 entries: 'cross', 'cross_int', 'svd_int'
        mode.extend(['cross', 'cross_int', 'svd_int'])
        scale.extend([p, p, p])
        error.extend([e_tst, e_tstti, e_tstti_svd])
        T.extend([info_fine['t'], t_ttc_i + info_coarse['t'], t_svd_i + t_mask])
        rmax.extend([
            np.max(ten.ranks(Yr)),
            np.max(ten.ranks(mask_ttc_i)),
            np.max(ten.ranks(mask_svd_i))
        ])
        ernk.extend([
            ten.erank(Yr),
            ten.erank(mask_ttc_i),
            ten.erank(mask_svd_i)
        ])

        print((i+1)/reps, end='\r', flush=True)

        # Save the last 3 rows as JSON lines
        for j in range(-3, 0):
            raw_row = {
            'method': mode[j],
            'scale':  scale[j],
            'error':  error[j],
            'time':   T[j],
            'rmax':   rmax[j],
            'erank':  ernk[j]
            }
            # Convert any Tensor or NumPy scalar to plain Python
            safe_row = {}
            for k, v in raw_row.items():
                if isinstance(v, tn.Tensor):
                    safe_row[k] = v.item() if v.numel() == 1 else v.tolist()
                elif isinstance(v, np.generic):
                    safe_row[k] = v.item()
                else:
                    safe_row[k] = v

            with open(jsonl_path, 'a') as f:
                f.write(json.dumps(safe_row) + '\n')
                f.flush()
                os.fsync(f.fileno())

        print(f"  rep {i+1}/{reps} saved", end='\r', flush=True)
        print('error',error[-3::],flush=True)
        print('rmax',rmax[-3::],flush=True)



# %% Collect interpolation data

df = pd.DataFrame({
    'method': mode,
    'scale': scale,
    'error': error,
    'time': T,
    'rmax': rmax,
    'erank': ernk
})

# %% Save results

df.to_csv('./metrics/data/wing_results.csv', index=False)

