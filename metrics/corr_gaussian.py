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
from metrics.mutils import*

# %% Configuration

''' 
# Create a random 2D correlation matrix
A = tn.randn(2, 2)
cov = A @ A.t()  
D = tn.diag(1 / tn.sqrt(tn.diag(cov)))
corr = D @ cov @ D  
'''
#Midcorrelated variables
corr = np.array([[1.0, 0.5],
                         [0.5, 1.0]]) 
# %% Coarse grid 

p_coarse = 10
num_points = 2**p_coarse
l = 4

# %% Data preparation
# prepare output directory & JSONL file
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
jsonl_path = os.path.join(BASE_DIR, 'metrics', 'data', 'corr_gaussian_results_2.jsonl')

# %% Gather Data for Cross

mode = []
scale  = []
error = []
T= []
rmax, ernk = [], []


reps = 10
pmax=17
for p in range(p_coarse+1, pmax+1):
    print(p,pmax,flush=True)
    pfinal = p
    #get statistics for cross
    for i in range(reps):
        # SVD coarse
        t = time()
        x = tn.linspace(-l, l, num_points + 1, dtype=tn.float64)[:-1]
        y = tn.linspace(-l, l, num_points + 1, dtype=tn.float64)[:-1]
        X, Y = tn.meshgrid(x, y, indexing='ij')

        pdf_grid = correlated_gaussian_pdf(X, Y, mean=[0, 0], corr=corr)
        gauss_svd = tntt.TT(zM(pdf_grid,p_coarse),[2,2]*p_coarse,eps=1e-12)
        t_gaussian = time() - t

        #get boundary tt
        t0 = time()
        bx_svd = tntt.TT( correlated_gaussian_pdf(tn.full_like(y,l), y, mean=[0, 0], corr=corr), [2]*p_coarse )
        by_svd = tntt.TT( correlated_gaussian_pdf(x, tn.full_like(x, l), mean=[0, 0], corr=corr), [2]*p_coarse )
        f11 = circles(*tn.ones(2)).item()
        t_boundaries = time() - t0

        fboundary = [bx_svd,by_svd,f11]

        #TTcross coarse
        ncoarse = [ 2,2]*p_coarse
        m         = None  # Number of calls to target function
        e         =  1e-10   # Desired accuracy
        nswp      = 4   # Sweep number
        r         = 20      # TT-rank of the initial tensor
        dr_min    = 2      # Cross parameter (minimum number of added rows)
        dr_max    = 5      # Cross parameter (maximum number of added rows)

        f = lambda I: correlated_gaussian_pdf_ten(ind_to_r_ten(I,a=-l,b=l,d=2,pdp=1), corr=corr)
        
        Yc = ten.rand(ncoarse, r)
        cache,info_coarse = {}, {}
        gauss_ttc = ten.cross(f, Yc, m, e, nswp, dr_min=dr_min, dr_max=dr_max,log=False,cache=cache,info=info_coarse)

        # do the interpolation
        t = time()
        gauss_svd_i = qtt_skcubic2d(gauss_svd, pfinal,fboundary, eps=1e-10, order=1)
        t_svd_i = time() - t
        gauss_svd_i = [c.numpy() for c in gauss_svd_i.cores]

        t = time()
        gauss_ttc_i = qtt_skcubic2d( tntt.TT([tn.tensor(c,dtype=tn.float64) for c in gauss_ttc]) ,pfinal, fboundary, eps=1e-12, order=1)
        t_ttc_i = time() - t
        gauss_ttc_i = [c.numpy() for c in gauss_ttc_i.cores]

        #cross over the fine grid

        n = [ 2,2]*pfinal   # Shape of the tensor
        m         = None  # Number of calls to target function
        e         =  1e-10   # Desired accuracy
        nswp      = 5  # Sweep number
        r         = 20 + (p-p_coarse)*5      # TT-rank of the initial tensor
        dr_min    = 2      # Cross parameter (minimum number of added rows)
        dr_max    = 5      # Cross parameter (maximum number of added rows)

        Y = ten.rand(n, r)
        info_fine,cache = {}, {}
        Y = ten.cross(f, Y, m, e, nswp, dr_min=dr_min, dr_max=dr_max, cache=cache,log=True,info=info_fine)
        Yr = ten.truncate(Y, 1e-14) 

        '''
        #validate over 1e5 points
        m_tst = int(1.E+6)
        # Random multi-indices for the test points:
        I_tst = np.vstack([np.random.choice(k, m_tst) for k in n]).T
        y_tst = f(I_tst)
        norm = np.linalg.norm(y_tst)

        y_tt = ten.get_many(Yr, I_tst)
        e_tst = np.linalg.norm(y_tt - y_tst) / norm

        y_tti = ten.get_many(gauss_ttc_i, I_tst)
        e_tstti = np.linalg.norm(y_tti - y_tst) / norm
    
        y_tti_svd = ten.get_many(gauss_svd_i, I_tst)
        e_tstti_svd = np.linalg.norm(y_tti - y_tst) / norm
        '''
        print('doing errors...',flush=True)
        N = 2**pfinal
        x = tn.linspace(-l, l, N + 1, dtype=tn.float64)[:-1]
        y = tn.linspace(-l, l, N + 1, dtype=tn.float64)[:-1]
        X, Y = tn.meshgrid(x, y, indexing='ij')
        mask_grid = correlated_gaussian_pdf(X, Y, mean=[0, 0], corr=corr)
        mask_full = zM(mask_grid,pfinal)
        norm = tn.linalg.norm(mask_full)

        y_tt = tntt.TT( [tn.tensor(c,dtype=tn.float64) for c in Yr] ).full().reshape(N,N)
        e_tst = tn.linalg.norm(mask_full - y_tt) / norm

        y_tti = tntt.TT( [tn.tensor(c,dtype=tn.float64) for c in gauss_ttc_i] ).full().reshape(N,N)
        e_tstti = tn.linalg.norm(mask_full - y_tti) / norm
    
        y_tti_svd = tntt.TT( [tn.tensor(c,dtype=tn.float64) for c in gauss_svd_i] ).full().reshape(N,N)
        e_tstti_svd = tn.linalg.norm(mask_full - y_tti_svd) / norm


        # Append results for all three methods
        mode.append('cross')
        scale.append(p)
        error.append(e_tst)
        T.append(info_fine['t'])
        rmax.append(np.max(ten.ranks(Yr)))
        ernk.append(ten.erank(Yr))

        mode.append('cross_int')
        scale.append(p)
        error.append(e_tstti)
        T.append(t_ttc_i + t_boundaries + info_coarse['t'])
        rmax.append(np.max(ten.ranks(gauss_ttc_i)))
        ernk.append(ten.erank(gauss_ttc_i))

        mode.append('svd_int')
        scale.append(p)
        error.append(e_tstti_svd)
        T.append(t_svd_i + t_gaussian + t_boundaries)
        rmax.append(np.max(ten.ranks(gauss_svd_i)))
        ernk.append(ten.erank(gauss_svd_i))

        print((i+1)/reps, end='\r', flush=True)
        print('error',error,flush=True)

        # Save the last three entries (cross, cross_int, svd_int) to JSONL
        for j in range(3):
            raw_row = {
            'method': mode[-3 + j],
            'scale':  scale[-3 + j],
            'error':  error[-3 + j],
            'time':   T[-3 + j],
            'rmax':   rmax[-3 + j],
            'erank':  ernk[-3 + j]
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

df.to_csv('./metrics/data/corr_gaussian_results_2.csv', index=False)

