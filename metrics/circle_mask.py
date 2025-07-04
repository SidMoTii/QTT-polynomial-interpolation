# %%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch as tn
import torchtt as tntt
import teneva as ten
import numpy as np
import matplotlib.pyplot as plt
from time import time

from src.qtt_interpolation.utils import *
from src.qtt_interpolation.int_tools import *
import pandas as pd
import seaborn as sns
from metrics.mutils import*

# %% Configuration

alpha = 100
ftn = lambda X,Y : circles(X,Y,alpha=alpha)
ften = lambda X: circle_ten(X,alpha=alpha)
# %% Coarse grid 

p_coarse = 10
num_points = 2**p_coarse


# %% Gather Data for Cross

mode = []
scale  = []
error = []
T= []
rmax, ernk = [], []


pmax=17
reps = 10
for p in range(p_coarse+1, pmax+1):
    print(p,pmax, flush=True)
    pfinal = p
    #get statistics for cross
    for i in range(reps):

        # SVD coarse
        t = time()
        x = tn.linspace(0, 1, num_points + 1, dtype=tn.float64)[:-1]
        y = tn.linspace(0, 1, num_points + 1, dtype=tn.float64)[:-1]
        X, Y = tn.meshgrid(x, y, indexing='ij')

        mask_grid = ftn(X, Y)
        mask_svd = tntt.TT(zM(mask_grid,p_coarse),[2,2]*p_coarse,eps=1e-14)
        t_mask = time() - t


        ncoarse = [ 2,2]*p_coarse
        m         = None  # Number of calls to target function
        e         =  1e-6   # Desired accuracy
        nswp      = 3   # Sweep number
        r         = 30      # TT-rank of the initial tensor
        dr_min    = 2      # Cross parameter (minimum number of added rows)
        dr_max    = 5      # Cross parameter (maximum number of added rows)

        f = lambda I: ften(ind_to_r_ten(I,a=0,b=1,d=2,pdp=1))
        
        Yc = ten.rand(ncoarse, r)
        cache,info_coarse = {}, {}
        mask_ttc = ten.cross(f, Yc, m, e, nswp, dr_min=dr_min, dr_max=dr_max,log=True,cache=cache,m_cache_scale=5,info=info_coarse)
        mask_ttc = ten.truncate(mask_ttc,1e-14)

        # do the interpolation
        t = time()
        mask_svd_i = qtt_skcubic2d_p(mask_svd, pfinal, eps=1e-10, order=1)
        t_svd_i = time() - t
        mask_svd_i = [c.numpy() for c in mask_svd_i.cores]

        t = time()
        mask_ttc_i = qtt_skcubic2d_p( tntt.TT([tn.tensor(c,dtype=tn.float64) for c in mask_ttc]) ,pfinal, eps=1e-10, order=1)
        t_ttc_i = time() - t
        mask_ttc_i = [c.numpy() for c in mask_ttc_i.cores]

        #cross over the fine grid

        n = [ 2,2]*pfinal   # Shape of the tensor
        m         = None  # Number of calls to target function
        e         =  1e-6   # Desired accuracy
        nswp      = 3   # Sweep number
        r         = 30 + (p-p_coarse)*5      # TT-rank of the initial tensor
        dr_min    = 2      # Cross parameter (minimum number of added rows)
        dr_max    = 5      # Cross parameter (maximum number of added rows)

        Y = ten.rand(n, r)
        info_fine,cache = {}, {}
        Y = ten.cross(f, Y, m, e, nswp, dr_min=dr_min, dr_max=dr_max, cache=cache,m_cache_scale=5,log=True,info=info_fine)
        Yr = ten.truncate(Y, 1e-14) 

        '''
        #validate over 1e5 points
        m_tst = int(1e6)
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
        '''
        print('doing errors...',flush=True)
        N = 2**pfinal
        x = tn.linspace(0, 1, N + 1, dtype=tn.float64)[:-1]
        y = tn.linspace(0, 1, N + 1, dtype=tn.float64)[:-1]
        X, Y = tn.meshgrid(x, y, indexing='ij')
        mask_grid = ftn(X, Y)
        mask_full = zM(mask_grid,p_coarse)
        norm = tn.linalg.norm(mask_full)

        y_tt = tntt.TT( [tn.tensor(c,dtype=tn.float64) for c in Yr] ).full().reshape(N,N)
        e_tst = np.linalg.norm(mask_full - y_tt) / norm

        y_tti = tntt.TT( [tn.tensor(c,dtype=tn.float64) for c in mask_ttc_i] ).full().reshape(N,N)
        e_tstti = np.linalg.norm(mask_full - y_tti) / norm
    
        y_tti_svd = tntt.TT( [tn.tensor(c,dtype=tn.float64) for c in mask_svd_i] ).full().reshape(N,N)
        e_tstti_svd = np.linalg.norm(mask_full - y_tti_svd) / norm

        mode.append('cross')
        scale.append(p)
        error.append(e_tst)
        T.append(info_fine['t'])
        rmax.append(np.max(ten.ranks(Yr)))
        ernk.append(ten.erank(Yr))

        mode.append('cross_int')
        scale.append(p)
        error.append(e_tstti)
        T.append(t_ttc_i + info_coarse['t'])
        rmax.append(np.max(ten.ranks(mask_ttc_i)))
        ernk.append(ten.erank(mask_ttc_i))

        mode.append('svd_int')
        scale.append(p)
        error.append(e_tstti_svd)
        T.append(t_svd_i + t_mask)
        rmax.append(np.max(ten.ranks(mask_svd_i)))
        ernk.append(ten.erank(mask_svd_i))

        print((i+1)/reps, end='\r',flush=True)



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

df.to_csv('./metrics/data/mask_results.csv', index=False)

