# %%
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # two up
sys.path.insert(0, str(BASE_DIR))
from src.qtt_interpolation.utils import *
from src.qtt_interpolation.int_tools import *
from src.qtt_interpolation.comb import *
from metrics.mutils import *
import teneva as ten
import numpy as np
from time import time
import json
import os

# %% Data preparation
# prepare output directory & JSONL file

jsonl_path = BASE_DIR / "metrics" / "data" / "3d_wing_c10_test.jsonl"


# %% Configuration

def mean_std(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return x.item(), 0.0
    return x.mean(), x.std(ddof=1)  # sample std

def ften3d(I,a,b):
    X = ind_to_r_g(I,a=a,b=b,d=3,pdp=1)
    z,y,x = X[:,0], X[:,1], X[:,2]
    return wing_mask_pc( tn.tensor(x,dtype=tn.float64), tn.tensor(y,dtype=tn.float64),tn.tensor(z,dtype=tn.float64))

f_wing = lambda I : ften3d(I,a=[-zlim,-2,-2],b=[zlim,2,2])
#------------------

p_coarse = 10
print('initial dim ',p_coarse)
zlim     = 1/16
num_points = 2**p_coarse
pmax = 15
reps = 1
mode = []
scale  = []
error = []
serror = []
T= []
rmax, ernk = [], []
n_e = 7


# SVD coarse interleaved
print('doing ttsvd', flush=True)
t = time()
x = np.linspace(-2,  2, 2**p_coarse, endpoint=False)  # ← extended from [-1,1]
y   = np.linspace(-2,  2, 2**p_coarse, endpoint=False)
z   = np.linspace(-zlim, zlim, 2**p_coarse, endpoint=False)
X, Y, Z = np.meshgrid(x, y,z, indexing='ij')

mask_grid = wing_mask(X,Y,Z)
mask_svd = tntt.TT(zM3(mask_grid,p_coarse),[2,2,2]*p_coarse,eps=1e-12)
t_mask_svd = time() - t
print('TTsvd finished',flush=True)
print('doing comb',flush=True)
t = time()
x = np.linspace(-2,  2, 2**p_coarse, endpoint=False)  # ← extended from [-1,1]
y   = np.linspace(-2,  2, 2**p_coarse, endpoint=False)
z   = np.linspace(-zlim, zlim, 2**p_coarse, endpoint=False)
X, Y, Z = np.meshgrid(x, y,z, indexing='ij')
mask_grid = wing_mask(X,Y,Z)
comb_qtt = qttc_from_TT(mask_grid,1e-12)
t_mask_c = time() - t
print('TTcomb finished',flush=True)

for p in range(p_coarse+1, pmax+1):
    print(p,pmax, flush=True)

    #get statistics for cross
    for i in range(reps):

        # TT cross interleaved
        ncoarse = [ 2,2,2]*p_coarse
        m         = None  # Number of calls to target function
        e         =  1e-5   # Desired accuracy
        nswp      = 3   # Sweep number
        r         = 100      # TT-rank of the initial tensor
        dr_min    = 2      # Cross parameter (minimum number of added rows)
        dr_max    = 5      # Cross parameter (maximum number of added rows)

        
        Yc = ten.rand(ncoarse, r)
        cache,info_coarse = {}, {}
        mask_ttc = ten.cross(f_wing, Yc, m, e, nswp, dr_min=dr_min, dr_max=dr_max,log=True,cache=cache,m_cache_scale=5,info=info_coarse)
        mask_ttc = ten.truncate(mask_ttc,1e-12)
        print('coarse TTC finished',flush=True)

        # do the interpolation
        t = time()
        mask_svd_i = qtt_skcubic3d_p(mask_svd, p, eps = 1e-3, order=2)
        t_svd_i = time() - t
        #mask_svd_i = [c.numpy() for c in mask_svd_i.cores]

        t = time()
        mask_ttc_i = qtt_skcubic3d_p( tntt.TT([tn.tensor(c,dtype=tn.float64) for c in mask_ttc]) ,p, eps=1e-3, order=2)
        t_ttc_i = time() - t
        #mask_ttc_i = [c.numpy() for c in mask_ttc_i.cores]

        #cross over the fine grid

        n = [ 2,2,2]*p   # Shape of the tensor
        m         = None  # Number of calls to target function
        e         =  1e-4   # Desired accuracy
        nswp      = 4   # Sweep number
        r         = 100 + (p-p_coarse)*20      # TT-rank of the initial tensor
        dr_min    = 2      # Cross parameter (minimum number of added rows)
        dr_max    = 5      # Cross parameter (maximum number of added rows)

        Y = ten.rand(n, r)
        info_fine,cache = {}, {}
        Y = ten.cross(f_wing, Y, m, e, nswp, dr_min=dr_min, dr_max=dr_max, cache=cache,m_cache_scale=5,log=True,info=info_fine)
        Yr = ten.truncate(Y, 1e-3) 
        print('ttc finished',flush=True)

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
        #N = 2**p
        #x = np.linspace(-2,  2, 2**p, endpoint=False)  # ← extended from [-1,1]
        #y   = np.linspace(-2,  2, 2**p, endpoint=False)
        #z   = np.linspace(-zlim, zlim, 2**p, endpoint=False)
        #X, Y, Z = np.meshgrid(x, y,z, indexing='ij')
        #mask_grid = wing_mask(X,Y,Z)
        #mask_full = zM3(mask_grid,p)
        #norm = tn.linalg.norm(mask_grid)
        #y_tt = tntt.TT( [tn.tensor(c,dtype=tn.float64) for c in Yr] ).full().reshape(N,N,N)
        #e_tst = tn.linalg.norm(mask_full - y_tt) / norm

        #y_tti = mask_ttc_i.full().reshape(N,N,N)
        #e_tstti = tn.linalg.norm(mask_full - y_tti) / norm
    
        #y_tti_svd = mask_svd_i.full().reshape(N,N,N)
        #e_tstti_svd = tn.linalg.norm(mask_full - y_tti_svd) / norm

        Nsamples = int(1e4)
        a = [-zlim,-2,-2]
        b = [zlim,2,2]
        e_svd, e_tti, e_ttc = [], [], []
        for k in range(n_e):
            samp = tt_mc_sample(mask_svd_i.cores, Nsamples, rng = None, return_joint=False)
            vals = mask_svd.apply_mask(samp.T)
            x = (b[0]-a[0])*(samp.T[:,0::3] * tn.tensor([2**(-i-1) for i in range(p)])).sum(1) + a[0]
            y = (b[1]-a[1])*(samp.T[:,1::3] * tn.tensor([2**(-1-i) for i in range(p)])).sum(1) + a[1]
            z = (b[2]-a[2])*(samp.T[:,2::3] * tn.tensor([2**(-1-i) for i in range(p)])).sum(1) + a[2]
            e = tn.from_numpy(wing_mask_pc(z,y,x))-vals
            e_tstti_svd = (e*e).sum()/Nsamples
   
            samp = tt_mc_sample(mask_ttc_i.cores, Nsamples, rng = None, return_joint=False)
            vals = mask_ttc_i.apply_mask(samp.T)
            x = (b[0]-a[0])*(samp.T[:,0::3] * tn.tensor([2**(-i-1) for i in range(p)])).sum(1) + a[0]
            y = (b[1]-a[1])*(samp.T[:,1::3] * tn.tensor([2**(-1-i) for i in range(p)])).sum(1) + a[1]
            z = (b[2]-a[2])*(samp.T[:,2::3] * tn.tensor([2**(-1-i) for i in range(p)])).sum(1) + a[2]
            e = tn.from_numpy(wing_mask_pc(z,y,x))-vals
            e_tstti = (e*e).sum()/Nsamples

            samp = tt_mc_sample([tn.from_numpy(c) for c in Yr], Nsamples, rng = None, return_joint=False)
            vals = ten.act_one.get_many(Yr,samp.T)
            x = (b[0]-a[0])*(samp.T[:,0::3] * tn.tensor([2**(-i-1) for i in range(p)])).sum(1) + a[0]
            y = (b[1]-a[1])*(samp.T[:,1::3] * tn.tensor([2**(-1-i) for i in range(p)])).sum(1) + a[1]
            z = (b[2]-a[2])*(samp.T[:,2::3] * tn.tensor([2**(-1-i) for i in range(p)])).sum(1) + a[2]
            e = tn.from_numpy(wing_mask_pc(z,y,x))-vals
            e_tst = (e*e).sum()/Nsamples

            print(k+1, 'samples dome', flush=True)

            e_svd.append(e_tstti_svd.item())
            e_tti.append(e_tstti.item())
            e_ttc.append(e_tst.item())

        e_tstti_svd, s_tstti_svd = mean_std(e_svd)
        e_tstti, s_tstti = mean_std(e_tti)
        e_tst, s_tst = mean_std(e_ttc)
        print('errors', e_tst,e_tstti,e_tstti_svd)


        # Save only the last 3 entries: 'cross', 'cross_int', 'svd_int'
        mode.extend(['cross', 'cross_int', 'svd_int'])
        scale.extend([p, p, p])
        error.extend([e_tst, e_tstti, e_tstti_svd])
        serror.extend([s_tst, s_tstti, s_tstti_svd])
        T.extend([info_fine['t'], t_ttc_i + info_coarse['t'], t_svd_i + t_mask_svd])
        rmax.extend([
            np.max(ten.ranks(Yr)),
            np.max(mask_ttc_i.R),
            np.max(mask_svd_i.R)
        ])
        ernk.extend([
            tt_comp(Yr),
            tt_comp(mask_ttc_i),
            tt_comp(mask_svd_i)
        ])


        ## Do comb
        print('interpolating comb',flush=True)
        t = time()
        comb = comb_interpolate_p(comb_qtt,p, eps =  1e-3,order = 2)
        ti_comb = time()-t
        print('comb done',flush=True)
        #x = np.linspace(-2,  2, 2**p, endpoint=False)  # ← extended from [-1,1]
        #y   = np.linspace(-2,  2, 2**p, endpoint=False)
        #z   = np.linspace(-zlim, zlim, 2**p, endpoint=False)
        #X, Y, Z = np.meshgrid(x, y,z, indexing='ij')
        #mask_grid = wing_mask(X,Y,Z)
        #full_comb = combqtt_full(comb)
        #err = tn.linalg.norm(mask_grid - full_comb)/tn.linalg.norm(mask_grid)

        a = [-2,-2,-zlim]
        b = [2,2,zlim]
        e_comb = []
        for k in range(n_e):
            samp = sample_from_comb_cw(comb,Nsamples)
            samples = samp.reshape(3,p,-1)
            x = (b[0]-a[0])*((samples[0].T) * tn.tensor([2**(-i-1) for i in range(p)])).sum(1) + a[0]
            y = (b[1]-a[1])*((samples[1].T) * tn.tensor([2**(-1-i) for i in range(p)])).sum(1) + a[1]
            z = (b[2]-a[2])*((samples[2].T) * tn.tensor([2**(-1-i) for i in range(p)])).sum(1) + a[2]
            print(k,' sampling comb',flush=True)
            vals = apply_mask_c(comb,samp,sample_type='column')
            e = tn.from_numpy(wing_mask_pc(x,y,z))-vals
            err = (e*e).sum()/Nsamples
            e_comb.append(err.item())

        err,s_comb = mean_std(e_comb)
        print(err, s_comb,flush=True)
        mode.append('qtt-tucker')
        T.append(t_mask_c+ti_comb)
        rmax.append(comb_max_rank(comb))
        scale.append(p)
        ernk.append(comb_comp(comb))
        error.append(err)
        serror.append(s_comb)

        # Save the last 3 rows as JSON lines
        for j in range(-4, 0):
            raw_row = {
            'method': mode[j],
            'scale':  scale[j],
            'error':  error[j],
            'serror': serror[j],
            'time':   T[j],
            'rmax':   rmax[j],
            'erank':  ernk[j],
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