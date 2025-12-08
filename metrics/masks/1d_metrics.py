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

jsonl_path = BASE_DIR / "metrics" / "data" / "masks" / "1d_funciton.jsonl"


# %% Configuration

def mean_std(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return x.item(), 0.0
    return x.mean(), x.std(ddof=1)  # sample std

def ften3d(I,a,b):
    X = ind_to_r_g(I,a=a,b=b,d=1,pdp=1)
    x = X[:,0]
    return f_h3( tn.tensor(x,dtype=tn.float64)).numpy()
fu = lambda I : ften3d(I,a=[0],b=[1])

#%% parameters------------------
tn.set_default_dtype(tn.float64)

p_coarse = 18
p_final = 28
reps = 10
n_e = 10
Nsamples = 2**16

scale = []
mode = []
times = []
error = []
s_error = []
rmax = []
comp = []
#%% simulaiton

t0 = time()
x0 = tn.linspace(0, 1, 2**p_coarse+1,requires_grad=True,dtype=tn.float64)[0:-1]
t1 = time()
y0 = f_h3(x0)
fc0 = tntt.TT(y0.detach(),[2]*(p_coarse),eps=1e-14)
t_coarse = time() - t0


for i in range(reps):

    for p in range(p_coarse+1,p_final+1):
        print('dimension ',p)
        t = time()
        ext = qtt_I_1D(fc0, p,f_h3( tn.tensor(1.0,requires_grad=False)), eps = 1e-14, order = 1, p_derivative = 0,boundary = 'linear')
        t_int = time() - t
        rmax_tti = np.max(ext.R)
        comp_tti = tt_comp(ext)
        t = time()
        dext = (2**p_coarse)*qtt_I_1D(fc0, p,f_h3( tn.tensor(1.0,requires_grad=False)), eps = 1e-14, order = 1, p_derivative = 1,boundary = 'linear')
        t_dint = time() - t
        rmax_dtti = np.max(dext.R)
        comp_dtti = tt_comp(ext)
        t = time()
        ddext = (4**p_coarse)*qtt_I_1D(fc0, p,f_h3( tn.tensor(1.0,requires_grad=False)), eps = 1e-14, order = 2, p_derivative = 2,boundary = 'linear')
        t_ddint = time() - t
        rmax_ddtti  =np.max(ddext.R)
        comp_ddtti = tt_comp(ext)

        ncoarse = [2]*p
        m         = None  # Number of calls to target function
        e         =  1e-14   # Desired accuracy
        nswp      = 5   # Sweep number
        r         = 20 + 5*(p-p_coarse)      # TT-rank of the initial tensor
        dr_min    = 2      # Cross parameter (minimum number of added rows)
        dr_max    = 5      # Cross parameter (maximum number of added rows)

        

        Yc = ten.rand(ncoarse, r)
        cache,info_coarse = {}, {}
        ten_cross = ten.cross(fu, Yc, m, e, nswp, dr_min=dr_min, dr_max=dr_max,log=True,cache=cache,m_cache_scale=5,info=info_coarse)
        ten_cross = ten.truncate(ten_cross, 1e-14) 
        rmax_cross = np.max(get_R(ten_cross))
        comp_cross = tt_comp(ten_cross)

        Dop = R_qtt( 2**p -1, p,dtype = tn.float64) - L_qtt( 1, p ,dtype = tn.float64) - 2*dmpo(0,0,p) + dmpo(0,1,p) +  2*dmpo(2**p -1,2**p -1,p) - dmpo(2**p -1,2**p -2,p)
        Dop = (Dop*2**p/2).round(1e-14)

        t = time()
        d_cross = (Dop @ tntt.TT([tn.from_numpy(c) for c in ten_cross])).round(1e-14)
        tdc = time()-t
        t = time()
        dd_cross = (Dop @ d_cross).round(1e-14)
        tddc = time()-t
        rmax_dcross = np.max(d_cross.R)
        comp_dcross = tt_comp(d_cross)
        rmax_ddcross = np.max(dd_cross.R)
        comp_ddcross = tt_comp(dd_cross)


        e_tti, e_tc = [],[]
        de_tti,dde_tti = [],[]
        d_ttc,dd_ttc = [],[]
        print('doing samples',flush=True)
        for e in range(n_e):

            samples = tt_mc_sample(ext.cores,Nsamples)
            ttv = ext.apply_mask(samples.T)
            w = np.array([ 2**(-i-1) for i in range(p)])
            weighted_sum = (w @ samples.numpy()).T
            rv = f_h3(tn.from_numpy(weighted_sum))
            e_tti.append((ttv-rv).norm().detach().numpy()/np.sqrt(Nsamples))

            samples = tt_mc_sample(dext.cores,Nsamples)
            ttv = dext.apply_mask(samples.T)
            w = np.array([ 2**(-i-1) for i in range(p)])
            x = tn.tensor( (w @ samples.numpy()).T, requires_grad=True, dtype=tn.float64)
            rv = f_h3(x)
            dy_dx = tn.autograd.grad(
                outputs=rv, inputs=x,
                grad_outputs=tn.ones_like(rv),
                create_graph=True
            )[0]
            de_tti.append((dy_dx-ttv).norm().detach().numpy()/np.sqrt(Nsamples))

            samples = tt_mc_sample(ddext.cores,Nsamples)
            ttv = ddext.apply_mask(samples.T)
            w = np.array([ 2**(-i-1) for i in range(p)])
            x = tn.tensor( (w @ samples.numpy()).T, requires_grad=True, dtype=tn.float64)
            rv = f_h3(x)
            dy_dx = tn.autograd.grad(
                outputs=rv, inputs=x,
                grad_outputs=tn.ones_like(rv),
                create_graph=True
            )[0]
            d2y_dx2 = tn.autograd.grad(
                outputs=dy_dx, inputs=x,
                grad_outputs=tn.ones_like(dy_dx),
                create_graph=False
            )[0]
            dde_tti.append((d2y_dx2-ttv).norm().detach().numpy()/np.sqrt(Nsamples))


            samples = tt_mc_sample([tn.from_numpy(c) for c in ten_cross],Nsamples)
            ttv = tn.from_numpy( ten.act_one.get_many(ten_cross,samples.T) )
            w = np.array([ 2**(-i-1) for i in range(p)])
            weighted_sum = (w @ samples.numpy()).T
            rv = f_h3(tn.from_numpy(weighted_sum))
            e_tc.append((ttv-rv).norm().detach().numpy()/np.sqrt(Nsamples))

            samples = tt_mc_sample(d_cross.cores,Nsamples)
            ttv = d_cross.apply_mask(samples.T)
            w = np.array([ 2**(-i-1) for i in range(p)])
            weighted_sum = (w @ samples.numpy()).T
            x = tn.tensor( (w @ samples.numpy()).T, requires_grad=True, dtype=tn.float64)
            rv = f_h3(x)
            dy_dx = tn.autograd.grad(
            outputs=rv, inputs=x,
            grad_outputs=tn.ones_like(rv),
            create_graph=True
            )[0]

            d_ttc.append((ttv-dy_dx).norm().detach().numpy()/np.sqrt(Nsamples))


            samples = tt_mc_sample(dd_cross.cores,Nsamples)
            ttv = dd_cross.apply_mask(samples.T)
            w = np.array([ 2**(-i-1) for i in range(p)])
            weighted_sum = (w @ samples.numpy()).T
            x = tn.tensor( (w @ samples.numpy()).T, requires_grad=True, dtype=tn.float64)
            rv = f_h3(x)
            dy_dx = tn.autograd.grad(
            outputs=rv, inputs=x,
            grad_outputs=tn.ones_like(rv),
            create_graph=True
            )[0]

            d2y_dx2 = tn.autograd.grad(
                outputs=dy_dx, inputs=x,
                grad_outputs=tn.ones_like(dy_dx),
                create_graph=False
            )[0]

            dd_ttc.append((ttv-d2y_dx2).norm().detach().numpy()/np.sqrt(Nsamples))

        etti,s_tti= mean_std(e_tti)
        ettc,s_ttc = mean_std(e_tc)
        detti,s_dtti = mean_std(de_tti)
        ddetti,s_ddtti = mean_std(dde_tti)
        dettc,s_dttc = mean_std(d_ttc)
        ddettc,s_ddttc = mean_std(dd_ttc)
        

        scale.extend([p,p,p,p,p,p])
        mode.extend(['cross','TTI','d cross','d TTTI','dd cross', 'dd TTI'])
        times.extend([info_coarse['t'],t_int+t_coarse,info_coarse['t'] + tdc,t_coarse + t_dint,info_coarse['t']+tddc,t_coarse+t_ddint])
        rmax.extend([rmax_cross,rmax_tti,rmax_dcross, rmax_dtti,rmax_ddcross,rmax_ddtti])
        error.extend([ettc,etti,dettc,detti,ddettc,ddetti])
        s_error.extend([s_ttc,s_tti,s_dttc,s_dtti,s_ddttc,s_ddtti])
        comp.extend([comp_cross,comp_tti,comp_dcross,comp_dtti,comp_ddcross,comp_ddtti])


        # Save the last 3 rows as JSON lines
        for j in range(-6, 0):
            raw_row = {
            'method': mode[j],
            'scale':  scale[j],
            'error':  error[j],
            'serror': s_error[j],
            'time':   times[j],
            'rmax':   rmax[j],
            'erank':  comp[j],
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
    
