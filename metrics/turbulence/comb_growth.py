# %%
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # two up
sys.path.insert(0, str(BASE_DIR))
import pandas as pd
from src.qtt_interpolation.utils import *
from metrics.turbulence.cascade import *
from metrics.turbulence.qttcascade import *
from src.qtt_interpolation.comb import *
import os

# %% Parameter Setup
dims, comp, max_rank = [], [], []
samples_per_dim = 30
eps = 1e-8
nrank    = 20
rrank = 200
epsilon = 1.0
var = 1
print("sim began, qtt growth")
method = 'cubic'
order = 2
# %% Data Collection
for i in range(5, 16):
    print("dimension" + str(i),flush=True)
    for _ in range(samples_per_dim // 3):
        try:
            A  = gen_comb_cascade_3d(
            Nscales=i, nrank=nrank,rrank=rrank, levels=None, seed=None,
            epsilon=epsilon, method='cubic', order=1,
            eps=eps, var=var,field='velocity')
            for j in range(3):
                    comb = A[j]
                    dims.append(i)
                    comp.append(comb_comp(comb,reduced=True))
                    max_rank.append(comb_max_rank(comb))
        except:
            print(i, " sim failed", "method: ", method, "nrank: " , nrank, "order: ", order , flush=True)

# %% Create DataFrame
data = pd.DataFrame({
    'Dimension': dims,
    'compression': comp,
    'MaxRank': max_rank
})

# %% Save to CSV
data.to_csv(str(BASE_DIR)+"/metrics/data/turbulence/rcg_200_10_e8_o2.csv", index=False)

print("done")