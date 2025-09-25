# %%
import pandas as pd
from tutils import *
from cascade import *

# %% Parameter Setup
dims, eranks, max_rank = [], [], []
samples_per_dim = 30
eps = 1e-8
nrank = 10
print("sim began")
method = 'skc'
order = 2
# %% Data Collection
for i in range(5, 13):
    print("dimension" + str(i),flush=True)
    for _ in range(samples_per_dim // 3):
        try:
            A = gen_TT_cascade_3d(
                Nscales=i, nrank=nrank, levels=None, seed=None,
                epsilon=1, method=method,
                eps=eps, boundary='periodic', order=order, field='velocity'
            )
            for j in range(3):
                mps = A[j].to_qtt(eps=eps)
                
                dims.append(i)
                eranks.append(erank(mps.R, mps.N))
                max_rank.append(max(mps.R))
        except:
            print(i, " sim failed", "method: ", method, "nrank: " , nrank, "order: ", order , flush=True)

# %% Create DataFrame
data = pd.DataFrame({
    'Dimension': dims,
    'ERank': eranks,
    'MaxRank': max_rank
})

# %% Save to CSV
data.to_csv("./data/rV5c2_1em8_p.csv", index=False)

print("done")