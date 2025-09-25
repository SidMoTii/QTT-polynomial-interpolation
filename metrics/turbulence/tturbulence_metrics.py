# %%
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # two up
sys.path.insert(0, str(BASE_DIR))


#def load_tensors_pt(path, device="cpu"):
#    bundle = tn.load(path, map_location=device)
#    return bundle["tensors"], bundle.get("meta", {})

from metrics.turbulence.cascade import *
from metrics.turbulence.qttcascade import *
from src.qtt_interpolation.comb import *
import pandas as pd
from time import time


# %% Configuration
nscales  = 10
N        = 2**nscales
L        = 1 # 2 * np.pi
nrank    = 20
rrank = 250
epsilon  = 1.0
var      = 1
eps      = 1e-12
max_sep  = N//2
num_runs = 10
initial = 0


# %% Precompute radial bins
kx = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
ky = kx
kz = 2 * np.pi * np.fft.rfftfreq(N, d=L/N)
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
KX_shifted = np.fft.fftshift(KX)
KY_shifted = np.fft.fftshift(KY)
KZ_shifted = np.fft.fftshift(KZ)
k_mag = np.sqrt(KX_shifted**2 + KY_shifted**2 + KZ_shifted**2)

k_min     = 1.0
k_max     = k_mag.max() / np.sqrt(3)
num_bins  = N // 2
k_bins    = np.linspace(k_min, k_max, num_bins + 1)
k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
k_magnitude_flat = k_mag.ravel()

# Define bins in k (spherical shells)

k_min = k_magnitude_flat[k_magnitude_flat > 0].min()  # exclude zero to avoid log(0)
k_max = k_magnitude_flat.max()/1.7
bins = np.linspace(0.0, k_max, num_bins+1)  # num_bins bins
bin_centers = 0.5 * (bins[:-1] + bins[1:])
delta_k = bins[1:] - bins[:-1]

# %% Main loop
flat_records   = []
energy_records = []

for run_idx in range(initial,initial+num_runs):
    print("run", run_idx,flush=True)
    Ax, Ay, Az = gen_comb_cascade_3d(
        Nscales=nscales, nrank=nrank,rrank=rrank, levels=None, seed=None,
        epsilon=epsilon, method='cubic', order=2,
        eps=eps, var=var,field='velocity'
    )
    print('cascade done')
    #tntt.save(Ax, f'./data/mps/vel/8_qtt_8/vx_run{run_idx}.TT')
    #tntt.save(Ay, f'./data/mps/vel/8_qtt_8/vy_run{run_idx}.TT')
    #tntt.save(Az, f'./data/mps/vel/8_qtt_8/vz_run{run_idx}.TT')


    A_x = combqtt_full(Ax)
    A_y = combqtt_full(Ay)
    A_z = combqtt_full(Az)

    #u, v, w = compute_velocity_from_vector_potential_p(A_x.numpy(), A_y.numpy(), A_z.numpy(), L)
    #u, v, w = tn.from_numpy(u), tn.from_numpy(v), tn.from_numpy(w)
    u, v, w = A_x, A_y, A_z
    
    tn.save({"cores": [[c.detach().cpu() for c in branch] for branch in Ax]}, str(BASE_DIR)+ f"/metrics/data/turbulence/TN_turbulence/vx_10_{run_idx}.pt")
    tn.save({"cores": [[c.detach().cpu() for c in branch] for branch in Ay]}, str(BASE_DIR)+ f"/metrics/data/turbulence/TN_turbulence/vy_10_{run_idx}.pt")
    tn.save({"cores": [[c.detach().cpu() for c in branch] for branch in Az]}, str(BASE_DIR)+ f"/metrics/data/turbulence/TN_turbulence/vz_10_{run_idx}.pt")

    u_hat = np.fft.rfftn(u)
    v_hat = np.fft.rfftn(v)
    w_hat = np.fft.rfftn(w)
    E_hat = (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2) / (N**6) /2
    E_hat_shifted = np.fft.fftshift(E_hat)

    E_flat = E_hat.ravel()
    
    # 3. Get wavenumbers in 3D.
    e_flat = E_hat_shifted.flatten()


    # Bin the energy using np.histogram with weights:
    bin_energy, _ = np.histogram(k_magnitude_flat, bins=bins, weights=e_flat)
    # Normalize by the bin width to obtain an estimate of the energy density E(k)
    E_k = bin_energy / delta_k

    nonzero = (E_k > 0)  & (bin_centers > 0)

    for kc, Ek in zip(k_centers[nonzero], E_k[nonzero]):
        energy_records.append({
            'run_idx': run_idx,
            'k_center': kc,
            'E_k': Ek
        })

    seps, flats = compute_flatness_p(u.numpy(), max_sep, L)
    for sep, flat in zip(seps, flats):
        flat_records.append({
            'run_idx': run_idx,
            'separation': sep,
            'flatness': flat
        })

df_flatness = pd.DataFrame(flat_records)
df_energy = pd.DataFrame(energy_records)

df_flatness.to_csv(str(BASE_DIR)+ '/metrics/data/turbulence/flatness_comb10_e12.csv', index=False)
df_energy.to_csv(str(BASE_DIR)+ '/metrics/data/turbulence/energy_comb10_e12.csv', index=False)