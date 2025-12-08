import torch as tn
import numpy as np

tn.set_default_dtype(tn.float64)

### 2D auxiliary functions

def ind_to_r(I, a=0, b=1, d=1, pdp=1):
    """
    This function maps a multi-digit index to a real number in [a, b)^d.
    Each entry in I[k] is an integer in 0,...,2**pdp-1 and is expanded to its binary representation.
    The resulting binary digits are concatenated to form a long vector of 0's and 1's for each sample.
    Returns a list of size d.
    """
    # I: shape (num_samples, d), each entry in 0,...,2**pdp-1
    num_samples = I.shape[0]
    l = I.shape[1] * pdp  # total number of binary digits per sample
    # Convert each entry to binary and concatenate
    I_bin = ((I.unsqueeze(-1) >> tn.arange(pdp-1, -1, -1)) & 1).reshape(num_samples, -1)
    powers = tn.tensor([2**(-i) for i in range(1, l//d+1)],dtype=tn.float64)
    result = []
    for j in range(d):
        idx = tn.arange(j, l, d)
        group = I_bin[:, idx]
        wI = group * powers
        x = (b - a) * wI.sum(axis=1) + a
        result.append(x)
    return result

def circle(x,y,r=0.125):
    """
    This function creates a mask for the points in the grid.
    It returns a boolean tensor where True indicates that the point is inside the unit square.
    """
    
    return tn.where((x-0.5)**2 + (y-0.5)**2 <= r**2, 1, 0)

def correlated_gaussian_pdf(x, y, mean=None, corr=None):
    """
    Compute the PDF value of a 2D Gaussian at point (x, y) with given mean and correlation matrix.
    Args:
        x (float or tn.Tensor): x-coordinate(s)
        y (float or tn.Tensor): y-coordinate(s)
        mean (list or None): Mean of the distribution. If None, uses [0, 0].
        corr (2x2 tn.Tensor or None): Correlation matrix. If None, uses identity.
    Returns:
        tn.Tensor: PDF value(s) at (x, y)
    """
    if mean is None:
        mean = tn.zeros(2, dtype=tn.float64)
    if corr is None:
        corr = tn.eye(2,dtype=tn.float64)
    elif type(corr) is not tn.Tensor:
        corr = tn.tensor(corr,dtype=tn.float64)
    pos = tn.stack([x, y], dim=-1)
    mean = tn.tensor(mean, dtype=pos.dtype, device=pos.device)
    corr_inv = tn.linalg.inv(corr)
    det_corr = tn.linalg.det(corr)
    diff = pos - mean
    exponent = -0.5 * tn.sum(diff @ corr_inv * diff, dim=-1)
    norm = 1.0 / (2 * tn.pi * tn.sqrt(det_corr))
    return norm * tn.exp(exponent)

def circles(x,y,alpha=100,r=0.125):
    """
    This function creates a mask for the points in the grid.
    It returns a boolean tensor where True indicates that the point is inside the unit square.
    """
    return tn.where((x-0.5)**2 + (y-0.5)**2 <= r**2, 1, tn.exp(-alpha*( (x-0.5)**2 + (y-0.5)**2 - r**2) ))

def ind_to_r_ten(I, a=0, b=1, d=1, pdp=1):
    """
    This function maps a multi-digit index to a real number in [a, b)^d.
    Each entry in I[k] is an integer in 0,...,2**pdp-1 and is expanded to its binary representation.
    The resulting binary digits are concatenated to form a long vector of 0's and 1's for each sample.
    Returns a numpy array of shape (num_samples, d).
    """
    I = np.asarray(I)
    num_samples = I.shape[0]
    l = I.shape[1]*pdp  # total number of binary digits per sample
    # Convert each entry to binary and concatenate
    I_bin = ((I[..., None] >> np.arange(pdp-1, -1, -1)) & 1).reshape(num_samples, -1)
    powers = np.array([2**(-i) for i in range(1, l//d+1)])
    result = np.empty((num_samples, d))
    for j in range(d):
        idx = np.arange(j, l, d)
        group = I_bin[:, idx]
        wI = group * powers
        x = (b - a) * wI.sum(axis=1) + a
        result[:, j] = x
    return result

def circle_ten(X,alpha=100,r=0.125):
    """
    This function creates a mask for the points in the grid.
    It returns a boolean tensor where True indicates that the point is inside the unit square.
    """
    x, y = X[:, 0], X[:, 1]
    return np.where((x-0.5)**2 + (y-0.5)**2 <= r**2, 1, np.exp(-alpha*( (x-0.5)**2 + (y-0.5)**2 - r**2) ))

def correlated_gaussian_pdf_ten(X, mean=None, corr=None):
    """
    Compute the PDF value of a 2D Gaussian at points X with given mean and correlation matrix.
    Args:
        X (np.ndarray): shape (num_samples, 2), each row is a point [x, y]
        mean (list or None): Mean of the distribution. If None, uses [0, 0].
        corr (2x2 np.ndarray or None): Correlation matrix. If None, uses identity.
    Returns:
        np.ndarray: PDF value(s) at each row of X
    """
    X = np.asarray(X)
    if mean is None:
        mean = [0.0, 0.0]
    if corr is None:
        corr = np.eye(2)
    mean = np.array(mean, dtype=X.dtype)
    corr_inv = np.linalg.inv(corr)
    det_corr = np.linalg.det(corr)
    diff = X - mean
    exponent = -0.5 * np.sum(diff @ corr_inv * diff, axis=1)
    norm = 1.0 / (2 * np.pi * np.sqrt(det_corr))
    return norm * np.exp(exponent)



def symmetric_wing(x: tn.Tensor,
                   y: tn.Tensor,
                   rad: float,
                   new='swing',alpha=100):
    """
    x, y: same‐shaped tensors of coordinates (e.g. from meshgrid)
    rad: radius factor (you had t = 10*rad)
    new: `"wing1"` or other, controls the k‐shift
    returns: bump values = exp(-1e4 * (v^2 - (y - k*(arg-0.5))^2 + |...|))
    """
    # thickness parameter
    t = 10.0 * rad
    # horizontal shift
    x0 = -.5
    y0 = 0

    y = y-y0

    # small camber if "wing1"
    k = 0.4 if new == "wing1" else 0.0

    # polynomial coefficients
    csq = 0.2969 * t
    c1, c2, c3, c4 = (-0.1260*t, -0.3516*t, 0.2843*t, -0.1015*t)

    # compute "arg" with the domain test |x - 0.5 - x0| <= 0.5
    cond = (tn.abs(x - 0.5 - x0) <= 0.5)
    outside = tn.sign(x0 - x) + tn.sign(x - x0 - 1)
    arg = tn.where(cond, x - x0, outside)

    # ensure non-negative for sqrt
    arg_s = tn.clamp(arg, min=0.0)

    # profile v(arg)
    v = (c1*arg + c2*arg**2 + c3*arg**3 + c4*arg**4
         + csq * tn.sqrt(arg_s))

    # raw signed distance squared difference
    raw = v**2 - (y - k*(arg - 0.5))**2

    # keep only positive part (so values ≥ 0)
    pos = raw + tn.abs(raw)

    # final bump
    return tn.exp(-alpha * pos)




#### 3D wing functions


# --- Wing parameterization (yours) ---
b, c_r, c_t = 2.0, 1.0, 0.5
Lambda_LE = np.deg2rad(15)
phi       = 0.0
t_chord   = 0.12
ALPHA = 2**8

def chord(eta):
    return c_r*(1 - np.abs(eta))/2 + c_t*(1 + np.abs(eta))/2

def z_planform(eta):
    return 0.5 * b * eta * np.tan(phi)

def y_t(xi):
    return t_chord*(0.2969 * np.sqrt(xi)
             - 0.1260 * xi
             - 0.3516 * xi**2
             + 0.2843 * xi**3
             - 0.1015 * xi**4)

# --- One-time normalization (thickness = alpha / real_thick) ---
def precompute_thickness(alpha=ALPHA, samples=1000):
    xi_test    = np.linspace(0.0, 1.0, samples, endpoint=True)
    max_y_t    = float(np.max(y_t(xi_test)))
    c_root     = float(chord(0.0))
    real_thick = 2.0 * c_root * max_y_t
    if real_thick <= 0.0:
        raise ValueError("real_thick must be positive.")
    return np.float64(alpha / real_thick)

# use this by default; pass your own if you like
THICKNESS = precompute_thickness(alpha=ALPHA, samples=1000)

# --- Shared NumPy core: builds NON-inverted mask (0 inside → 1 outside) ---
def _wing_mask_core_np(ETA, XI, ZG, thickness):
    ETA = np.asarray(ETA, dtype=np.float64)
    XI  = np.asarray(XI,  dtype=np.float64)
    ZG  = np.asarray(ZG,  dtype=np.float64)
    if ETA.shape != XI.shape or ETA.shape != ZG.shape:
        raise ValueError("ETA, XI, ZG must have identical shapes.")

    mask = np.ones_like(ZG, dtype=np.float64)

    # full 3D core/band (equiv. to your 2D-slice + broadcast)
    in_core = (ETA >= -1.0) & (ETA <= 1.0) & (XI >= -1.0) & (XI <= 1.0)
    in_band = (XI  >= -0.5) & (XI  <= 0.5)
    use = in_core & in_band
    if not np.any(use):
        return mask

    # local surfaces only where used
    eta_u = ETA[use]
    xi_u  = XI[use]
    z_u   = ZG[use]

    xi_par = xi_u + 0.5                  # [-0.5,0.5] → [0,1]
    C_u    = chord(eta_u)
    Zt_u   = C_u * y_t(xi_par)
    zc_u   = z_planform(eta_u)
    zup_u  = zc_u + Zt_u
    zlow_u = zc_u - Zt_u

    # inside
    inside_u = (z_u >= zlow_u) & (z_u <= zup_u)
    mask[use] = np.where(inside_u, 0.0, mask[use])

    # above
    above_u = z_u > zup_u
    if np.any(above_u):
        dz = z_u[above_u] - zup_u[above_u]
        mask[use][above_u] = 1.0 - np.exp(-thickness * dz)

    # below
    below_u = z_u < zlow_u
    if np.any(below_u):
        dz = zlow_u[below_u] - z_u[below_u]
        mask[use][below_u] = 1.0 - np.exp(-thickness * dz)

    return mask  # 0 inside, →1 outside

# --- Your public APIs (same names), returning 1 - mask ---
def wing_mask(ETA, XI, ZG, alpha=ALPHA, thickness=THICKNESS):
    """
    Grid mask (torch float64). Returns 1 inside, 0 outside.
    """
    # if caller gave a different alpha but no thickness override, recompute
    if thickness is None:
        thickness = precompute_thickness(alpha=alpha, samples=1000)
    base = _wing_mask_core_np(ETA, XI, ZG, thickness)
    return tn.tensor(1.0 - base, dtype=tn.float64)

def wing_mask_pc(x, y, z, alpha=ALPHA, thickness=THICKNESS):
    """
    Pointwise mask (NumPy 1D). Returns 1 inside, 0 outside.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    if not (x.size == y.size == z.size):
        raise ValueError("x, y, z must have the same length.")
    # match alpha if caller overrides it
    if thickness is None:
        thickness = precompute_thickness(alpha=ALPHA, samples=1000)
    base = _wing_mask_core_np(x, y, z, thickness)
    return 1.0 - base

'''
# --- Wing parameterization ---
b, c_r, c_t = 2.0, 1.0, 0.5
Lambda_LE = np.deg2rad(15)
phi       = 0.0
t_chord        = 0.12

def chord(eta):
    return c_r*(1 - np.abs(eta))/2 + c_t*(1 + np.abs(eta))/2

def z_planform(eta):
    return 0.5 * b * eta * np.tan(phi)

def y_t(xi):
    return t_chord*(0.2969 * np.sqrt(xi)
             - 0.1260 * xi
             - 0.3516 * xi**2
             + 0.2843 * xi**3
             - 0.1015 * xi**4)



def wing_mask(ETA, XI, ZG, alpha=1000.0):
    """
    Build the 3D wing mask on given grids (exactly mirrors the reference code).

    ETA, XI, ZG : same-shaped ndarrays from meshgrid(indexing='ij'), i.e.
                  ETA = eta-grid, XI = xi-grid, ZG = z-grid
    alpha       : falloff factor (use 1000.0 to match reference)

    Returns
    -------
    mask : torch tensor (float64) with same shape as ZG
           =0 inside wing, transitions to 1 outside
    """
    # --- chordwise band in [-0.5, 0.5] over xi (XI) ---
    mask_band = (XI[:, :, 0] >= -0.5) & (XI[:, :, 0] <= 0.5)

    # --- 2D upper/lower surfaces (z at the band) ---
    Zup2d  = np.zeros_like(ETA[:, :, 0], dtype=np.float64)
    Zlow2d = np.zeros_like(ETA[:, :, 0], dtype=np.float64)

    xi_param = XI[:, :, 0][mask_band] + 0.5   # map [-0.5,0.5] → [0,1]
    eta_m    = ETA[:, :, 0][mask_band]

    # chord(), y_t(), z_planform() are assumed globally defined
    C_m = chord(eta_m)
    Zt  = C_m * y_t(xi_param)

    Zup2d[mask_band]  = z_planform(eta_m) + Zt
    Zlow2d[mask_band] = z_planform(eta_m) - Zt

    # --- Broadcast to 3D (use the same form as your script) ---
    Zup3d  = Zup2d[:, :, None]  * np.ones_like(ZG, dtype=np.float64)
    Zlow3d = Zlow2d[:, :, None] * np.ones_like(ZG, dtype=np.float64)

    # --- Thickness scale (exactly as in your script) ---
    xi_test    = np.linspace(0.0, 1.0, 1000)
    max_y_t    = np.max(y_t(xi_test))
    c_root     = chord(0.0)
    real_thick = 2.0 * c_root * max_y_t

    # --- Initialize mask to 1 everywhere (float64) ---
    mask = np.ones_like(ZG, dtype=np.float64)

    # --- Core region in eta/xi (2D, then lift to 3D) ---
    mask_eta  = (ETA[:, :, 0] >= -1.0) & (ETA[:, :, 0] <= 1.0)
    mask_xi   = (XI[:,  :, 0] >= -1.0) & (XI[:,  :, 0] <= 1.0)
    mask_core = mask_eta & mask_xi

    core3d = mask_core[:, :, None]
    band3d = mask_band[:, :, None]

    # 1) Inside wing → mask = 0
    inside = core3d & band3d & (ZG >= Zlow3d) & (ZG <= Zup3d)
    mask[inside] = 0.0

    # 2) Above upper surface → exponential falloff
    above = core3d & band3d & (ZG > Zup3d)

    zd_above = ZG - Zup3d
    mask[above] = 1.0 - np.exp(-(alpha/real_thick) * zd_above[above])

    # 3) Below lower surface → exponential falloff
    below = core3d & band3d & (ZG < Zlow3d)

    zd_below = Zlow3d - ZG
    mask[below] = 1.0 - np.exp(-(alpha/real_thick) * zd_below[below])

    # torch output to match your downstream code
    return 1-tn.tensor(mask, dtype=tn.float64)

def wing_mask_pc(x, y, z, alpha=1000.0):
    """
    Pointwise wing mask.

    x, y, z : 1D arrays/lists of the same length (eta, xi, z per point)
    alpha   : sharpness of exponential falloff

    Returns
    -------
    mask : 1D torch tensor with values in [0,1]
           0 inside the wing; transitions to 1 outside.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    assert x.shape == y.shape == z.shape, "x, y, z must have the same shape"

    n = x.size
    mask = np.ones(n, dtype=np.float64)

    # core region in eta/xi and chordwise band
    in_core = (x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= 1.0)
    in_band = (y >= -0.5) & (y <= 0.5)
    use     = in_core & in_band
    if not np.any(use):
        return tn.tensor(mask, dtype=tn.float64)  # all ones

    # thickness normalization
    xi_test    = np.linspace(0.0, 1.0, 1000)
    real_thick = 2.0 * float(chord(0.0)) * float(np.max(y_t(xi_test)))
    if real_thick <= 0.0:
        return tn.tensor(mask, dtype=tn.float64)

    # compute local upper/lower surfaces only for selected points
    eta_u   = x[use]
    xi_u    = y[use]
    xi_par  = xi_u + 0.5                  # map [-0.5,0.5] → [0,1]
    C_u     = chord(eta_u)
    Zt_u    = C_u * y_t(xi_par)
    zc_u    = z_planform(eta_u)
    zup_u   = zc_u + Zt_u
    zlow_u  = zc_u - Zt_u
    z_u     = z[use]

    # inside
    inside_u = (z_u >= zlow_u) & (z_u <= zup_u)
    mask[use] = np.where(inside_u, 0.0, mask[use])

    # above
    above_u = z_u > zup_u
    if np.any(above_u):
        dz = z_u[above_u] - zup_u[above_u]
        mask[use][above_u] = 1.0 - np.exp(-(alpha/real_thick) * dz)

    # below
    below_u = z_u < zlow_u
    if np.any(below_u):
        dz = zlow_u[below_u] - z_u[below_u]
        mask[use][below_u] = 1.0 - np.exp(-(alpha/real_thick) * dz)

    return 1-mask
'''

def ind_to_r_g(I, a, b, d=3, pdp=1, dtype=np.float64):
    """
    This function maps a multi-digit index to a real number in [a, b)^d.
    Each entry in I[k] is an integer in 0,...,2**pdp-1 and is expanded to its binary representation.
    The resulting binary digits are concatenated to form a long vector of 0's and 1's for each sample.
    Returns a numpy array of shape (num_samples, d).
    """
    I = np.asarray(I)
    num_samples = I.shape[0]
    l = I.shape[1]*pdp  # total number of binary digits per sample
    # Convert each entry to binary and concatenate
    I_bin = ((I[..., None] >> np.arange(pdp-1, -1, -1)) & 1).reshape(num_samples, -1)
    powers = np.array([2**(-i) for i in range(1, l//d+1)])
    result = np.empty((num_samples, d))
    for j in range(d):
        idx = np.arange(j, l, d)
        group = I_bin[:, idx]
        wI = group * powers
        x = (b[j] - a[j]) * wI.sum(axis=1) + a[j]
        result[:, j] = x

    return result

def _gauss(x, c, s):
    return tn.exp(-0.5*((x - c)/s)**2)

def _chirp(x, f0, f1):
    phase = 2*tn.pi*(f0*x + 0.5*(f1 - f0)*x**2)
    return tn.sin(phase)

def _val_deriv(fun, a, device, dtype):
    a = tn.tensor(a, device=device, dtype=dtype, requires_grad=True)
    y = fun(a)
    (dy,) = tn.autograd.grad(y, a, create_graph=False)
    return y.detach(), dy.detach()

def highrank_function(x, noise_std=0.0, eps=1e-12,
                      texture_gain=1.0,   # ↑ for more waviness
                      squash_range=4.0,   # targets ~[-squash_range, +squash_range]
                      squash_soft=2.0):   # smaller → stronger squash
    """
    Wavier C¹ function with same structure; constant+linear corrections keep C¹ at 0.20, 0.50, 0.80.
    Final tanh squash keeps values mostly within [-4,4] (adjust squash_range/soft to taste).
    """
    device, dtype = x.device, x.dtype

    # Global background (zero-mean-ish to avoid big drift)
    bg = (
        0.06*tn.sin(2*tn.pi*1.9*x + 0.2)
      + 0.05*tn.sin(2*tn.pi*3.7*x + 1.1)
      + 0.04*tn.cos(2*tn.pi*6.8*x + 0.7)
    )

    # ---------------- pieces (add oscillatory "texture" terms) ----------------
    def p1_fun(t):
        cusp = ((t - 0.08)**2 + eps)**(0.35)
        base = (0.28*tn.sin(6*tn.pi*t)
              + 0.25*_chirp(t, 2, 18) * _gauss(t, 0.15, 0.08)
              + 0.12*cusp
              + 0.10*(t - 0.10)**3 - 0.09*(t - 0.10)**4
              + 0.08*_gauss(t, 0.05, 0.03)*tn.sin(80*tn.pi*t))
        tex  = (
              0.18*tn.sin(120*tn.pi*t)*_gauss(t, 0.07, 0.025)
            + 0.14*tn.cos(90*tn.pi*t )*_gauss(t, 0.12, 0.05)
            + 0.08*tn.sin(2*tn.pi*11.3*t)*tn.sin(2*tn.pi*3.1*t)
        )
        return base + texture_gain*tex

    def p2_fun(t):
        rat1 =  0.10/(1.0 + ((t - 0.27)/0.015)**2)
        rat2 = -0.08/(1.0 + ((t - 0.44)/0.020)**2)
        base = (0.30*tn.cos(50*tn.pi*t) * _gauss(t, 0.30, 0.06)
              + 0.24*tn.sin(120*tn.pi*t)* _gauss(t, 0.38, 0.03)
              + rat1 + rat2
              + 0.18*_chirp(t, 14, 5) * _gauss(t, 0.46, 0.10)
              + 0.06*tn.tanh(35.0*tn.sin(18*tn.pi*t)) * _gauss(t, 0.34, 0.09))
        tex  = (
              0.22*tn.sin(180*tn.pi*t)*_gauss(t, 0.33, 0.035)
            + 0.16*tn.cos(230*tn.pi*t)*_gauss(t, 0.41, 0.028)
            + 0.10*tn.sin(2*tn.pi*7.7*t) * tn.sin(2*tn.pi*35*t)
        )
        return base + texture_gain*tex

    def p3_fun(t):
        base = (0.26*tn.sin(80*tn.pi*t) * _gauss(t, 0.66, 0.05)
              + 0.22*tn.cos(160*tn.pi*t)* _gauss(t, 0.72, 0.022)
              + 0.18*_chirp(t, 28, 8) * _gauss(t, 0.70, 0.11)
              + 0.05*tn.sin(6*tn.pi*t) * tn.sin(90*tn.pi*t) * _gauss(t, 0.62, 0.07)
              + 0.07*(t - 0.68)**2 - 0.10*(t - 0.68)**3)
        tex  = (
              0.20*tn.sin(200*tn.pi*t)*_gauss(t, 0.64, 0.030)
            + 0.18*tn.cos(260*tn.pi*t)*_gauss(t, 0.72, 0.020)
            + 0.10*tn.sin(2*tn.pi*5.5*t) * tn.sin(2*tn.pi*48*t)
        )
        return base + texture_gain*tex

    def p4_fun(t):
        cuspR = ((1.0 - t)**2 + eps)**(0.45)
        base = (0.24*tn.sin(220*tn.pi*t) * tn.exp(-60.0*(1.0 - t))
              + 0.16*tn.cos(100*tn.pi*t) * _gauss(t, 0.90, 0.03)
              + 0.12*cuspR
              + 0.05/(1.0 + ((t - 0.92)/0.015)**2)
              + 0.06*(t - 0.86)**4 - 0.08*(t - 0.86)**5)
        tex  = (
              0.18*tn.sin(300*tn.pi*t)*_gauss(t, 0.88, 0.020)
            + 0.12*tn.cos(180*tn.pi*t)*_gauss(t, 0.94, 0.018)
        )
        return base + texture_gain*tex

    a1, a2, a3 = 0.20, 0.50, 0.80
    p1 = p1_fun(x); p2 = p2_fun(x); p3 = p3_fun(x); p4 = p4_fun(x)

    # --------- C¹ corrections: match value & slope at the three joints -------
    v1, d1 = _val_deriv(p1_fun, a1, device, dtype)
    v2, d2 = _val_deriv(p2_fun, a1, device, dtype)
    c0_12, c1_12 = v1 - v2, d1 - d2
    p2_adj = p2 + c0_12 + c1_12*(x - a1)

    v2_a2, d2_a2 = _val_deriv(p2_fun, a2, device, dtype)
    vL_a2, dL_a2 = v2_a2 + c0_12 + c1_12*(a2 - a1), d2_a2 + c1_12
    v3, d3 = _val_deriv(p3_fun, a2, device, dtype)
    c0_23, c1_23 = vL_a2 - v3, dL_a2 - d3
    p3_adj = p3 + c0_23 + c1_23*(x - a2)

    v3_a3, d3_a3 = _val_deriv(p3_fun, a3, device, dtype)
    vL_a3, dL_a3 = v3_a3 + c0_23 + c1_23*(a3 - a2), d3_a3 + c1_23
    v4, d4 = _val_deriv(p4_fun, a3, device, dtype)
    c0_34, c1_34 = vL_a3 - v4, dL_a3 - d4
    p4_adj = p4 + c0_34 + c1_34*(x - a3)

    f_core = tn.where(x < a1, p1,
              tn.where(x < a2, p2_adj,
                tn.where(x < a3, p3_adj, p4_adj)))

    # Smooth extras (don’t affect joint regularity)
    series = sum((0.12/(k**1.25))*tn.sin(2*tn.pi*(1.7*k)*x + 0.3*(k % 3))
                 for k in range(1, 9))
    f0 = f_core + bg + 0.10*series

    # ---- Smooth range control to ~[-4,4] (keeps C¹, not C² at the joints) ---
    f = squash_range * tn.tanh(f0 / squash_soft)

    if noise_std > 0:
        f = f + noise_std*tn.randn_like(x)
    return f

def f_h3(x, eps=1e-12, amp=1.0):
    """
    Globally C^2 but NOT C^3, via localized (x-a)_+^3 bumps (C^2 hinges).
    With quadratic-spline interpolation on a uniform grid:
      ||f - I_h f||_{L2} = O(h^3).
    (With cubic splines you'll typically see ~h^{3.5-ε}.)
    """
    # smooth base (reuse a softened version of f_h4’s ingredients)
    base = (0.28*tn.sin(16*tn.pi*x) * _gauss(x, 0.20, 0.07)
          + 0.24*tn.cos(44*tn.pi*x) * _gauss(x, 0.36, 0.05)
          + 0.20*_chirp(x, 5, 18)         * _gauss(x, 0.58, 0.12)
          + 0.18*tn.sin(120*tn.pi*x)* _gauss(x, 0.73, 0.03))

    bg = (0.07*tn.sin(2*tn.pi*1.8*x + 0.2)
        + 0.05*tn.cos(2*tn.pi*3.3*x + 0.9))

    # C^2 “kink-of-order-3” atoms: (x-a)_+^3 * window
    def c2_bump(x, a, s, w=1.0):
        return w * tn.relu(x - a)**3 * _gauss(x, a, s)

    kinks = (
        c2_bump(x, 0.22, 0.030,  +0.8)
      + c2_bump(x, 0.37, 0.025,  -0.6)
      + c2_bump(x, 0.61, 0.035,  +0.7)
      + c2_bump(x, 0.82, 0.022,  -0.5)
    )

    # symmetric localizers to keep amplitude balanced
    pads = (
        -0.6*tn.relu(0.28 - x)**3 * _gauss(x, 0.28, 0.030)
        +0.5*tn.relu(0.42 - x)**3 * _gauss(x, 0.42, 0.028)
        -0.5*tn.relu(0.68 - x)**3 * _gauss(x, 0.68, 0.030)
        +0.4*tn.relu(0.88 - x)**3 * _gauss(x, 0.88, 0.022)
    )

    f0 = base + bg + kinks + pads
    return amp * tn.tanh(f0 / 2.5)