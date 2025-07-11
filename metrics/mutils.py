import torch as tn
import numpy as np


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