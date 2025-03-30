import torch as tn
import torchtt as tntt
from scipy.special import comb
from scipy.optimize import fsolve
import numpy as np
import math
# build delta d_j(i) =0 if i!=j 1 otherwise

def dmps(i,n,dtype=tn.float64):
    bs = f"{i:0{n}b}"

    cores = []

    for j in range(n):
        cores.append( tn.tensor([[1,0] if bs[j] == '0' else [0,1]],dtype=dtype).reshape(1,2,1) )

    return tntt.TT(cores)

# build delta for mpo

def dmpo(i,j,n,dtype=tn.float64):
    bsm = f"{i:0{n}b}"
    bsn = f"{j:0{n}b}"

    cores = []

    for a in range(n):
        if (bsm[a] == '0' and bsn[a]=='0'):
            cores.append(tn.tensor( [[1,0],[0,0]] ,dtype=dtype).reshape(1,2,2,1) )
        
        elif (bsm[a] == '0' and bsn[a]=='1'):
            cores.append(tn.tensor( [[0,1],[0,0]] ,dtype=dtype).reshape(1,2,2,1) )
        
        elif (bsm[a] == '1' and bsn[a]=='0'):
            cores.append(tn.tensor( [[0,0],[1,0]] ,dtype=dtype).reshape(1,2,2,1) )

        else:
            cores.append(tn.tensor( [[0,0],[0,1]] ,dtype=dtype).reshape(1,2,2,1) )


    return tntt.TT(cores)


# build linear monomial
def X_qtt(d, dtype = tn.float64):

    # create first core

    c0 = tn.zeros([1,2,2], dtype=dtype)
    c0[:,0,:] = tn.tensor([1,0])
    c0[:,1,:] = tn.tensor([1,1/2])

    # create intermediate cores
    icores = []

    for i in range(1,d-1):
        ci = cl = tn.zeros([2,2,2], dtype=dtype)
        ci[:,0,:] = tn.eye(2)
        ci[:,1,:] = tn.tensor([[1,1/2**(i+1)],[0,1]])    

        icores.append(ci)
    #create last core

    cl = tn.zeros([2,2,1], dtype=dtype)
    cl[:,0,:] = tn.tensor([0,1]).reshape(2,1)
    cl[:,1,:] = tn.tensor([1/2**(d),1]).reshape(2,1)


    return tntt.TT([c0] + icores + [cl])


#build identity MPO
def I_qtt(d,dtype=tn.float64):

    return tntt.TT([tn.eye(2,dtype=dtype).reshape(1,2,2,1)]*d)

#build shift matrices

# Right shift matrix
def R_qtt( indx, d,dtype = tn.float64):

    z = tn.zeros((2,2),dtype=dtype)
    I = tn.eye(2,dtype=dtype)
    J = tn.tensor([[0,1],[0,0]],dtype=dtype)
    Jp = tn.tensor([[0,0],[1,0]],dtype=dtype)
    #P = tn.tensor([[0,1],[1,0]],dtype=dtype)

    R = [ tn.stack([z, J],dim=1).reshape(1,2,2,2), tn.stack([J, I ],dim=1).reshape(1,2,2,2) ] 
    W = [ tn.stack([ tn.stack([I, Jp],dim=1), tn.stack([z,J],dim=1)]), tn.stack([ tn.stack([Jp, z],dim=1), tn.stack([J,I],dim=1) ]) ]
    V = [ tn.stack([I, z],dim=0).reshape(2,2,2,1), tn.stack([Jp, J ],dim=0).reshape(2,2,2,1) ]

    bs = f"{indx:0{d}b}"

    cores = [R[0] if bs[0]=='0' else R[1]]

    for b in bs[1:-1]:

        cores.append(W[0] if b == '0' else W[1])

    
    cores.append(V[0] if bs[-1] == '0' else V[1])

    return tntt.TT(cores)

#left shift matrix
def L_qtt( indx, d,dtype = tn.float64):

    z = tn.zeros((2,2),dtype=dtype)
    I = tn.eye(2,dtype=dtype)
    J = tn.tensor([[0,1],[0,0]],dtype=dtype)
    Jp = tn.tensor([[0,0],[1,0]],dtype=dtype)
    #P = tn.tensor([[0,1],[1,0]],dtype=dtype)

    Q = [ tn.stack([I, Jp],dim=1).reshape(1,2,2,2), tn.stack([Jp, z ],dim=1).reshape(1,2,2,2) ] 
    W = [ tn.stack([ tn.stack([I, Jp],dim=1), tn.stack([z,J],dim=1)]), tn.stack([ tn.stack([Jp, z],dim=1), tn.stack([J,I],dim=1) ]) ]
    V = [ tn.stack([I, z],dim=0).reshape(2,2,2,1), tn.stack([Jp, J ],dim=0).reshape(2,2,2,1) ]

    bs = f"{indx:0{d}b}"

    cores = [Q[0] if bs[0]=='0' else Q[1]]

    for b in bs[1:-1]:

        cores.append(W[0] if b == '0' else W[1])

    
    cores.append(V[0] if bs[-1] == '0' else V[1])

    return tntt.TT(cores)

def qtt_polynomial_cores(a, d,dir='f',basis = 'm',dtype = tn.float64):
    """
    Constructs QTT cores for a polynomial M(x) = sum(a_k * x^k for k=0..p).
    
    Args:
        a (list): Polynomial coefficients [a_0, a_1, ..., a_p].
        d (int): Number of QTT dimensions (log2(grid size)).
    
    Returns:
        list: QTT cores as PyTorch tensors.
    """
    # Initialize QTT cores list
    cores = []
    p = len(a)-1

    #define the polynomial basis
    if basis == 'm':
        def phi(x,k):
            return pow(x,k)
    elif basis == 'a':
        a = tn.tensor(a,dtype=dtype)
        alpha = (pow(2,d)-1)/pow(2,d)/(p)
        def phi(x,k):
            return x*pow(x+alpha*k,k-1)

        m_list = []
        for k in range(1, p + 2):
            row = [0] * (k - 1)
            for i in range(0, p + 2 - k):
                b = comb(i + k - 1, i)
                term = b * pow(-(k - 1),i)*pow(alpha,i)
                row.append(term)

            m_list.append(row)

        m_list[0][0] = 1
        M = tn.tensor(m_list, dtype=dtype)
        #print(M)
        a = M @ a


    if dir == 'f':
        # First core G1
        G1 = tn.zeros((1, 2, p + 1),dtype=dtype)  # Shape (1, 2, n+1)
        for s in range(p + 1):
            G1[0, 0, s] = a[s]
            G1[0, 1, s] = sum(a[k] * comb(k, s) * (phi(1/2,k-s)) for k in range(s, p + 1))
        cores.append(G1)

        # Intermediate cores G(x)
        for l in range(1, d - 1):
            G = tn.zeros((p + 1, 2, p + 1),dtype=dtype)  # Shape (n+1, 2, n+1)
            G[:, 0, :] = tn.eye(p+1,p+1) 
            for i in range(p + 1):
                for j in range(p + 1):
                    G[i, 1, j] = comb(i, i-j) * (phi(2**(-l-1),i-j)) if i >= j else 0
            cores.append(G)

        # Last core Gd
        Gd = tn.zeros((p + 1, 2, 1),dtype=dtype)  # Shape (n+1, 2, 1)
        v1 = tn.zeros(p+1)
        v1[0] = 1
        v2 = tn.tensor([1]+[phi(2**(-d),i) for i in range(1,p+1)])
        Gd[:, 0, 0] = v1
        Gd[:, 1, 0] = v2
        cores.append(Gd)

        return cores
    
    elif dir == 'b':
        # First core G1
        G1 = tn.zeros((1, 2, p + 1),dtype=dtype)  # Shape (1, 2, n+1)
        for s in range(p + 1):
            G1[0, 0, s] = a[s]
            G1[0, 1, s] = sum(a[k] * comb(k, s) * ( phi(2**(-d),k-s)  ) for k in range(s, p + 1))
        cores.append(G1)

        # Intermediate cores G(x)
        for l in range(1, d - 1):
            G = tn.zeros((p + 1, 2, p + 1),dtype=dtype)  # Shape (n+1, 2, n+1)
            G[:, 0, :] = tn.eye(p+1,p+1) 
            for i in range(p + 1):
                for j in range(p + 1):
                    G[i, 1, j] = comb(i, i-j) * ( phi(2**(-d+l),i-j) ) if i >= j else 0
            cores.append(G)

        # Last core Gd
        Gd = tn.zeros((p + 1, 2, 1),dtype=dtype)  # Shape (n+1, 2, 1)
        v1 = tn.zeros(p+1)
        v1[0] = 1
        v2 = tn.tensor([1]+[ phi(2**(-1),i)  for i in range(1,p+1)])
        Gd[:, 0, 0] = v1
        Gd[:, 1, 0] = v2
        cores.append(Gd)

        return [c.permute(2,1,0) for c in cores[::-1]]
    
def qtt_polynomial2_cores(coeffs, d,dtype=tn.float64):

    p = len(coeffs)-1
    print(p)
    Q, R = tn.linalg.qr(tn.tensor(coeffs,dtype=dtype))

    cores = []
    #first X cores
    for l in range(1, d - 1):
        G = tn.zeros((p + 1, 2, p + 1),dtype=dtype)  # Shape (n+1, 2, n+1)
        G[:, 0, :] = tn.eye(p+1,p+1) 
        for i in range(p + 1):
            for j in range(p + 1):
                G[i, 1, j] = comb(i, i-j) * ( (2**(-d+l)) ** (i - j)) if i >= j else 0
        cores.append(G)

    # Last core Gd
    Gd = tn.zeros((p + 1, 2, 1),dtype=dtype)  # Shape (n+1, 2, 1)
    v1 = tn.zeros(p+1)
    v1[0] = 1
    v2 = tn.tensor([1]+[(2**(-1))**i for i in range(1,p+1)])
    Gd[:, 0, 0] = v1
    Gd[:, 1, 0] = v2
    cores.append(Gd)
    coresx = [c.permute(2,1,0) for c in cores[::-1]]

    #lastXcore 
    coreX = tn.zeros((p + 1, 2, p+1),dtype=dtype)

    for r in range(p+1):
        for s in range(p + 1):
            coreX[s, 0, r] = Q[s,r]
            coreX[s, 1, r] = sum(Q[k,r] * comb(k, s) * ((2**(-d)) ** (k - s)) for k in range(s, p + 1))


    #first Y core
    coreY = tn.zeros((p + 1, 2, p+1),dtype=dtype)
    for r in range(p+1):
        for s in range(p + 1):
            coreY[r, 0, s] = R[r,s]
            coreY[r, 1, s] = sum(R[r,k] * comb(k, s) * ((2**(-1)) ** (k - s)) for k in range(s, p + 1))

    #last y cores 
    cores = []
    # Intermediate cores G(x)
    for l in range(1, d - 1):
        G = tn.zeros((p + 1, 2, p + 1),dtype=dtype)  # Shape (n+1, 2, n+1)
        G[:, 0, :] = tn.eye(p+1,p+1) 
        for i in range(p + 1):
            for j in range(p + 1):
                G[i, 1, j] = comb(i, i-j) * ( (2**(-l-1)) ** (i - j)) if i >= j else 0
        cores.append(G)

    # Last core Gd
    Gd = tn.zeros((p + 1, 2, 1),dtype=dtype)  # Shape (n+1, 2, 1)
    v1 = tn.zeros(p+1)
    v1[0] = 1
    v2 = tn.tensor([1]+[(2**(-d))**i for i in range(1,p+1)])
    Gd[:, 0, 0] = v1
    Gd[:, 1, 0] = v2
    cores.append(Gd)

    return coresx + [coreX] + [coreY] + cores


def connect(mps1,mps2,pd=3):

    if type(mps1) != list:
        cores1 = mps1.cores
    else:
        cores1 = mps1
        
    if type(mps2) != list:
        cores2 = mps2.cores
    else:
        cores2 = mps2

    cc = tn.einsum( 'ab,bc -> ac', cores1[-1].reshape(-1,pd), cores2[0].reshape(pd,-1))

    newcores = cores1[0:-1] + [ tn.einsum('ab, bcd -> acd',cc,cores2[1])] +  cores2[2:]

    return tntt.TT(newcores)

def hs(n_cores,t='1'):
    
    if t == '1':
        c1 = tntt.TT(tn.tensor([1,0],dtype=tn.float64).reshape(1,2,1))
        return tntt.kron(c1,tntt.ones([2]*(n_cores-1)))
    else:
        c1 = tntt.TT(tn.tensor([0,1],dtype=tn.float64).reshape(1,2,1))
        return tntt.kron(c1,tntt.ones([2]*(n_cores-1)))
    
def reduce(mps,i):
    cores = mps.cores
    new_cores = cores[:-2] + [tn.einsum('abc,cd -> abd',cores[-2],cores[-1][:,i,:])]

    return tntt.TT(new_cores)


def erank(ranks, dimensions):
    """
    Compute the effective rank (erank) of a TT representation.

    Parameters:
    - ranks (list): List of TT ranks, [r_0, r_1, ..., r_d].
    - dimensions (list): List of dimensions of the tensor, [n_1, n_2, ..., n_d].

    Returns:
    - float: The effective rank r_e.
    """
    d = len(dimensions)  # Number of dimensions

    # Compute the total number of parameters (S)
    S = sum(ranks[i] * dimensions[i] * ranks[i+1] for i in range(0, d - 1))

    # Define the function for fsolve
    def equation(re):
        return (
            re * dimensions[0]
            + sum(re**2 * dimensions[i] for i in range(1, d - 1))
            + re * dimensions[d - 1]
            - S
        )

    # Solve for r_e using fsolve
    r_e_initial_guess = np.mean(ranks)  # Initial guess for r_e
    r_e = fsolve(equation, r_e_initial_guess)[0]

    return r_e

def reversett(tt):

    cores = tt.cores
    newcores = [c.permute(2,1,0) for c in cores[::-1] ]

    return tntt.TT(newcores)

def reversemtt(mtt):

    cores = mtt.cores
    newcores = [c.permute(3,1,2,0) for c in cores[::-1] ]

    return tntt.TT(newcores)

def bit_reverse_indices(n):
    N = 2 ** n
    idx = tn.arange(N)
    idx_tensor = idx.view(*([2] * n))
    idx_reversed = idx_tensor.permute(*reversed(range(n)))
    return idx_reversed.reshape(-1)

def compute_morton_indices(n):
    """
    Compute a 2D tensor of Morton (Z-order) indices for a grid of size (2**n, 2**n).

    The Morton code for the coordinate (i, j) is given by interleaving the bits of i and j.
    
    Parameters:
        n (int): The number of bits, so that the grid size is (2**n, 2**n).
    
    Returns:
        tn.Tensor: A tensor of shape (2**n, 2**n) containing the Morton codes.
    """
    N = 2 ** n
    # Create a grid of row and column indices.
    # The 'indexing="ij"' keyword (available in recent PyTorch versions) ensures
    # that i corresponds to rows and j to columns.
    i, j = tn.meshgrid(tn.arange(N), tn.arange(N), indexing='ij')
    morton = tn.zeros((N, N), dtype=tn.int64)
    
    for bit in range(n):
        # Extract the bit at position 'bit' for i and j.
        # Then shift them to their appropriate locations in the interleaved number.
        morton |= (((i >> bit) & 1) << (2 * bit)) | (((j >> bit) & 1) << (2 * bit + 1))
    
    return morton

def zM(M, n):
    """
    Rearranges the entries of a matrix M (of shape (2**n, 2**n)) according to Z-order.
    
    The new coordinate for an element originally at (i,j) is determined by its Morton code:
      new_row = morton(i,j) // (2**n)
      new_col = morton(i,j) % (2**n)
    
    Parameters:
        M (tn.Tensor): The original 2D tensor of shape (2**n, 2**n).
        n (int): Number of bits such that M.shape = (2**n, 2**n).
    
    Returns:
        tn.Tensor: A new matrix of the same shape as M with entries rearranged in Z-order.
    """
    N = 2 ** n
    # Compute the Morton code for each coordinate.
    morton = compute_morton_indices(n)  # shape (N, N)
    # Compute new row and column indices for each element.
    new_i = morton // N  # integer division
    new_j = morton % N
    # Create an empty matrix to hold the rearranged entries.
    M_z = tn.empty_like(M)
    
    # Get original coordinate grids.
    idx = tn.arange(N)
    old_i, old_j = tn.meshgrid(idx, idx, indexing='ij')
    
    # Scatter the elements: for each (i, j) in the original matrix,
    # place M[i, j] at the new position (new_i[i, j], new_j[i, j]).
    M_z[new_i, new_j] = M[old_i, old_j]
    
    return M_z

def izM(M_z, n):
    """
    Given a 2D tensor M_z of shape (2**n, 2**n) whose elements are arranged
    in Z‑order, return a new matrix with the original (row‐major) ordering.
    
    This is done by computing the inverse permutation of the Morton (Z‑order)
    mapping and applying it to the flattened data.
    
    Parameters:
        M_z (tn.Tensor): A 2D tensor of shape (2**n, 2**n) in Z‑order.
        n (int): Such that the matrix is of shape (2**n, 2**n).
    
    Returns:
        tn.Tensor: The matrix with the original ordering.
    """
    N = 2 ** n
    # Ensure M_z is 2D.
    if M_z.ndim == 1:
        if M_z.numel() != N * N:
            raise ValueError("Number of elements does not equal 2**n * 2**n")
        M_z = M_z.reshape(N, N)
    
    # Compute the Morton permutation.
    morton = compute_morton_indices(n)  # shape (N, N)
    perm = morton.flatten()  # perm is a tensor of length N*N.
    
    # Compute the inverse permutation.
    inv_perm = tn.empty_like(perm)
    inv_perm[perm] = tn.arange(perm.numel(), dtype=perm.dtype)
    
    M_z_flat = M_z.flatten()
    # Apply the inverse permutation to recover the original flat ordering.
    M_orig_flat = M_z_flat[inv_perm]
    M_orig = M_orig_flat.reshape(N, N)
    return M_orig

def zkron(a,b):

    coresA = a.cores 
    coresB = b.cores

    l = len(coresA)
    zcores = []
    for i in range(len(coresA)):
        coreA = coresA[i].contiguous()  # Ensure contiguity
        coreB = coresB[i].contiguous()
        try:
            zcores.append(tn.kron(coreA, coreB))
        except RuntimeError as e:
            print(f"Error at index {i} with shapes {coreA.shape} and {coreB.shape}")
            raise e

    return tntt.TT(zcores)

def zkron3(a,b,c):

    coresA = a.cores 
    coresB = b.cores
    coresC = c.cores

    l = len(coresA)
    zcores = []
    for i in range(len(coresA)):
        coreA = coresA[i].contiguous()  # Ensure contiguity
        coreB = coresB[i].contiguous()
        coreC = coresC[i].contiguous()
        try:
            zcores.append(tn.kron(tn.kron(coreA, coreB),coreC))
        except RuntimeError as e:
            print(f"Error at index {i} with shapes {coreA.shape} and {coreB.shape}")
            raise e

    return tntt.TT(zcores)

def zukron(a,b):

    coresA = a.cores 
    coresB = b.cores

    l = len(coresA)
    zcores = []
    for i in range(len(coresA)):
        coreA = coresA[i].contiguous()  # Ensure contiguity
        coreB = coresB[i].contiguous()
        try:
            m1 = tn.kron(coreA, tn.eye( int(coreB.shape[0]) , int(coreB.shape[0])).reshape(int(coreB.shape[0]) , 1,int(coreB.shape[0]) ) )
            m2 = tn.kron(tn.eye( int(coreA.shape[2]) , int(coreA.shape[2]) ).reshape(int(coreA.shape[2]) , 1, int(coreA.shape[2])),coreB )
            zcores.append(m1)
            zcores.append(m2)
        except RuntimeError as e:
            print(f"Error at index {i} with shapes {coreA.shape} and {coreB.shape}")
            raise e
    return tntt.TT(zcores)

def zukron3(a,b,c):

    coresA = a.cores 
    coresB = b.cores
    coresC = c.cores

    l = len(coresA)
    zcores = []
    for i in range(len(coresA)):
        coreA = coresA[i].contiguous()  # Ensure contiguity
        coreB = coresB[i].contiguous()
        coreC = coresC[i].contiguous()
        try:
            rb = int(coreB.shape[0])
            rbr = int(coreB.shape[2])
            rc = int(coreC.shape[0])
            rar = int(coreA.shape[2]) 
            m1 = tn.kron( tn.kron(coreA, tn.eye(rb,rb).reshape(rb,1,rb) ) , tn.eye(rc, rc).reshape(rc,1,rc )  )
            m2 = tn.kron( tn.kron( tn.eye( rar, rar ).reshape(rar,1, rar),coreB ), tn.eye( rc,rc ).reshape(rc,1,rc) )
            m3 = tn.kron( tn.kron( tn.eye( rar, rar ).reshape(rar,1, rar),tn.eye( rbr,rbr ).reshape(rbr,1,rbr) ), coreC )
            zcores.append(m1)
            zcores.append(m2)
            zcores.append(m3)
        except RuntimeError as e:
            print(f"Error at index {i} with shapes {coreA.shape} and {coreB.shape}")
            raise e

    return tntt.TT(zcores)

def z_order_to_normal_torch(z_order_tensor, rows, cols):
    """
    Map a 2D array from Z-order to normal order using Pytn.

    Parameters:
        z_order_tensor (tn.Tensor): 1D tensor in Z-order.
        rows (int): Number of rows in the normal array.
        cols (int): Number of columns in the normal array.

    Returns:
        tn.Tensor: 2D tensor in normal order.
    """
    def interleave_bits(x, y):
        """
        Interleave bits of x and y for Morton (Z) order.
        """
        z = 0
        for i in range(max(x.bit_length(), y.bit_length())):
            z |= ((x >> i) & 1) << (2 * i + 1)
            z |= ((y >> i) & 1) << (2 * i)
        return z

    # Generate all (row, col) indices
    row_indices = tn.arange(rows, dtype=tn.int64)
    col_indices = tn.arange(cols, dtype=tn.int64)

    # Create a grid of indices
    grid_rows, grid_cols = tn.meshgrid(row_indices, col_indices, indexing="ij")
    
    # Flatten the grid
    flat_rows = grid_rows.flatten()
    flat_cols = grid_cols.flatten()

    # Compute Z-order indices for the entire grid
    z_indices = tn.tensor([interleave_bits(r.item(), c.item()) for r, c in zip(flat_rows, flat_cols)])

    # Map Z-order tensor to normal order
    normal_tensor = tn.zeros((rows, cols), dtype=z_order_tensor.dtype)
    for idx, z_idx in enumerate(z_indices):
        if z_idx < z_order_tensor.numel():  # Ensure within bounds
            r, c = divmod(idx, cols)
            normal_tensor[r, c] = z_order_tensor[z_idx]

    return normal_tensor




def plot_volume_voxels_binary(volume, threshold=0.5):
    """
    Plot a 3D volume (of shape (2**n, 2**n, 2**n)) as voxels after thresholding the values 
    to 0's and 1's, and then rescale the axis labels so that they run from 0 to 1.
    
    Parameters:
      volume    : PyTorch tensor with volumetric data.
      threshold : A threshold value to create a binary volume (default 0.5).
    """
    # Detach and move to CPU if needed
    if volume.requires_grad:
        volume = volume.detach()
    vol = volume.cpu().numpy()
    
    # Create a boolean (binary) volume by thresholding.
    vol_binary = vol > threshold
    
    # Get the number of voxels along each axis.
    N = vol_binary.shape[0]  # Assumes volume shape is (N, N, N)
    
    # Plot using voxels (using default index coordinates).
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(vol_binary, edgecolor='gray', facecolors='k')
    
    # Rescale the tick labels.
    # The voxel plotting uses indices 0 ... N (or 0 ... N-1), so we re-label them to 0 ... 1.
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()
    
    # Create new labels by dividing the tick positions by N.
    ax.set_xticklabels([f"{x:.2f}" for x in xticks/(xticks.max()-xticks.min())])
    ax.set_yticklabels([f"{y:.2f}" for y in yticks/(yticks.max()-yticks.min())])
    ax.set_zticklabels([f"{z:.2f}" for z in zticks/(zticks.max()-zticks.min())])
    
    # Optionally, you can also adjust the limits (they should already be 0 to N).
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_zlim(0, N)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Volume Visualization (Binary, axes scaled to [0,1])')
    plt.show()



def interleave_bits_3d(x, y, z):
        """
        Interleave the bits of x, y, and z to produce a Morton (Z-order) index.
        The convention here is:
          - The bit from x goes to bit position (3*i + 2)
          - The bit from y goes to bit position (3*i + 1)
          - The bit from z goes to bit position (3*i)
        for each bit position i.
        """
        res = 0
        m = max(x.bit_length(), y.bit_length(), z.bit_length())
        for i in range(m):
            res |= ((x >> i) & 1) << (3 * i + 2)
            res |= ((y >> i) & 1) << (3 * i + 1)
            res |= ((z >> i) & 1) << (3 * i)
        return res

def z_order_to_normal_torch_3d(z_order_tensor, dim0, dim1, dim2):
    """
    Map a 3D array from Z-order (Morton order) to canonical (normal row-major) order using Pytn.

    Parameters:
        z_order_tensor (tn.Tensor): 1D tensor containing the 3D volume in Morton order.
        dim0 (int): Size along the first dimension.
        dim1 (int): Size along the second dimension.
        dim2 (int): Size along the third dimension.

    Returns:
        tn.Tensor: A 3D tensor of shape (dim0, dim1, dim2) in canonical order.
    """
    

    # Generate all (r, c, d) indices for the canonical tensor.
    r_indices = tn.arange(dim0, dtype=tn.int64)
    c_indices = tn.arange(dim1, dtype=tn.int64)
    d_indices = tn.arange(dim2, dtype=tn.int64)

    # Create a grid of indices using 'ij' indexing.
    grid_r, grid_c, grid_d = tn.meshgrid(r_indices, c_indices, d_indices, indexing="ij")
    
    # Flatten the grid arrays.
    flat_r = grid_r.flatten()
    flat_c = grid_c.flatten()
    flat_d = grid_d.flatten()

    # Compute the Morton (Z-order) index for each (r, c, d) coordinate.
    z_indices = tn.tensor(
        [interleave_bits_3d(r.item(), c.item(), d.item()) 
         for r, c, d in zip(flat_r, flat_c, flat_d)],
        dtype=tn.int64
    )

    # Create an empty tensor for the canonical (normal) ordering.
    normal_tensor = tn.zeros((dim0, dim1, dim2), dtype=z_order_tensor.dtype)

    # Total number of elements in the volume.
    total_elements = dim0 * dim1 * dim2

    # For each canonical index (from 0 to total_elements-1),
    # determine its (r, c, d) coordinate in the canonical order.
    # Then use the corresponding Morton index (if in bounds) to get the value
    # from the z_order_tensor.
    for idx, z_idx in enumerate(z_indices):
        # Ensure the Morton index is within bounds.
        if z_idx < z_order_tensor.numel():
            # Compute 3D indices from the canonical flattened index.
            r = idx // (dim1 * dim2)
            rem = idx % (dim1 * dim2)
            c = rem // dim2
            d = rem % dim2
            normal_tensor[r, c, d] = z_order_tensor[z_idx]

    return normal_tensor

def reduce(mps,i):
    cores = mps.cores
    new_cores = cores[:-2] + [tn.einsum('abc,cd -> abd',cores[-2],cores[-1][:,i,:])]

    return tntt.TT(new_cores)

def reduceg(mps,i,j):
    cores = mps.cores

    if j == -1:
        new_cores = cores[:j-1] + [tn.einsum('abc,cd -> abd',cores[j-1],cores[j][:,i,:])] 
    elif j != 0:
        new_cores = cores[:j-1] + [tn.einsum('abc,cd -> abd',cores[j-1],cores[j][:,i,:])] + cores[j+1:] 
    else:
        new_cores =  [tn.einsum('ac,cbd -> abd',cores[0][:,i,:], cores[1])] + cores[2:] 

    return tntt.TT(new_cores)

def kron3(a,b,c):
    return tntt.kron(a,tntt.kron(b,c))


def op_reshape(op):

    cores = op.cores
    ncores = []
    for i in range(len(cores)):
        c = cores[i]
        dims = c.shape
        ncores.append(c.reshape(dims[0],2,2,dims[-1]))

    return tntt.TT(ncores)

# Right shift super matrix
def RO_qtt( l,dtype = tn.float64):

    z = tn.zeros((2,2),dtype=dtype)
    I = tn.eye(2,dtype=dtype)
    J = tn.tensor([[0,1],[0,0]],dtype=dtype)
    Jp = tn.tensor([[0,0],[1,0]],dtype=dtype)

    R = [ tn.stack([z, J],dim=1).reshape(1,2,2,2), tn.stack([J, I ],dim=1).reshape(1,2,2,2) ] 
    W = [ tn.stack([ tn.stack([I, Jp],dim=1), tn.stack([z,J],dim=1)]), tn.stack([ tn.stack([Jp, z],dim=1), tn.stack([J,I],dim=1) ]) ]
    V = [ tn.stack([I, z],dim=0).reshape(2,2,2,1), tn.stack([Jp, J ],dim=0).reshape(2,2,2,1) ]

    cores = [tn.stack((R[0],R[1]),dim=-1).reshape((1,4,2,2))] + [tn.stack((W[0],W[1]),dim=-1).reshape(2,4,2,2)]*(l-2) + [tn.stack((V[0],V[1]),dim=-1).reshape(2,4,2,1)]

    return tntt.TT(cores)

#left shift super matrix
def LO_qtt( l,dtype = tn.float64):

    z = tn.zeros((2,2),dtype=dtype)
    I = tn.eye(2,dtype=dtype)
    J = tn.tensor([[0,1],[0,0]],dtype=dtype)
    Jp = tn.tensor([[0,0],[1,0]],dtype=dtype)

    Q = [ tn.stack([I, Jp],dim=1).reshape(1,2,2,2), tn.stack([Jp, z ],dim=1).reshape(1,2,2,2) ] 
    W = [ tn.stack([ tn.stack([I, Jp],dim=1), tn.stack([z,J],dim=1)]), tn.stack([ tn.stack([Jp, z],dim=1), tn.stack([J,I],dim=1) ]) ]
    V = [ tn.stack([I, z],dim=0).reshape(2,2,2,1), tn.stack([Jp, J ],dim=0).reshape(2,2,2,1) ]

    cores = [tn.stack((Q[0],Q[1]),dim=-1).reshape((1,4,2,2))] + [tn.stack((W[0],W[1]),dim=-1).reshape(2,4,2,2)]*(l-2) + [tn.stack((V[0],V[1]),dim=-1).reshape(2,4,2,1)]

    return tntt.TT(cores)

# permutation super matrix
def PO_qtt( l,dtype = tn.float64):

    z = tn.zeros((2,2),dtype=dtype)
    I = tn.eye(2,dtype=dtype)
    J = tn.tensor([[0,1],[0,0]],dtype=dtype)
    Jp = tn.tensor([[0,0],[1,0]],dtype=dtype)
    H = tn.tensor([[0,1],[1,0]],dtype=dtype)

    P = [ tn.stack([I, H],dim=1).reshape(1,2,2,2), tn.stack([H, I ],dim=1).reshape(1,2,2,2) ] 
    W = [ tn.stack([ tn.stack([I, Jp],dim=1), tn.stack([z,J],dim=1)]), tn.stack([ tn.stack([Jp, z],dim=1), tn.stack([J,I],dim=1) ]) ]
    V = [ tn.stack([I, z],dim=0).reshape(2,2,2,1), tn.stack([Jp, J ],dim=0).reshape(2,2,2,1) ]

    cores = [tn.stack((P[0],P[1]),dim=-1).reshape((1,4,2,2))] + [tn.stack((W[0],W[1]),dim=-1).reshape(2,4,2,2)]*(l-2) + [tn.stack((V[0],V[1]),dim=-1).reshape(2,4,2,1)]

    return tntt.TT(cores)

#S super matrix
def SO_qtt( l,dtype = tn.float64):

    z = tn.zeros((2,2),dtype=dtype)
    I = tn.eye(2,dtype=dtype)
    J = tn.tensor([[0,1],[0,0]],dtype=dtype)
    Jp = tn.tensor([[0,0],[1,0]],dtype=dtype)
    eye = tn.eye(2,dtype=tn.float64)

    S = [ eye[1].reshape(1,1,1,2), eye[0].reshape(1,1,1,2) ] 
    W = [ tn.stack([ tn.stack([I, Jp],dim=1), tn.stack([z,J],dim=1)]), tn.stack([ tn.stack([Jp, z],dim=1), tn.stack([J,I],dim=1) ]) ]
    V = [ tn.stack([I, z],dim=0).reshape(2,2,2,1), tn.stack([Jp, J ],dim=0).reshape(2,2,2,1) ]

    cores = [tn.stack((S[0],S[1]),dim=-1).reshape((1,1,2,2))] + [tn.stack((W[0],W[1]),dim=-1).reshape(2,4,2,2)]*(l-1) + [tn.stack((V[0],V[1]),dim=-1).reshape(2,4,2,1)]

    return tntt.TT(cores)

def iO_qtt(d, dtype=tn.float64):
    X = tn.tensor([[0,1],[1,0]],dtype=dtype).reshape(1,2,2,1)
    return tntt.TT([X]*d)