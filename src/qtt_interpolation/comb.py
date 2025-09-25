import torchtt as tntt
from .utils import *
from .int_tools import d_coeff_mat
from torchtt._decomposition import SVD, rank_chop, lr_orthogonal
import sys
import secrets

def lcf(cores):
    R = [1] + [c.shape[2] for c in cores[:-1]] + [1]
    return tntt._decomposition.lr_orthogonal( cores, R, False )[0]

def rcf(cores):
    R = [1] + [c.shape[2] for c in cores[:-1]] + [1]
    return tntt._decomposition.rl_orthogonal( cores, R, False )[0]

def qttc_from_TT(ten,eps=1e-10):

    if type(ten) == tntt.TT:
        tt = ten
    else:
        tt = tntt.TT(ten,shape = list(ten.shape) ,eps=eps)
    cores = tt.cores

    comb = []
    for i in range(len(cores)):

        c = cores[i]
        rL, N, rR = c.shape
        head = c.permute(0, 2, 1).contiguous().reshape(rL*rR, N)
        p = int(np.log2(N))
        comb.append(tntt.TT(head, shape= [rL*rR]+ [2]*p, eps=eps).cores)


    return comb

# Tree TN functions and compressions
def get_roott(comb):
    """
    Build TT cores from comb cores.
    Each comb[i][0] is assumed to be (1, Rprod, N_i) with Rprod = rL_i * rR_i.
    We recover (rL_i, N_i, rR_i) using the previous core's right rank.
    """
    cores = []
    n_trunks = len(comb)

    for i in range(n_trunks):
        c = comb[i][0]
        assert c.dim() == 3, f"comb[{i}][0] must be 3D, got {tuple(c.shape)}"
        one, Rprod, N = c.shape
        assert one == 1, f"expected left singleton (1, Rprod, N) at site {i}, got {tuple(c.shape)}"

        # left rank (L) is previous core's right rank; 1 at the first site
        L = 1 if i == 0 else cores[-1].shape[2]
        assert (Rprod % L) == 0, f"Rprod={Rprod} not divisible by L={L} at site {i}"
        R = Rprod // L

        # Reorder and split: (1, Rprod, N) -> (1, N, Rprod) -> (1, N, L, R) -> (L, N, R)
        tmp = c.permute(0, 2, 1).contiguous()     # (1, N, Rprod)
        tmp = tmp.reshape(1, N, L, R)             # (1, N, L, R)
        core_i = tmp.permute(2, 1, 3, 0).contiguous().reshape(L, N, R)  # (L, N, R)

        cores.append(core_i)

    return cores

def update_comb(comb, root):
    """
    root[i]: TT core with shape (L, N, R)
    comb[i][0] will be set to shape (1, L*R, N) with R fastest inside L*R.
    """
    for i in range(len(comb)):
        L, N, R = root[i].shape
        tmp = root[i].permute(0, 2, 1).contiguous()  # (L, R, N)
        comb[i][0] = tmp.reshape(1, L * R, N)
    return

def get_R(root):
    r = [1]
    for i in range(len(root)):
        r.append(root[i].shape[2])
    return r


def comb_full(comb):

    root = get_roott(comb)
    r = get_R(root)
    f_tensors = []
    for i in range(len(comb)):
        if i == 0 or i == len(comb)-1:
            f_tensors.append(tntt.TT(comb[i],eps=1e-15).full())
        else:
            f_tensors.append(tntt.TT(comb[i],eps=1e-15).full().reshape(r[i],r[i+1],-1).swapaxes(1,2))

    full = tn.einsum('xa, xby, yc -> abc',*f_tensors)
        
    return full

def comb_full2(comb):
    root = tntt.TT(get_roott(comb),eps=1e-15)
    R = root.full()        # core, indices: a,b,c
    r = root.N             # mode sizes (for the factor reshapes)
    # factors to matrices: (r[i], -1) -> ax, by, cz
    A = comb[0][1].reshape(r[0], -1).contiguous()
    B = comb[1][1].reshape(r[1], -1).contiguous()
    C = comb[2][1].reshape(r[2], -1).contiguous()

    # 3D contraction: result has shape (X, Y, Z)
    full3 = tn.einsum('abc, ax, by, cz -> xyz', R, A, B, C)

    full2 = full3.contiguous()
    return full2



def _clone_comb(comb):
    new = []
    for branch in comb:
        new_branch = []
        for t in branch:
            new_branch.append(t.clone())
        new.append(new_branch)
    return new

def tree_compress(comb, eps=1e-8):
    """
    Returns a compressed COPY of `comb` (does not modify the input).
    Assumes external helpers exist: rcf, lcf, update_comb, get_roott, tntt.TT
    """
    out = _clone_comb(comb)

    # right-orthogonalize branches
    for i in range(len(out)):
        out[i] = rcf(out[i])

    # get the root (TT), compress it, then left-orthogonalize
    root = get_roott(out)
    roott_tt = tntt.TT(root,eps=eps).round(eps).cores
    roott_tt = lcf(roott_tt)
    #print( (root- tntt.TT(roott_tt,eps=1e-15)).norm().item()/root.norm().item()  )

    # push updated core info back into branches
    update_comb(out, roott_tt)

    # compress branches (except the first)
    for i in range(len(out) - 1, 0, -1):
        out[i] = rcf(tntt.TT(out[i],eps=eps).round(eps).cores)

    # recompute a right-orthogonal root cores list
    roott_cores = rcf(get_roott(out))    # list of cores; roott_cores[0] has shape (L, N, R)
    # push updated core info back into branches
    update_comb(out, roott_cores)
    
    # update the first branch head from root[0] with CORRECT reshape
    L, N, R = roott_cores[0].shape
    tmp = roott_cores[0].permute(0, 2, 1).contiguous()  # (L, R, N) so (L,R) are adjacent
    out[0][0] = tmp.reshape(1, L * R, N)                # (1, L*R, N)

    # compress the first branch
    out[0] = tntt.TT(out[0],eps=eps).round(eps).cores

    return out

def round_tt(tt_cores,R,eps,Rmax,is_ttm=False):
    """
    Rounds a TT-tensor (tt_cores have to be orthogonal)

    Parameters
    ----------
    tt_cores : list of torch tensors.
        Orthogonal TT cores.
    R : list of integers of length d+1.
        ranks of the TT-decomposition.
    eps : double.
        desired rounding accuracy.
    Rmax : list of integers
        the maximum rank that is allowed.

    Returns
    -------
    tt_cores : list of torch tensors.
        The TT-cores of the rounded tensor.
    R : list of inteders of length d+1.
        rounded ranks.

    """
    d = len(tt_cores)
    if d == 1:
        tt_cores = [tt_cores[0].clone()]
        return tt_cores, R
    tt_cores, R = lr_orthogonal(tt_cores, R, is_ttm)
    core_now = tt_cores[-1]
    eps = eps / np.sqrt(d-1) 

    
    for i in range(d-1,0,-1):
        core_next = tt_cores[i-1]
        
        core_now = tn.reshape(core_now,[R[i],-1])
        core_next = tn.reshape(core_next,[-1,R[i]])
        
        
        U, S, V = SVD(core_now)
        if S.is_cuda:
            r_now = min([Rmax[i],rank_chop(S.cpu().numpy(),tn.linalg.norm(S).cpu().numpy()*eps)])
        else:
            r_now = min([Rmax[i],rank_chop(S.numpy(),tn.linalg.norm(S).numpy()*eps)])
    
        U = U[:,:r_now]
        S = S[:r_now]
        V = V[:r_now,:]
        
        U = U @ tn.diag(S)
        R[i] = r_now
        core_next = core_next @ U
        core_now = V
        
        tt_cores[i] = tn.reshape(core_now,[R[i]]+list(tt_cores[i].shape[1:-1])+[R[i+1]])
        tt_cores[i-1] = tn.reshape(core_next,[R[i-1]]+list(tt_cores[i-1].shape[1:-1])+[R[i]])
        
        core_now = core_next
    
    return tt_cores, R

def tree_compress_2(comb0, eps=1e-8, Rmax = sys.maxsize, Rbmax = sys.maxsize,is_ttm=False):
    comb = _clone_comb(comb0)
    d = len(comb)
    fd = len(comb[0])
    if not isinstance(Rmax, list):
            Rmax = [1] + d*[Rmax] + [1]
    if not isinstance(Rbmax, list):
            Rbmax = [1] + fd*[Rbmax] + [1]
    for i in range(d-1,0,-1):
        Rb = get_R(comb[i] )
        comb[i],_ = round_tt(comb[i],Rb,eps,Rbmax)

        root = get_roott(comb)
        R = get_R(root)
        root, R = lr_orthogonal(root,R,is_ttm)  
        core_now = root[i]
        core_next = root[i-1]
        eps = eps / np.sqrt(d-1) 
        core_now = tn.reshape(core_now,[R[i],-1])
        core_next = tn.reshape(core_next,[-1,R[i]])
        U, S, V = SVD(core_now)
        if S.is_cuda:
            r_now = min([Rmax[i],rank_chop(S.cpu().numpy(),tn.linalg.norm(S).cpu().numpy()*eps)])
        else:
            r_now = min([Rmax[i],rank_chop(S.numpy(),tn.linalg.norm(S).numpy()*eps)])
        U = U[:,:r_now]
        S = S[:r_now]
        V = V[:r_now,:]
        
        U = U @ tn.diag(S)
        R[i] = r_now
        core_next = core_next @ U
        core_now = V
        
        root[i] = tn.reshape(core_now,[R[i]]+list(root[i].shape[1:-1])+[R[i+1]])
        root[i-1] = tn.reshape(core_next,[R[i-1]]+list(root[i-1].shape[1:-1])+[R[i]])
        
        update_comb(comb, root)
        
    comb[0],_ = round_tt(comb[0],get_R(comb[0]),eps,Rbmax)

    return comb

# --------- build comb from (root core, factor matrices U) ----------
def combqtt(core, U, eps=1e-10):
    """
    Build 'comb' branches:
      - QTT-decompose each U[i] with dims [ranks[i], 2, 2, ..., 2]
      - Replace its first QTT core by a 'head' that couples the root TT ranks.
    Returns: list of branches, each a list of TT cores.
    """
    ranks = list(core.shape)
    roott = tntt.TT(core, ranks, eps=eps)           # root TT with 3 cores (rL, N, rR)
    rc = roott.cores                                 # each core: (rL, N, rR)
    rr = roott.R                                     # TT ranks [r0, r1, r2, r3], r3=1

    comb = []
    for i in range(3):
        Ui = U[i].T
        assert Ui.shape[0] == ranks[i], f"U[{i}] must be (ranks[{i}], 2**p)"
        cols = Ui.shape[1]
        p = int(np.log2(cols));  assert 2**p == cols, f"U[{i}].shape[1] must be power of 2"

        # QTT of Ui: dims [ranks[i], 2, 2, ..., 2]
        btt = tntt.TT(Ui, [ranks[i]] + [2]*p, eps=eps).cores
        # head from root core: flatten (rL, N, rR) -> (rL*rR, N) with rR-fast
        c = rc[i]  # (rL, N, rR)
        ri = c.permute(0, 2, 1).contiguous().reshape(rr[i]*rr[i+1], ranks[i])  # (a, b)

        # contract with first QTT core (1, ranks[i], r1): ab, xbc -> xac  => (1, a, r1)
        head = tn.einsum('ab,xbc->xac', ri, btt[0])
        comb.append([head] + btt[1:])

    return comb


# --------- reconstruct full dense tensor from comb ----------
def combqtt_full(comb):
    """
    Reconstruct the full dense tensor of shape (2**p0, 2**p1, 2**p2).
    Strategy:
      - Recover the root coupling tensor R from comb (via get_roott).
      - For each branch i, contract all its cores to a dense (rL*rR, 2**p_i),
        then reshape to (rL, rR, 2**p_i).
      - Contract: full[x,y,z] = sum_{a,b,c} R[a,b,c] * F0[a,b,x] * F1[b,c,y] * F2[c,*,z]
    """
    root = get_roott(comb)          # TT with 3 cores; root.full() -> (a,b,c)

    rr = get_R(root)                     # TT ranks [r0, r1, r2, r3], expect r3 == 1

    # Dense factors from branches
    F0 = tntt.TT(comb[0], eps=1e-15).full()  # (r0*r1, 2**p0)
    F1 = tntt.TT(comb[1], eps=1e-15).full()  # (r1*r2, 2**p1)
    F2 = tntt.TT(comb[2], eps=1e-15).full()  # (r2*r3, 2**p2)

    F0 = F0.reshape(rr[1], -1)        # (a, b, X)
    F1 = F1.reshape(rr[1], rr[2], -1)        # (b, c, Y)
    F2 = F2.reshape(rr[2], -1)        # (c, r3, Z)

    # Contract to full (X, Y, Z)
    full = tn.einsum('bx,bcy,cz->xyz', F0, F1, F2)
    return full

    

def tu_to_comb(core, U, eps=1e-10):
    comb = []
    ranks = list(core.shape)
    roott = tntt.TT(core, ranks, eps=eps)
    roott_cores = roott.cores      # each core: (rL, N, rR)
    rranks = roott.R               # TT ranks

    for i in range(len(ranks)):
        # factor "btt": swap to (pd, bd, 1)
        s = U[i].shape
        btt = U[i].transpose(0, 1).reshape(s[1], s[0], 1).contiguous()

        # core "ri": make (1, rL*rR, N) with rR fastest inside the product
        c = roott_cores[i]                         # (rL, N, rR)
        ri = c.permute(0, 2, 1).contiguous()       # (rL, rR, N)
        ri = ri.reshape(1, rranks[i]*rranks[i+1], ranks[i])  # (1, rL*rR, N)

        comb.append([ri.to(tn.float64), btt.to(tn.float64)])

    return comb

import torch.nn.functional as F

def blockdiag_first_site(c1, c2):
    """
    c1: (1, pd, bd1)
    c2: (1, pd, bd2)
    returns: (1, pd, bd1+bd2)
    """
    bd1, bd2 = c1.shape[2], c2.shape[2]
    return F.pad(c1, (0, bd2, 0, 0, 0, 0)) + F.pad(c2, (bd1, 0, 0, 0, 0, 0))

def blockdiag_last_site(c1, c2):
    """
    c1: (bd1, pd, 1)
    c2: (bd2, pd, 1)
    returns: (bd1+bd2, pd, 1)
    """
    bd1, bd2 = c1.shape[0], c2.shape[0]
    return F.pad(c1, (0, 0, 0, 0, 0, bd2)) + F.pad(c2, (0, 0, 0, 0, bd1, 0))

def blockdiag_core(c1, c2,pad_left: bool = True, pad_right: bool = True) :
    """
    Put c1 in the top-left block and c2 in the bottom-right block of the rank dims.
    c1, c2 shapes: (rL, N, rR); must have the same N.
    pad_left/right allow skipping padding on the boundary ends if needed.
    """
    assert c1.dim() == 3 and c2.dim() == 3, "expected 3D cores (rL, N, rR)"
    assert c1.shape[1] == c2.shape[1], "mode size mismatch"

    r1L, N, r1R = c1.shape
    r2L, _, r2R = c2.shape

    # pad tuples are (lastL, lastR, midL, midR, firstL, firstR)
    pad_c1 = (
        0, (r2R if pad_right else 0),  # add zeros to the RIGHT of last dim
        0, 0,                          # no padding on the mode dim
        0, (r2L if pad_left else 0),   # add zeros to the RIGHT of first dim
    )
    pad_c2 = (
        (r1R if pad_right else 0), 0,  # add zeros to the LEFT  of last dim
        0, 0,
        (r1L if pad_left else 0), 0,   # add zeros to the LEFT  of first dim
    )

    return F.pad(c1, pad_c1) + F.pad(c2, pad_c2)

def blockdiag_3bond(c1: tn.Tensor, c2: tn.Tensor,
                    grow=(True, True, True)) -> tn.Tensor:
    """
    Block-diagonal merge of two 3-bond cores.
    c1, c2: (rL, rM, rR)  (all dimensions are bond ranks)
    grow: which legs to grow (L, M, R). If a leg is False, that dim must match.

    Returns: (rL1 + rL2 if grow[0] else rL1,  rM1 + rM2 if grow[1] else rM1,
              rR1 + rR2 if grow[2] else rR1)
    """
    assert c1.dim() == 3 and c2.dim() == 3, "expected 3D tensors (rL, rM, rR)"
    r1L, r1M, r1R = c1.shape
    r2L, r2M, r2R = c2.shape

    # If a leg is not grown, sizes must match
    if not grow[0]: assert r1L == r2L, "left ranks must match when not growing"
    if not grow[1]: assert r1M == r2M, "middle ranks must match when not growing"
    if not grow[2]: assert r1R == r2R, "right ranks must match when not growing"

    # torch.nn.functional.pad order for 3D: (lastL, lastR, midL, midR, firstL, firstR)
    pad_c1 = (
        0, (r2R if grow[2] else 0),   # dim2 (rR): add zeros on the RIGHT
        0, (r2M if grow[1] else 0),   # dim1 (rM): add zeros on the RIGHT
        0, (r2L if grow[0] else 0),   # dim0 (rL): add zeros on the RIGHT
    )
    pad_c2 = (
        (r1R if grow[2] else 0), 0,   # dim2: add zeros on the LEFT
        (r1M if grow[1] else 0), 0,   # dim1: add zeros on the LEFT
        (r1L if grow[0] else 0), 0,   # dim0: add zeros on the LEFT
    )
    return F.pad(c1, pad_c1) + F.pad(c2, pad_c2)

def _head_to_3bond(head, L, R):
    # head: (1, L*R, rM) with R fastest inside L*R  →  (L, rM, R)
    x = head.contiguous().reshape(1, L*R, -1)     # ensure viewable
    x = x.reshape(L, R, -1).permute(0, 2, 1).contiguous()  # (L, rM, R)
    return x

def _3bond_to_head(x):
    # x: (Lsum, rM, Rsum)  →  (1, Lsum*Rsum, rM) with Rsum fastest
    Ls, rM, Rs = x.shape
    return x.permute(0, 2, 1).contiguous().reshape(1, Ls*Rs, rM)

def comb_sum(c1, c2):
    assert len(c1) == len(c2)
    out = []

    root1 = get_roott(c1)
    root2 = get_roott(c2)
    r1 = get_R(root1)   # TT ranks of root1: [r0, r1, r2, r3]
    r2 = get_R(root2)   # TT ranks of root2

    n_trunks = len(c1)
    for i in range(n_trunks):
        branch = []
        m = len(c1[i])
        for j in range(m):
            A, B = c1[i][j], c2[i][j]

            if j == 0:
                if i == 0 or i == n_trunks - 1:
                    # boundary branch head already has shape (1, a, rM) ~ (rL=1, rM=a, rR=rM)
                    # grow middle & right, keep left (1) unchanged
                    branch.append(blockdiag_3bond(A, B, grow=(False, True, True)))
                else:
                    # interior branch: turn head (1, L*R, rM) into (L, rM, R),
                    # block-diag, then map back to (1, (L+L’)*(R+R’), rM)
                    L1, R1 = r1[i],   r1[i+1]
                    L2, R2 = r2[i],   r2[i+1]

                    A3 = _head_to_3bond(A, L1, R1)
                    B3 = _head_to_3bond(B, L2, R2)

                    X = blockdiag_3bond(A3, B3)   # (L1+L2, rM, R1+R2)
                    H = _3bond_to_head(X)         # (1, (L1+L2)*(R1+R2), rM)
                    branch.append(H)

            elif j == m - 1:
                # last site of a QTT branch: (bd, 2, 1)
                branch.append(blockdiag_last_site(A, B))

            else:
                # middle QTT cores: (rL, 2, rR)
                branch.append(blockdiag_core(A, B))

        out.append(branch)
    return out

def comb_op(comb,op,leg_indx):

    d = comb[leg_indx][0].shape[1]
    c_op = tntt.kron( tntt.TT(tn.eye(d,dtype=tn.float64)), op )
    nc = c_op @ tntt.TT(comb[leg_indx])
    comb[leg_indx] = nc.cores
    return 


def comb_comp(comb,reduced=False):
    entries = 0
    dim = 1
    if reduced:
        for i in range(len(comb)):
            if i == 0 or i == len(comb)-1:
                n_comb = [tn.einsum('abc,cde->abde',comb[i][0],comb[i][1])]+ comb[i][2:]
            else:
                n_comb = comb[i]
            entries += sum([tn.numel(c) for c in n_comb[i]])
            N = [comb[i][j].shape[1] for j in range(1,len(comb[i]))]
            dim *= int(np.prod(np.array(N)))
        return entries/dim
    else:
        for i in range(len(comb)):
            entries += sum([tn.numel(c) for c in comb[i]])
            N = [comb[i][j].shape[1] for j in range(1,len(comb[i]))]
            dim *= int(np.prod(np.array(N)))
        return entries/dim

def tt_comp(tt):
    if type(tt) == list :
        d = 1
        for i in range(len(tt)):
            s = tt[i].shape
            d *= int(s[1])
        return sum([tn.numel(tn.from_numpy(c)) for c in tt])/d
    else:
        d = 1
        for i in range(len(tt.N)):
            d *= int(tt.N[i])
        return sum([tn.numel(c) for c in tt.cores])/d


def comb_max_rank(comb):
    rmax = 0
    root = get_roott(comb)
    R = get_R(root)
    rmax = max(rmax, max(R))
    for i in range(len(comb)):
        rmax = max( rmax, max( get_R( comb[i] ) ) )
    
    return rmax

def comb_interpolate_p(comb0,fscale, eps =  1e-14,order = 1, p_derivative = [0,0,0]):

    comb = _clone_comb(comb0)

    ni = len(comb[0])-1
    nc = fscale - ni



    if order == 1:
        # Build Kernel interpolant
        Mkc = tn.tensor([
            [0, 2, 0, 0],
            [-1, 0, 1, 0],
            [2, -5, 4, -1],
            [-1, 3, -3, 1]
        ], dtype=tn.float64) /2
        Mkct = Mkc.t()

    elif order == 2:
        Mkct = tn.tensor([
            [1, -3, 3, -1],
            [4, 0, -6, 3],
            [1, 3, 3, -3],
            [0,0, 0, 1]
        ], dtype=tn.float64) /6

    sm10 = P_qtt( 2**ni -1, ni)
    O20 = P_qtt( 2**ni -2, ni)
    Om10 = P_qtt( 1, ni)

    epsp = 1e-14
    Mx = d_coeff_mat(Mkct, p_derivative[0]) 
    My = d_coeff_mat(Mkct, p_derivative[1])
    Mz = d_coeff_mat(Mkct, p_derivative[2])
    polsx = [tntt.TT(qtt_polynomial_cores(Mx[i], nc)).round(epsp) for i in range(4)]
    polsy = [tntt.TT(qtt_polynomial_cores(My[i], nc)).round(epsp) for i in range(4)]
    polsz = [tntt.TT(qtt_polynomial_cores(Mz[i], nc)).round(epsp) for i in range(4)]
    pols = [polsx, polsy, polsz]
    # interpolate dimensions

    for i in range(3):

        d = comb[i][0].shape[1]
        sm1 = tntt.kron( tntt.TT(tn.eye(d,dtype=tn.float64), shape=[(d,d)]), sm10 )
        Om1 = tntt.kron( tntt.TT(tn.eye(d,dtype=tn.float64),shape=[(d,d)]), Om10 )
        O2 = tntt.kron( tntt.TT(tn.eye(d,dtype=tn.float64),shape=[(d,d)]), O20 )

        l = tntt.TT(comb[i],eps=1e-15)

        int_l = tntt.kron(Om1 @  l, pols[i][0]) + tntt.kron(l, pols[i][1]) + tntt.kron(sm1 @ l, pols[i][2]) + tntt.kron(O2 @ l, pols[i][3])

        comb[i] = int_l.round(eps).cores

    comb = tree_compress(comb, eps=eps)

    return comb

def d_coeff_mat2(M, order=0):
    if order == 0:
        return M
    elif order == 1:
        return M[:, 1:] * tn.tensor([1, 2], dtype=M.dtype)
    elif order == 2:
        return M[:, 2:] * tn.tensor([2], dtype=M.dtype)
    else:
        raise ValueError("Only derivative orders 0, 1, and 2 are supported.")
    
def qtt2pol(i,mk1,mk2,nc,eps=1e-10,d=0):
    m1 = d_coeff_mat2(mk1, d)  
    m2 = d_coeff_mat2(mk2, d)  
    phi0 = hs(nc,'1')*tntt.TT(qtt_polynomial_cores(m1[i], nc)) 
    phi1 = hs(nc,'0')*tntt.TT(qtt_polynomial_cores(m2[i], nc)) 
    return (phi0+phi1).round(eps)

def comb_interpolate_p_quad(comb0,fscale, eps =  1e-14, p_derivative = [0,0,0]):

    comb = _clone_comb(comb0)

    ni = len(comb[0])-1
    nc = fscale - ni

    mk1 = tn.tensor([
        [1/8, -1/2, 1/2],
        [3/4,  0.0, -1.0],
        [1/8,  1/2, 1/2],
        [0.0,  0.0, 0.0]
    ], dtype=tn.float64)
    mk2 = tn.tensor([
        [0.0,  0.0, 0.0],
        [9/8,  -3/2, 1/2],
        [-1/4,  2.0, -1],
        [1/8,  -1/2, 1/2]], dtype=tn.float64)

    sm10 = P_qtt( 2**ni -1, ni)
    O20 = P_qtt( 2**ni -2, ni)
    Om10 = P_qtt( 1, ni)

    epsp = 1e-14
    
    polsx = [qtt2pol(i,mk1,mk2,nc,d=p_derivative[0],eps=epsp) for i in range(4)]
    polsy = [qtt2pol(i,mk1,mk2,nc,d=p_derivative[1],eps=epsp) for i in range(4)]
    polsz = [qtt2pol(i,mk1,mk2,nc,d=p_derivative[2],eps=epsp) for i in range(4)]
    
    pols = [polsx, polsy, polsz]
    # interpolate dimensions

    for i in range(3):

        d = comb[i][0].shape[1]
        sm1 = tntt.kron( tntt.TT(tn.eye(d,dtype=tn.float64), shape=[(d,d)]), sm10 )
        Om1 = tntt.kron( tntt.TT(tn.eye(d,dtype=tn.float64),shape=[(d,d)]), Om10 )
        O2 = tntt.kron( tntt.TT(tn.eye(d,dtype=tn.float64),shape=[(d,d)]), O20 )

        l = tntt.TT(comb[i],eps=1e-15)

        int_l = tntt.kron(Om1 @  l, pols[i][0]) + tntt.kron(l, pols[i][1]) + tntt.kron(sm1 @ l, pols[i][2]) + tntt.kron(O2 @ l, pols[i][3])

        comb[i] = int_l.round(eps).cores

    comb = tree_compress(comb, eps=eps)

    return comb



def _clone_comb(comb):
    new = []
    for branch in comb:
        new_branch = []
        for t in branch:
            new_branch.append(t.clone())
        new.append(new_branch)
    return new

def fuse_root(root):

    return tn.einsum('abc,cde,efg -> bdf', *root)

def marginal_probs_comb(core, Tprev, l_index):
    if l_index == 0:
        ampls = tn.einsum('nijk,iax->naxjk', Tprev, core)
    elif l_index == 1:
        ampls = tn.einsum('nijk,jax->naixk', Tprev, core)
    else:
        ampls = tn.einsum('nijk,kax->naijx', Tprev, core)
    probs = (ampls**2).sum(dim=(2,3,4))
    return probs

def marginal_probs_comb_rw(row, Tprev):

    ampls = tn.einsum('nijk,iax,jby,kcz->nabcxyz', Tprev, row[0],row[1],row[2]).flatten(1,3)
    probs = (ampls**2).sum(dim=(2,3,4))
    return probs

def sample_r(probs, rng, sentinel=-1):
    p = probs.clamp_min(0)
    sums = p.sum(dim=1); nz = sums > 0
    out = tn.full((p.size(0),), sentinel, device="cpu")
    if nz.any():
        if rng is None:
            rng = tn.Generator(device="cpu").seed()
        out[nz] = tn.multinomial(p[nz], 1, replacement=True, generator=rng).squeeze(1)
    return out

def sample_from_comb(comb, Nsamples, rng = None, return_joint=False):

    if rng == None:
        rng = ensure_torch_rng()

    D = len(comb)
    L = len(comb[0])
    combc = _clone_comb(comb)
    #right orthogonalize the cores

    for i in range(D):
        combc[i] = rcf(combc[i])
    
    root =  get_roott(combc)
    fr = fuse_root(root)

    # first_core
    c0x = combc[0][1]
    mps0 = tn.einsum('abc,axy -> bcxy',c0x,fr)

    p0 = (mps0**2).sum(dim=(1,2,3))
    p0 = tn.clamp(p0, min=0)
    p0 /= p0.sum()
    first_samples = tn.multinomial(p0, num_samples=Nsamples,replacement=True, generator=rng)
    Tprev = mps0[first_samples]        # (Nsamples, D1)
    prob_prev = (Tprev**2).sum(dim=(1,2,3))        # current accumulated probability
    bitstrings = first_samples.reshape(1, Nsamples)
    # iterate over rows
    for j in range(1,L):

        for k in range(3):
            #skip first core
            if j == 1 and k == 0:
                pass
            else:
                cur_core = combc[k][j]
                probs_cur = marginal_probs_comb(cur_core,Tprev,k)
                probs_cond = probs_cur / prob_prev[:, None]
                samples_site = sample_r(probs_cond, rng)
                Tsite = cur_core[:,samples_site,:]   
                if k == 0:
                    Tprev = tn.einsum('nijk,inx->nxjk', Tprev, Tsite)
                elif k == 1:
                    Tprev = tn.einsum('nijk,jnx->nixk', Tprev, Tsite)
                else:
                    Tprev = tn.einsum('nijk,knx->nijx', Tprev, Tsite)
                prob_prev = (Tprev**2).sum(dim=(1,2,3))  
                bitstrings = tn.concatenate((bitstrings, samples_site.reshape(1, Nsamples)), axis=0)

    if return_joint:
        return bitstrings, prob_prev

    return bitstrings

def ints_to_bits_concat(a, bits=None, msb_first=True):
    """
    a: tensor of nonnegative ints (any shape)
    bits: number of bits; if None, picked from max(a)
    msb_first: True -> [MSB ... LSB], False -> [LSB ... MSB]
    Returns:
      B: per-value bit tensor with an extra last dim of length `bits`
      concat_1d: all bits flattened into a single 1D tensor
    """
    a = tn.as_tensor(a, dtype=tn.long)
    maxv = int(a.max().item()) if a.numel() > 0 else 0
    bits = max(1, (maxv.bit_length() if bits is None else bits))

    if msb_first:
        shifts = tn.arange(bits-1, -1, -1, device=a.device)
    else:
        shifts = tn.arange(bits, device=a.device)

    B = ((a[..., None] >> shifts) & 1).to(a.dtype)  # shape: a.shape + (bits,)
    return B


def sample_from_comb_rw(comb, Nsamples, rng = None, return_joint=False):

    if rng == None:
        rng = ensure_torch_rng()

    D = len(comb)
    L = len(comb[0])
    combc = _clone_comb(comb)
    #right orthogonalize the cores

    for i in range(D):
        combc[i] = rcf(combc[i])
    
    root =  get_roott(combc)
    fr = fuse_root(root)

    # first_row
    r0 = [combc[i][1] for i in range(3)]
    mps0 = tn.einsum('xyz,xna,ymb,zlc -> nmlabc',fr,r0[0],r0[1],r0[2]).flatten(0,2)

    p0 = (mps0**2).sum(dim=(1,2,3))
    p0 = tn.clamp(p0, min=0)
    p0 /= p0.sum()

   
    first_samples = tn.multinomial(p0, num_samples=Nsamples,replacement=True, generator=rng)
    Tprev = mps0[first_samples]        # (Nsamples, D1)
    prob_prev = (Tprev**2).sum(dim=(1,2,3))        # current accumulated probability
    bitstrings = first_samples.reshape(1, Nsamples)

    # iterate over rows
    for j in range(2,L):

        cur_row = [combc[i][j] for i in range(3)]
        probs_cur = marginal_probs_comb_rw(cur_row,Tprev)
        probs_cond = probs_cur / prob_prev[:, None]
        samples_site = sample_r(probs_cond, rng)
 
        c_samples = ints_to_bits_concat(samples_site, bits=3, msb_first=True).T
        Trow = [cur_row[i][:,c_samples[i],:] for i in range(3)] 
        Tprev = tn.einsum('nxyz,xna,ynb,znc -> nabc',Tprev,Trow[0],Trow[1],Trow[2])
        prob_prev = (Tprev**2).sum(dim=(1,2,3))  
        bitstrings = tn.concatenate((bitstrings, samples_site.reshape(1, Nsamples)), axis=0)

    if return_joint:
        return bitstrings, prob_prev

    return bitstrings

def marginal_probs_cw(core, Tprev):

    ampls = tn.einsum('ncz,cmy->nmyz', Tprev, core)
    probs = (ampls**2).sum(dim=(2,3))
    return probs

def sample_from_comb_cw(comb, Nsamples, rng = None, return_joint=False):

    if rng == None:
        rng = ensure_torch_rng()

    D = len(comb)
    L = len(comb[0])
    combc = _clone_comb(comb)
    #right orthogonalize the cores

    for i in range(D):
        combc[i] = rcf(combc[i])
    
    root =  rcf(get_roott(combc))
    update_comb(combc,root)

    #first column
    mps0 = tn.einsum('xyz,ybc -> bcz',root[0],combc[0][1])
    p0 = (mps0**2).sum(dim=(1,2))
    p0 = tn.clamp(p0, min=0)
    p0 /= p0.sum()
    first_samples = tn.multinomial(p0, num_samples=Nsamples,replacement=True, generator=rng)
    Tprev = mps0[first_samples]        # (Nsamples, D1)
    prob_prev = (Tprev**2).sum(dim=(1,2))        # current accumulated probability
    bitstrings = first_samples.reshape(1, Nsamples)

    # iterate over rows
    for j in range(2,L):
        cur= combc[0][j] 
        probs_cur = marginal_probs_cw(cur,Tprev)
        probs_cond = probs_cur / prob_prev[:, None]
        samples_site = sample_r(probs_cond, rng)
        Trow = cur[:,samples_site,:]
        Tprev = tn.einsum('ncz,cny -> nyz',Tprev,Trow)
        prob_prev = (Tprev**2).sum(dim=(1,2))  
        bitstrings = tn.concatenate((bitstrings, samples_site.reshape(1, Nsamples)), axis=0)

    #other branches
    for i in range(1,3):
        Tprev = tn.einsum('ncz,zay->nay',Tprev,root[i])

        for j in range(1,L):
            cur= combc[i][j] 
            probs_cur = marginal_probs_cw(cur,Tprev)
            probs_cond = probs_cur / prob_prev[:, None]
            samples_site = sample_r(probs_cond, rng)
            Trow = cur[:,samples_site,:]
            Tprev = tn.einsum('ncz,cny -> nyz',Tprev,Trow)
            prob_prev = (Tprev**2).sum(dim=(1,2))  
            bitstrings = tn.concatenate((bitstrings, samples_site.reshape(1, Nsamples)), axis=0)

    if return_joint:
        return bitstrings, prob_prev

    return bitstrings



def apply_mask_b(branch, indices):
    d = len(branch) -1 
    M = len(indices)

    result = tn.ones((1,M), dtype=tn.float64)
    for i in range(d,0,-1):
        result = tn.einsum('jik,ki->ji',branch[i][:,indices[:,i-1],:],result)

    return tn.squeeze(result)

def apply_mask_c(comb,samples, sample_type='column'):

    root = get_roott(comb)
    #fr = fuse_root(root)
    if sample_type == 'row':
        bsamps = ints_to_bits_concat(samples, bits=3, msb_first=True).T
        mx = apply_mask_b(comb[0], bsamps[0]) 
        my = apply_mask_b(comb[1], bsamps[1]) 
        mz = apply_mask_b(comb[2], bsamps[2]) 
        r1 = tn.einsum('abc, bx -> axc', root[0],mx)
        r2 = tn.einsum('abc, bx -> axc', root[1],my)
        r3 = tn.einsum('abc, bx -> axc', root[2],mz)
        return tn.einsum('uxv,vxw,wxy ->x',r1,r2,r3)
        
    elif sample_type== 'seq':
        mx = apply_mask_b(comb[0], samples.T[:,0::3]) 
        my = apply_mask_b(comb[1], samples.T[:,1::3]) 
        mz = apply_mask_b(comb[2], samples.T[:,2::3]) 
        r1 = tn.einsum('abc, bx -> axc', root[0],mx)
        r2 = tn.einsum('abc, bx -> axc', root[1],my)
        r3 = tn.einsum('abc, bx -> axc', root[2],mz)
        return tn.einsum('uxv,vxw,wxy ->x',r1,r2,r3)
    
    elif sample_type == 'column':
        samples = samples.reshape(3,len(comb[0])-1,-1)
        mx = apply_mask_b(comb[0], samples[0].T) 
        my = apply_mask_b(comb[1], samples[1].T) 
        mz = apply_mask_b(comb[2], samples[2].T) 
        r1 = tn.einsum('abc, bx -> axc', root[0],mx)
        r2 = tn.einsum('abc, bx -> axc', root[1],my)
        r3 = tn.einsum('abc, bx -> axc', root[2],mz)
        return tn.einsum('uxv,vxw,wxy ->x',r1,r2,r3)
    

def _seed64(seq: np.random.SeedSequence | None) -> int:
    if seq is None:
        return secrets.randbits(64)  # non-deterministic, independent
    # spawn a child, take 64 bits for torch.Generator
    child = seq.spawn(1)[0]
    return int(child.generate_state(1, dtype=np.uint64)[0])

def rand_comb(N, R, r, *, var=1.0, dtype=tn.float64, device=None, generator: tn.Generator | None = None):
    """
    Draws one set of random 'cores'. If `generator` is None, a fresh torch.Generator
    with a unique 64-bit seed is used (independent draw).
    """
    d = len(N)
    D = len(r) - 1

    v1 = (var / np.prod(R)) ** (1.0 / d)
    v2 = (var / np.prod(r)) ** (1.0 / D)

    g = generator
    if g is None:
        g = tn.Generator(device=device)
        g.manual_seed(_seed64(None))

    cores = []
    for j in range(D):
        branch  = [None] * (d + 1)
        # first core in the branch
        branch[0] = tn.randn((1, r[j] * r[j+1], R[0]), dtype=dtype, device=device, generator=g) * np.sqrt(v2)
        # remaining d cores
        for i in range(d):
            shape = (R[i], N[i][0], N[i][1], R[i+1]) if isinstance(N[i], tuple) else (R[i], N[i], R[i+1])
            branch[i+1] = tn.randn(shape, dtype=dtype, device=device, generator=g) * np.sqrt(v1)
        cores.append(branch)
    return cores

def rand_comb(N, R,r, var = 1.0, dtype = tn.float64, device = None):

    d = len(N)
    D = len(r)-1
    v1 = var / np.prod(R) 
    v = v1**(1/(d))

    v2  = var / np.prod(r)
    v2 = v2**(1/D)
    cores = []
    for j in range(D):
        branch  = [None] * (d+1)
        branch[0] = tn.randn([1,r[j]*r[j+1],R[0]], dtype = dtype, device = device)*np.sqrt(v2)
        for i in range(d):
            branch[i+1] = tn.randn([R[i],N[i][0],N[i][1],R[i+1]] if isinstance(N[i],tuple) else [R[i],N[i],R[i+1]], dtype = dtype, device = device)*np.sqrt(v)
        cores.append(branch)
    return cores

def gen_noisy_combs(Nscales,nrank,rrank=None,cscale=2,var=1,dims=3,pdim=2):
    if rrank is None:
        rrank = nrank
    noises = []
    for scale in range(cscale,Nscales):
        rcomb = rand_comb([pdim]*scale,[nrank]*scale+[1],[1]+[rrank]*(dims-1)+[1],var=var) 
        rcomb = tree_compress(rcomb,eps=1e-10)
        noises.append(rcomb)
    return noises

def comb_mul(a,comb):
    comb[0][0] = a*comb[0][0]
    return comb

def gen_comb_cascade_3d(Nscales=10, nrank=10, rrank=20,levels=None, seed=None,epsilon=0.1 , method = 'cubic', eps=1e-10,var=1,order=1, field='stream'):

    if levels is None:
        levels = Nscales
    if seed is not None:
        tn.manual_seed(seed)

    # Initialize the stream function uniformly.
    noisex = gen_noisy_combs(levels,nrank,rrank=rrank,var=var,cscale=2,dims=3,pdim=2)
    noisey = gen_noisy_combs(levels,nrank,rrank=rrank,var=var,cscale=2,dims=3,pdim=2)
    noisez = gen_noisy_combs(levels,nrank,rrank=rrank,var=var,cscale=2,dims=3,pdim=2)

    if field == 'stream':
        base_scaling = 2.0 ** (-4.0 / 3.0)
        for i in range(2,levels):
            scale = i 
            j = i - 2
            am = (base_scaling**scale) * epsilon 
            # Interpolate to full resolution using linear interpolation.
            if method =='linear':
                raise Exception('Not implemented')

            # Interpolate to full resolution using cuadratic interpolation.
            elif method == 'cuadratic':
                raise Exception('Not implemented')

            # Interpolate to full resolution using splines interpolation

            elif method == 'cubic':
                smooth_mpsx =  comb_mul(am, comb_interpolate_p(noisex[j],Nscales, eps=eps, order=order) )
                smooth_mpsy = comb_mul(am, comb_interpolate_p(noisey[j],Nscales, eps=eps, order=order) )
                smooth_mpsz = comb_mul(am, comb_interpolate_p(noisez[j],Nscales, eps=eps, order=order) )

            elif method == 'quadratic':
                smooth_mpsx = comb_mul(am, comb_interpolate_p_quad(noisex[j],Nscales, eps=eps))
                smooth_mpsy = comb_mul(am, comb_interpolate_p_quad(noisey[j],Nscales, eps=eps))
                smooth_mpsz = comb_mul(am, comb_interpolate_p_quad(noisez[j],Nscales, eps=eps))

            if i==2:
                Ax = smooth_mpsx
                Ay = smooth_mpsy
                Az = smooth_mpsz
            else:
                Ax = comb_sum(Ax,smooth_mpsx)
                Ax = tree_compress(Ax,eps=eps)
                Ay = comb_sum(Ay,smooth_mpsy)
                Ay = tree_compress(Ay,eps=eps)
                Az = comb_sum(Az,smooth_mpsz)
                Az = tree_compress(Az,eps=eps)

        return Az, Ay, Az

    elif field == 'velocity':
        base_scaling = 2.0 ** (-1.0/3.0)
        #pyAx = 0
        #pzAx = 0
        #pxAy = 0
        #pzAy = 0
        #pxAz = 0
        #pyAz = 0

        for i in range(2,levels):
            scale = i 
            j = i -2 
            am = (base_scaling**scale) * epsilon 
            # Interpolate to full resolution using linear interpolation.
            if method =='linear':
                raise Exception('Not implemented')

            # Interpolate to full resolution using cuadratic interpolation.
            elif method == 'cuadratic':
                raise Exception('Not implemented')

            # Interpolate to full resolution using splines interpolation

            elif method == 'cubic':
                pyAx = comb_mul(am, comb_interpolate_p(noisex[j],Nscales, eps=eps, order=order,p_derivative=[0,1,0]) )
                pzAx =  comb_mul(am, comb_interpolate_p(noisex[j],Nscales, eps=eps, order=order,p_derivative=[0,0,1]) )

                pxAy =  comb_mul(am, comb_interpolate_p(noisey[j],Nscales, eps=eps, order=order,p_derivative=[1,0,0]) )
                pzAy =  comb_mul(am, comb_interpolate_p(noisey[j],Nscales, eps=eps, order=order,p_derivative=[0,0,1]) )

                pxAz =  comb_mul(am, comb_interpolate_p(noisez[j],Nscales, eps=eps, order=order,p_derivative=[1,0,0]) )
                pyAz =  comb_mul(am, comb_interpolate_p(noisez[j],Nscales, eps=eps, order=order,p_derivative=[0,1,0]) )
            elif method == 'quadratic':
                pyAx =   comb_mul(am, comb_interpolate_p_quad(noisex[j],Nscales, eps=eps,p_derivative=[0,1,0]) )
                pzAx =   comb_mul(am, comb_interpolate_p_quad(noisex[j],Nscales, eps=eps,p_derivative=[0,0,1]) )

                pxAy =  comb_mul(am, comb_interpolate_p_quad(noisey[j],Nscales, eps=eps,p_derivative=[1,0,0]) )
                pzAy =  comb_mul(am, comb_interpolate_p_quad(noisey[j],Nscales, eps=eps,p_derivative=[0,0,1]) )

                pxAz =  comb_mul(am, comb_interpolate_p_quad(noisez[j],Nscales, eps=eps,p_derivative=[1,0,0]) )
                pyAz =  comb_mul(am, comb_interpolate_p_quad(noisez[j],Nscales, eps=eps,p_derivative=[0,1,0]) )
            
            
            if i==2:
                vx = comb_sum(pyAz, comb_mul(-1,pzAy))
                vy = comb_sum(pzAx, comb_mul(-1,pxAz))
                vz = comb_sum(pxAy, comb_mul(-1,pyAx))
            else:
                vx = comb_sum(vx, comb_sum(pyAz, comb_mul(-1,pzAy)))
                vx = tree_compress_2(vx,eps=eps)
                vy = comb_sum(vy, comb_sum(pzAx, comb_mul(-1,pxAz)))
                vy = tree_compress_2(vy,eps=eps)
                vz = comb_sum(vz, comb_sum(pxAy, comb_mul(-1,pyAx)))
                vz = tree_compress_2(vz,eps=eps)


        #vy = tree_compress(vy,eps=eps)
        #vx = tree_compress(vx,eps=eps)
        #vz = tree_compress(vz,eps=eps)

        return vx, vy, vz
    else:
        raise Exception('Field not implemented')