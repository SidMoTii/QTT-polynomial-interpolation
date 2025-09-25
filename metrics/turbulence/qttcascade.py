import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch as tn
import torchtt as tntt
from src.qtt_interpolation.utils import *


def d_coeff_mat(M, order=0):
    if order == 0:
        return M
    elif order == 1:
        return M[:, 1:] * tn.tensor([1, 2, 3], dtype=M.dtype)
    elif order == 2:
        return M[:, 2:] * tn.tensor([2, 6], dtype=M.dtype)
    else:
        raise ValueError("Only derivative orders 0, 1, and 2 are supported.")
    

def qtt_skcubic3d_p(mps, fscale, eps = 1e-15,epsilon = 0.1, order = 1, p_derivative = [0,0,0]):


    ni = len(mps.N)//3
    nc = fscale - ni 

    id0 = I_qtt(ni)
    ones = tntt.ones([2]*ni)
    onesf = tntt.ones([2]*nc)

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


    Mx = d_coeff_mat(Mkct, p_derivative[0]) 
    My = d_coeff_mat(Mkct, p_derivative[1])
    Mz = d_coeff_mat(Mkct, p_derivative[2])
    polsx = [tntt.kron( ones, zukron3(tntt.TT(qtt_polynomial_cores(Mx[i], nc)) , onesf,onesf) ).round(eps) for i in range(4)]
    polsy = [tntt.kron( ones, zukron3( onesf , tntt.TT( qtt_polynomial_cores(My[i], nc)), onesf )).round(eps) for i in range(4)]
    polsz = [tntt.kron( ones, zukron3( onesf , onesf , tntt.TT( qtt_polynomial_cores(Mz[i], nc)) )).round(eps) for i in range(4)]
    
    zones3 = zukron3(onesf, onesf, onesf)

    oneso = tntt.eye([2]*nc)
    id = tntt.kron(id0,oneso)
    sm1 = tntt.kron(sm10,oneso)
    Om1 = tntt.kron(Om10,oneso)
    O2 = tntt.kron(O20,oneso)

    mps = tntt.kron(mps, zones3 )
    #interpolate y,
    f_ky = (zukron3(id,Om1,id) @ mps) * ( polsy[0] ) + ( mps * polsy[1] ) + ( zukron3(id,sm1,id) @ mps ) * (polsy[2]) + ( zukron3(id,O2,id) @ mps ) * (polsy[3]) 
    f_ky = f_ky.round(eps)

    #interpolate x
    f_kxy =  (zukron3(Om1,id,id) @ f_ky) *( polsx[0] ) + ( f_ky *  polsx[1] ) + ( zukron3(sm1,id,id) @ f_ky ) * (polsx[2]) +  ( zukron3(O2,id,id) @ f_ky ) * ( polsx[3])
    f_kxy = f_kxy.round(eps)

    #interpolate z
    f_kcs = (zukron3(id,id,Om1) @ f_kxy) * ( polsz[0] ) + ( f_kxy * polsz[1] ) + ( zukron3(id,id,sm1) @ f_kxy ) * (polsz[2]) + ( zukron3(id,id,O2) @ f_kxy ) * (polsz[3]) 
    f_kcs = f_kcs.round(eps)

    return f_kcs

def d_coeff_mat2(M, order=0):
    if order == 0:
        return M
    elif order == 1:
        return M[:, 1:] * tn.tensor([1, 2], dtype=M.dtype)
    elif order == 2:
        return M[:, 2:] * tn.tensor([2], dtype=M.dtype)
    else:
        raise ValueError("Only derivative orders 0, 1, and 2 are supported.")
    
def qtt_skquad3d_p(mps, fscale, eps = 1e-15,epsilon = 0.1,boundary='lineare',p_derivative=[0,0,0]):

    ni = len(mps.N)//3
    nc = fscale - ni 
    # Final corrected B-spline matrices (4 samples × 3 monomial terms)
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

    id0 = I_qtt(ni)
    sm10 = R_qtt( 2**ni -1, ni)
    ones = tntt.ones([1]*ni)
    onesf = tntt.ones([2]*nc)
    sm10 = P_qtt( 2**ni -1, ni)
    O20 = P_qtt( 2**ni -2, ni)
    Om10 = P_qtt( 1, ni)



    def qtt2pol(i,d=0):
        m1 = d_coeff_mat2(mk1, d)  
        m2 = d_coeff_mat2(mk2, d)  
        phi0 = hs(nc,'1')*tntt.TT(qtt_polynomial_cores(m1[i], nc)) 
        phi1 = hs(nc,'0')*tntt.TT(qtt_polynomial_cores(m2[i], nc)) 
        return (phi0+phi1).round(eps)
    

    polsx = [tntt.kron( ones, zukron3(qtt2pol(i,p_derivative[0]), onesf,onesf) ).round(eps) for i in range(4)]
    polsy = [tntt.kron( ones, zukron3( onesf , qtt2pol(i,p_derivative[1]), onesf )).round(eps) for i in range(4)]
    polsz = [tntt.kron( ones, zukron3( onesf , onesf , qtt2pol(i,p_derivative[2]) )).round(eps) for i in range(4)]
    
    zones = zukron3(onesf, onesf, onesf)

    oneso = tntt.eye([2]*nc)
    id = tntt.kron(id0,oneso)
    sm1 = tntt.kron(sm10,oneso)
    Om1 = tntt.kron(Om10,oneso)
    O2 = tntt.kron(O20,oneso)

    oneso = tntt.eye([2]*nc)
    id = tntt.kron(id0,oneso)
    sm1 = tntt.kron(sm10,oneso)
    Om1 = tntt.kron(Om10,oneso)
    O2 = tntt.kron(O20,oneso)
    mps = tntt.kron(mps, zones )

    #interpolate y
    f_ky = (zukron3(id,Om1,id) @ mps) * ( polsy[0] ) + ( mps * polsy[1] ) + ( zukron3(id,sm1,id) @ mps ) * (polsy[2]) + ( zukron3(id,O2,id) @ mps ) * (polsy[3]) 
    f_ky = f_ky.round(eps)

    #interpolate x
    f_kxy =  (zukron3(Om1,id,id) @ f_ky) *( polsx[0] ) + ( f_ky *  polsx[1] ) + ( zukron3(sm1,id,id) @ f_ky ) * (polsx[2]) +  ( zukron3(O2,id,id) @ f_ky ) * ( polsx[3])
    f_kxy = f_kxy.round(eps)

    #interpolate z
    f_kcs = (zukron3(id,id,Om1) @ f_kxy) * ( polsz[0] ) + ( f_kxy * polsz[1] ) + ( zukron3(id,id,sm1) @ f_kxy ) * (polsz[2]) + ( zukron3(id,id,O2) @ f_kxy ) * (polsz[3]) 
    f_kcs = f_kcs.round(eps)

    return f_kcs

def gen_noisy_qtt(Nscales,nrank,cscale=2,var=1,dims=3,pdim=2):
    noises = []
    for scale in range(cscale,Nscales):
        noises.append(tntt.randn([pdim]*dims*scale,[1]+[nrank]*(dims*scale-1)+[1],var=var).to_qtt().round(rmax=5) ) 
    return noises

def mx(coesf):
        return [0] + coesf 
def mx1(coesf):
    mc = [-x for x in coesf] + [0]
    return np.array(mc) + np.array([0] + coesf )

def p_der(c, order=1):
    d = list(c[:])
    for _ in range(order):
        d = [i*d[i] for i in range(1, len(d))]
    return d or [0]

def qtt_pnoise_2d(qttnoise, f_scale,fdegree =3,eps=1e-10, p_derivative = [0,0] ):
    #initial perlins noise 32 x 32
    nnoise = len(qttnoise[0].N)//2
    nscales = f_scale
    next = nscales - nnoise


    #improve random qtts
    if len(qttnoise)==3:
        h=True
        nx,ny,hs = qttnoise
    else:
        h=False
        nx,ny = qttnoise

    # generate 4 corner contributions
    I = tn.eye(4,dtype=tn.float64)
    vs = [tntt.TT(I[i],[4]) for i in range(4)]
    S1 = P_qtt(2**nnoise-1,nnoise)
    S0 = I_qtt(nnoise)
    S1x = zukron(S1,S0)
    S1y = zukron(S0,S1)
    S1xy = zukron(S1,S1)

    c1 = tntt.kron(nx ,vs[0])
    c2 = tntt.kron( S1x@nx ,vs[1])
    c3 = tntt.kron( S1y@nx,vs[2])
    c4 = tntt.kron( S1xy@nx,vs[3])

    coarsex = c1+c2+c3+c4
    coarsex = coarsex.round(eps)

    c1 = tntt.kron(ny ,vs[0])
    c2 = tntt.kron( S1x@ny ,vs[1])
    c3 = tntt.kron( S1y@ny,vs[2])
    c4 = tntt.kron( S1xy@ny,vs[3])

    coarsey = c1+c2+c3+c4
    coarsey = coarsey.round(eps)

    if fdegree == 3 :
        coefs = [0,0,3,-2]
        icoefs = [1,0,-3,2]
    elif fdegree == 5:
        coefs = [0,0,0,10,-15,6]
        icoefs = [1,0,0,-10,15,-6] 
    

    psi1_x = tntt.TT( qtt_polynomial_cores( p_der(coefs,order=p_derivative[0]) ,next) )
    psi0_x = tntt.TT( qtt_polynomial_cores(p_der(icoefs,order=p_derivative[0]) ,next) )
    psi0x_x = tntt.TT( qtt_polynomial_cores( p_der(mx(icoefs),order=p_derivative[0])  ,next) )
    psi1xm_x = tntt.TT( qtt_polynomial_cores( p_der(mx1(coefs),order=p_derivative[0]) ,next) )

    psi1_y = tntt.TT( qtt_polynomial_cores( p_der(coefs,order=p_derivative[1]) ,next) )
    psi0_y = tntt.TT( qtt_polynomial_cores(p_der(icoefs,order=p_derivative[1]) ,next) )
    psi0x_y = tntt.TT( qtt_polynomial_cores( p_der(mx(icoefs),order=p_derivative[1])  ,next) )
    psi1xm_y = tntt.TT( qtt_polynomial_cores( p_der(mx1(coefs),order=p_derivative[1]) ,next) )

    # do x
    P1 = tntt.kron(vs[0],zukron(psi0x_x,psi0_y))
    P2 = tntt.kron(vs[1],zukron(psi1xm_x,psi0_y))
    P3 = tntt.kron(vs[2],zukron(psi0x_x,psi1_y))
    P4 = tntt.kron(vs[3],zukron(psi1xm_x,psi1_y))

    Pext = (P1+P2+P3+P4).round(eps)

    perTTx = connect(coarsex,Pext,pd=4)

    # do y
    P1 = tntt.kron(vs[0],zukron(psi0_x,psi0x_y))
    P2 = tntt.kron(vs[1],zukron(psi1_x,psi0x_y))
    P3 = tntt.kron(vs[2],zukron(psi0_x,psi1xm_y))
    P4 = tntt.kron(vs[3],zukron(psi1_x,psi1xm_y))

    Pext = (P1+P2+P3+P4).round(eps)

    perTTy = connect(coarsey,Pext,pd=4)

    perTT0 = (perTTx + perTTy).round(eps)

    if h :
        c1 = tntt.kron(hs ,vs[0])
        c2 = tntt.kron( S1x@hs ,vs[1])
        c3 = tntt.kron( S1y@hs,vs[2])
        c4 = tntt.kron( S1xy@hs,vs[3])
        hcoarse = c1+c2+c3+c4
        hcoarse = hcoarse.round(eps)
        # do h
        P1 = tntt.kron(vs[0],zukron(psi0_x,psi0_y))
        P2 = tntt.kron(vs[1],zukron(psi1_x,psi0_y))
        P3 = tntt.kron(vs[2],zukron(psi0_x,psi1_y))
        P4 = tntt.kron(vs[3],zukron(psi1_x,psi1_y))

        Pext = (P1+P2+P3+P4).round(eps)

        hTT = connect(hcoarse,Pext,pd=4)
        perTT0 = (perTT0 + hTT).round(eps)


    return perTT0

def fractal_noise(mps,noctaves,alpha=1/2,eps=1e-10, dims=2):
    fnoise = mps
    for i in range(1,noctaves):
        if i == 1:
            oc = reduceg(mps,0,-1)
            for _ in range(dims-1):
                oc = reduceg(oc,0,-1)
            oc = oc.round(eps)
            
        else:
            for _ in range(dims):
                oc = reduceg(oc,0,-1)
            oc = oc.round(eps)
        oc = alpha*tntt.kron(tntt.ones([2,2]),oc)
        fnoise += oc
    fnoise = fnoise.round(eps)
    return fnoise

def qtt_pnoise_3d(qttnoise, f_scale,fdegree =5,eps=1e-10, p_derivative = [0,0,0] ):
    #initial perlins noise 32 x 32
    nnoise = len(qttnoise[0].N)//3
    nscales = f_scale
    next = nscales - nnoise

    #improve random qtts
    nx,ny,nz = qttnoise

    # generate 8 corner contributions
    I = tn.eye(8,dtype=tn.float64)
    vs = [tntt.TT(I[i],[8]) for i in range(8)]
    S1 = P_qtt(2**nnoise-1,nnoise)
    S0 = I_qtt(nnoise)
    S1x = zukron3(S1,S0,S0)
    S1y = zukron3(S0,S1,S0)
    S1z = zukron3(S0,S0,S1)
    S1xy = zukron3(S1,S1,S0)
    S1xz = zukron3(S1,S0,S1)
    S1yz = zukron3(S0,S1,S1)
    S1xyz = zukron3(S1,S1,S1)

    # do x
    c1 = tntt.kron(nx ,vs[0])
    c2 = tntt.kron( S1x@nx ,vs[1])
    c3 = tntt.kron( S1y@nx,vs[2])
    c4 = tntt.kron( S1z@nx,vs[3])
    c5 = tntt.kron( S1xy@nx,vs[4])
    c6 = tntt.kron( S1xz@nx,vs[5])
    c7 = tntt.kron( S1yz@nx,vs[6])
    c8 = tntt.kron( S1xyz@nx,vs[7])

    coarsex = c1+c2+c3+c4+c5+c6+c7+c8
    coarsex = coarsex.round(eps)

    # do y
    c1 = tntt.kron(ny ,vs[0])
    c2 = tntt.kron( S1x@ny ,vs[1])
    c3 = tntt.kron( S1y@ny,vs[2])
    c4 = tntt.kron( S1z@ny,vs[3])
    c5 = tntt.kron( S1xy@ny,vs[4])
    c6 = tntt.kron( S1xz@ny,vs[5])
    c7 = tntt.kron( S1yz@ny,vs[6])
    c8 = tntt.kron( S1xyz@ny,vs[7])

    coarsey =  c1+c2+c3+c4+c5+c6+c7+c8
    coarsey = coarsey.round(eps)

    # do z
    c1 = tntt.kron(nz ,vs[0])
    c2 = tntt.kron( S1x@nz ,vs[1])
    c3 = tntt.kron( S1y@nz,vs[2])
    c4 = tntt.kron( S1z@nz,vs[3])
    c5 = tntt.kron( S1xy@nz,vs[4])
    c6 = tntt.kron( S1xz@nz,vs[5])
    c7 = tntt.kron( S1yz@nz,vs[6])
    c8 = tntt.kron( S1xyz@nz,vs[7])

    coarsez =  c1+c2+c3+c4+c5+c6+c7+c8
    coarsez = coarsez.round(eps)


    if fdegree == 3 :
        coefs = [0,0,3,-2]
        icoefs = [1,0,-3,2]
    elif fdegree == 5:
        coefs = [0,0,0,10,-15,6]
        icoefs = [1,0,0,-10,15,-6] 
    

    psi1_x = tntt.TT( qtt_polynomial_cores( p_der(coefs,order=p_derivative[0]) ,next) )
    psi0_x = tntt.TT( qtt_polynomial_cores(p_der(icoefs,order=p_derivative[0]) ,next) )
    psi0x_x = tntt.TT( qtt_polynomial_cores( p_der(mx(icoefs),order=p_derivative[0])  ,next) )
    psi1xm_x = tntt.TT( qtt_polynomial_cores( p_der(mx1(coefs),order=p_derivative[0]) ,next) )

    psi1_y = tntt.TT( qtt_polynomial_cores( p_der(coefs,order=p_derivative[1]) ,next) )
    psi0_y = tntt.TT( qtt_polynomial_cores(p_der(icoefs,order=p_derivative[1]) ,next) )
    psi0x_y = tntt.TT( qtt_polynomial_cores( p_der(mx(icoefs),order=p_derivative[1])  ,next) )
    psi1xm_y = tntt.TT( qtt_polynomial_cores( p_der(mx1(coefs),order=p_derivative[1]) ,next) )

    psi1_z = tntt.TT( qtt_polynomial_cores( p_der(coefs,order=p_derivative[2]) ,next) )
    psi0_z = tntt.TT( qtt_polynomial_cores(p_der(icoefs,order=p_derivative[2]) ,next) )
    psi0x_z = tntt.TT( qtt_polynomial_cores( p_der(mx(icoefs),order=p_derivative[2])  ,next) )
    psi1xm_z = tntt.TT( qtt_polynomial_cores( p_der(mx1(coefs),order=p_derivative[2]) ,next) )

    # do x
    P1 = tntt.kron(vs[0],zukron3(psi0x_x,psi0_y,psi0_z))
    P2 = tntt.kron(vs[1],zukron3(psi1xm_x,psi0_y,psi0_z))
    P3 = tntt.kron(vs[2],zukron3(psi0x_x,psi1_y,psi0_z))
    P4 = tntt.kron(vs[3],zukron3(psi1xm_x,psi1_y,psi0_z))
    P5 = tntt.kron(vs[4],zukron3(psi0x_x,psi0_y,psi1_z))
    P6 = tntt.kron(vs[5],zukron3(psi1xm_x,psi0_y,psi1_z))
    P7 = tntt.kron(vs[6],zukron3(psi0x_x,psi1_y,psi1_z))
    P8 = tntt.kron(vs[7],zukron3(psi1xm_x,psi1_y,psi1_z))

    Pext = (P1+P2+P3+P4+P5+P6+P7+P8).round(eps)

    perTTx = connect(coarsex,Pext,pd=8)

    # do y
    P1 = tntt.kron(vs[0],zukron3(psi0_x,psi0x_y,psi0_z))
    P2 = tntt.kron(vs[1],zukron3(psi1_x,psi0x_y,psi0_z))
    P3 = tntt.kron(vs[2],zukron3(psi0_x,psi1xm_y,psi0_z))
    P4 = tntt.kron(vs[3],zukron3(psi1_x,psi1xm_y,psi0_z))
    P5 = tntt.kron(vs[4],zukron3(psi0_x,psi0x_y,psi1_z))
    P6 = tntt.kron(vs[5],zukron3(psi1_x,psi0x_y,psi1_z))
    P7 = tntt.kron(vs[6],zukron3(psi0_x,psi1xm_y,psi1_z))
    P8 = tntt.kron(vs[7],zukron3(psi1_x,psi1xm_y,psi1_z))

    Pext = (P1+P2+P3+P4+P5+P6+P7+P8).round(eps)

    perTTy = connect(coarsey,Pext,pd=8)

    # do y
    P1 = tntt.kron(vs[0],zukron3(psi0_x,psi0_y,psi0x_z))
    P2 = tntt.kron(vs[1],zukron3(psi1_x,psi0_y,psi0x_x))
    P3 = tntt.kron(vs[2],zukron3(psi0_x,psi1_y,psi0x_x))
    P4 = tntt.kron(vs[3],zukron3(psi1_x,psi1_y,psi0x_x))
    P5 = tntt.kron(vs[4],zukron3(psi0_x,psi0_y,psi1xm_z))
    P6 = tntt.kron(vs[5],zukron3(psi1_x,psi0_y,psi1xm_z))
    P7 = tntt.kron(vs[6],zukron3(psi0_x,psi1_y,psi1xm_z))
    P8 = tntt.kron(vs[7],zukron3(psi1_x,psi1_y,psi1xm_z))

    Pext = (P1+P2+P3+P4+P5+P6+P7+P8).round(eps)

    perTTz = connect(coarsey,Pext,pd=8)

    perTT0 = (perTTx + perTTy + perTTz).round(eps)
    return perTT0

def gen_qTT_cascade_3d_perlin(Nscales=10, nrank=10, noise_scale=5,noctaves=5,  levels=None, seed=None,epsilon=0.1 , forder = 3, eps=1e-10,var=1,order=1, field='stream'):

    if levels is None:
        levels = Nscales
    if seed is not None:
        tn.manual_seed(seed)

    # Initialize gradients.
    noise_scale *=3
    noisex =   tntt.randn([2]*noise_scale,[1]+[nrank]*(noise_scale-1)+[1],var=var)
    noisey =   tntt.randn([2]*noise_scale,[1]+[nrank]*(noise_scale-1)+[1],var=var)
    noisez =   tntt.randn([2]*noise_scale,[1]+[nrank]*(noise_scale-1)+[1],var=var)

    if field == 'stream':

        base_scaling = 2.0 ** (-4.0 / 3.0)
        # Interpolate to full resolution using linear interpolation.
        if method =='linear':
            raise Exception('Not implemented')

        # Interpolate to full resolution using cuadratic interpolation.
        elif method == 'cuadratic':
            raise Exception('Not implemented')

        # Interpolate to full resolution using splines interpolation

        elif method == 'skc':
            smooth_mpsx =   qtt_skcubic3d_p(noisex,Nscales, eps=eps, epsilon=epsilon, order=order)
            smooth_mpsy =  qtt_skcubic3d_p(noisey,Nscales, eps=eps, epsilon=epsilon, order=order)
            smooth_mpsz =  qtt_skcubic3d_p(noisez,Nscales, eps=eps, epsilon=epsilon, order=order)
        
        elif method == 'skq':
            smooth_mpsx =  qtt_skquad3d_p(noisex,Nscales, eps=eps, epsilon=epsilon)
            smooth_mpsy =  qtt_skcubic3d_p(noisey,Nscales, eps=eps, epsilon=epsilon)
            smooth_mpsz =  qtt_skquad3d_p(noisez,Nscales, eps=eps, epsilon=epsilon)

        r_mpsx = smooth_mpsx
        r_mpsy = smooth_mpsy
        r_mpsz = smooth_mpsz

        Ax = smooth_mpsx
        Ay = smooth_mpsy
        Az = smooth_mpsz

        for k in range(1,noctaves):

            r_mpsx = reduce(reduce(reduce(r_mpsx,0),0),0)
            re_mpsx = tntt.kron(tntt.ones([2]*3*k),r_mpsx)
            Ax += base_scaling**k * re_mpsx

            r_mpsy = reduce(reduce(reduce(r_mpsy,0),0),0)
            re_mpsy = tntt.kron(tntt.ones([2]*3*k),r_mpsy)
            Ay += base_scaling**k * re_mpsy

            r_mpsz = reduce(reduce(reduce(r_mpsz,0),0),0)
            re_mpsz = tntt.kron(tntt.ones([2]*3*k),r_mpsz)
            Az += base_scaling**k * re_mpsz
            

        Ax = Ax.round(eps)
        Ay = Ay.round(eps)
        Az = Az.round(eps)

        return Az, Ay, Az

    elif field == 'velocity':
        base_scaling = 2.0 ** (-1.0/3.0)



        # Interpolate to full resolution using linear interpolation.
        if method =='linear':
            raise Exception('Not implemented')

        # Interpolate to full resolution using cuadratic interpolation.
        elif method == 'cuadratic':
            raise Exception('Not implemented')

        # Interpolate to full resolution using splines interpolation

        elif method == 'skc':
            pyAx =  qtt_skcubic3d_p(noisex,Nscales,  eps=eps, epsilon=epsilon, order=order, p_derivative=[0,1,0])
            pzAx =  qtt_skcubic3d_p(noisex,Nscales,  eps=eps, epsilon=epsilon, order=order,  p_derivative=[0,0,1])

            pxAy =  qtt_skcubic3d_p(noisey,Nscales,  eps=eps, epsilon=epsilon, order=order,  p_derivative=[1,0,0])
            pzAy =  qtt_skcubic3d_p(noisey,Nscales,  eps=eps, epsilon=epsilon, order=order,  p_derivative=[0,0,1])

            pxAz =  qtt_skcubic3d_p(noisez,Nscales,  eps=eps, epsilon=epsilon, order=order,  p_derivative=[1,0,0])
            pyAz =  qtt_skcubic3d_p(noisez,Nscales,  eps=eps, epsilon=epsilon, order=order,  p_derivative=[0,1,0])
        elif method == 'skq':
            pyAx =   qtt_skquad3d_p(noisex,Nscales,  eps=eps, epsilon=epsilon,  p_derivative=[0,1,0])
            pzAx =   qtt_skquad3d_p(noisex,Nscales,  eps=eps, epsilon=epsilon,  p_derivative=[0,0,1])

            pxAy =  qtt_skquad3d_p(noisey,Nscales,  eps=eps, epsilon=epsilon,  p_derivative=[1,0,0])
            pzAy =  qtt_skquad3d_p(noisey,Nscales,  eps=eps, epsilon=epsilon,  p_derivative=[0,0,1])

            pxAz =  qtt_skquad3d_p(noisez,Nscales,  eps=eps, epsilon=epsilon,  p_derivative=[1,0,0])
            pyAz =  qtt_skquad3d_p(noisez,Nscales,  eps=eps, epsilon=epsilon,  p_derivative=[0,1,0])

        vx = pyAz - pzAy
        vy = (-1)*pxAz + pzAx
        vz = pxAy - pyAx 

        r_mpsx = vx
        r_mpsy = vy
        r_mpsz = vz

        for k in range(1,noctaves):
            print('octave',k)
            r_mpsx = reduce(reduce(reduce(r_mpsx,0),0),0)
            re_mpsx = tntt.kron(tntt.ones([2]*3*k),r_mpsx)
            vx += base_scaling**k * re_mpsx

            r_mpsy = reduce(reduce(reduce(r_mpsy,0),0),0)
            re_mpsy = tntt.kron(tntt.ones([2]*3*k),r_mpsy)
            vy += base_scaling**k * re_mpsy

            r_mpsz = reduce(reduce(reduce(r_mpsz,0),0),0)
            re_mpsz = tntt.kron(tntt.ones([2]*3*k),r_mpsz)
            vz += base_scaling**k * re_mpsz

        vy = vy.round(eps)
        vx = vx.round(eps)
        vz = vz.round(eps)

        return vx, vy, vz
    else:
        raise Exception('Field not implemented')

def gen_qTT_cascade_3d_CC(Nscales=10, nrank=10, noise_scale=5,noctaves=5,  levels=None, seed=None,epsilon=0.1 , method = 'skc', eps=1e-10,var=1,order=1, field='stream'):

    if levels is None:
        levels = Nscales
    if seed is not None:
        tn.manual_seed(seed)

    # Initialize the stream function uniformly.
    noise_scale *=3
    noisex =   tntt.randn([2]*noise_scale,[1]+[nrank]*(noise_scale-1)+[1],var=var)
    noisey =   tntt.randn([2]*noise_scale,[1]+[nrank]*(noise_scale-1)+[1],var=var)
    noisez =   tntt.randn([2]*noise_scale,[1]+[nrank]*(noise_scale-1)+[1],var=var)

    if field == 'stream':

        base_scaling = 2.0 ** (-4.0 / 3.0)
        # Interpolate to full resolution using linear interpolation.
        if method =='linear':
            raise Exception('Not implemented')

        # Interpolate to full resolution using cuadratic interpolation.
        elif method == 'cuadratic':
            raise Exception('Not implemented')

        # Interpolate to full resolution using splines interpolation

        elif method == 'skc':
            smooth_mpsx =   qtt_skcubic3d_p(noisex,Nscales, eps=eps, epsilon=epsilon, order=order)
            smooth_mpsy =  qtt_skcubic3d_p(noisey,Nscales, eps=eps, epsilon=epsilon, order=order)
            smooth_mpsz =  qtt_skcubic3d_p(noisez,Nscales, eps=eps, epsilon=epsilon, order=order)
        
        elif method == 'skq':
            smooth_mpsx =  qtt_skquad3d_p(noisex,Nscales, eps=eps, epsilon=epsilon)
            smooth_mpsy =  qtt_skcubic3d_p(noisey,Nscales, eps=eps, epsilon=epsilon)
            smooth_mpsz =  qtt_skquad3d_p(noisez,Nscales, eps=eps, epsilon=epsilon)

        r_mpsx = smooth_mpsx
        r_mpsy = smooth_mpsy
        r_mpsz = smooth_mpsz

        Ax = smooth_mpsx
        Ay = smooth_mpsy
        Az = smooth_mpsz

        for k in range(1,noctaves):

            r_mpsx = reduce(reduce(reduce(r_mpsx,0),0),0)
            re_mpsx = tntt.kron(tntt.ones([2]*3*k),r_mpsx)
            Ax += base_scaling**k * re_mpsx

            r_mpsy = reduce(reduce(reduce(r_mpsy,0),0),0)
            re_mpsy = tntt.kron(tntt.ones([2]*3*k),r_mpsy)
            Ay += base_scaling**k * re_mpsy

            r_mpsz = reduce(reduce(reduce(r_mpsz,0),0),0)
            re_mpsz = tntt.kron(tntt.ones([2]*3*k),r_mpsz)
            Az += base_scaling**k * re_mpsz
            

        Ax = Ax.round(eps)
        Ay = Ay.round(eps)
        Az = Az.round(eps)

        return Az, Ay, Az

    elif field == 'velocity':
        base_scaling = 2.0 ** (-1.0/3.0)



        # Interpolate to full resolution using linear interpolation.
        if method =='linear':
            raise Exception('Not implemented')

        # Interpolate to full resolution using cuadratic interpolation.
        elif method == 'cuadratic':
            raise Exception('Not implemented')

        # Interpolate to full resolution using splines interpolation

        elif method == 'skc':
            pyAx =  qtt_skcubic3d_p(noisex,Nscales,  eps=eps, epsilon=epsilon, order=order, p_derivative=[0,1,0])
            pzAx =  qtt_skcubic3d_p(noisex,Nscales,  eps=eps, epsilon=epsilon, order=order,  p_derivative=[0,0,1])

            pxAy =  qtt_skcubic3d_p(noisey,Nscales,  eps=eps, epsilon=epsilon, order=order,  p_derivative=[1,0,0])
            pzAy =  qtt_skcubic3d_p(noisey,Nscales,  eps=eps, epsilon=epsilon, order=order,  p_derivative=[0,0,1])

            pxAz =  qtt_skcubic3d_p(noisez,Nscales,  eps=eps, epsilon=epsilon, order=order,  p_derivative=[1,0,0])
            pyAz =  qtt_skcubic3d_p(noisez,Nscales,  eps=eps, epsilon=epsilon, order=order,  p_derivative=[0,1,0])
        elif method == 'skq':
            pyAx =   qtt_skquad3d_p(noisex,Nscales,  eps=eps, epsilon=epsilon,  p_derivative=[0,1,0])
            pzAx =   qtt_skquad3d_p(noisex,Nscales,  eps=eps, epsilon=epsilon,  p_derivative=[0,0,1])

            pxAy =  qtt_skquad3d_p(noisey,Nscales,  eps=eps, epsilon=epsilon,  p_derivative=[1,0,0])
            pzAy =  qtt_skquad3d_p(noisey,Nscales,  eps=eps, epsilon=epsilon,  p_derivative=[0,0,1])

            pxAz =  qtt_skquad3d_p(noisez,Nscales,  eps=eps, epsilon=epsilon,  p_derivative=[1,0,0])
            pyAz =  qtt_skquad3d_p(noisez,Nscales,  eps=eps, epsilon=epsilon,  p_derivative=[0,1,0])

        vx = pyAz - pzAy
        vy = (-1)*pxAz + pzAx
        vz = pxAy - pyAx 

        r_mpsx = vx
        r_mpsy = vy
        r_mpsz = vz

        for k in range(1,noctaves):
            print('octave',k)
            r_mpsx = reduce(reduce(reduce(r_mpsx,0),0),0)
            re_mpsx = tntt.kron(tntt.ones([2]*3*k),r_mpsx)
            vx += base_scaling**k * re_mpsx

            r_mpsy = reduce(reduce(reduce(r_mpsy,0),0),0)
            re_mpsy = tntt.kron(tntt.ones([2]*3*k),r_mpsy)
            vy += base_scaling**k * re_mpsy

            r_mpsz = reduce(reduce(reduce(r_mpsz,0),0),0)
            re_mpsz = tntt.kron(tntt.ones([2]*3*k),r_mpsz)
            vz += base_scaling**k * re_mpsz

        vy = vy.round(eps)
        vx = vx.round(eps)
        vz = vz.round(eps)

        return vx, vy, vz
    else:
        raise Exception('Field not implemented')



def gen_qTT_cascade_3d(Nscales=10, nrank=10, levels=None, seed=None,epsilon=0.1 , method = 'skc', eps=1e-10,var=1,order=1, field='stream'):

    if levels is None:
        levels = Nscales
    if seed is not None:
        tn.manual_seed(seed)

    # Initialize the stream function uniformly.
    noisex = gen_noisy_qtt(levels,nrank,var=var,cscale=2,dims=1,pdim=8)
    noisey = gen_noisy_qtt(levels,nrank,var=var,cscale=2,dims=1,pdim=8)
    noisez = gen_noisy_qtt(levels,nrank,var=var,cscale=2,dims=1,pdim=8)

    if field == 'stream':
        Ax = 0
        Ay = 0
        Az = 0
        base_scaling = 2.0 ** (-4.0 / 3.0)
        for i in range(len(noisex)):
            scale = i + 2
            am = (base_scaling**scale) * epsilon 
            # Interpolate to full resolution using linear interpolation.
            if method =='linear':
                raise Exception('Not implemented')

            # Interpolate to full resolution using cuadratic interpolation.
            elif method == 'cuadratic':
                raise Exception('Not implemented')

            # Interpolate to full resolution using splines interpolation

            elif method == 'skc':
                smooth_mpsx =  am * qtt_skcubic3d_p(noisex[i],Nscales, eps=eps, epsilon=epsilon, order=order)
                smooth_mpsy = am * qtt_skcubic3d_p(noisey[i],Nscales, eps=eps, epsilon=epsilon, order=order)
                smooth_mpsz = am * qtt_skcubic3d_p(noisez[i],Nscales, eps=eps, epsilon=epsilon, order=order)
            
            elif method == 'skq':
                smooth_mpsx = am * qtt_skquad3d_p(noisex[i],Nscales, eps=eps, epsilon=epsilon)
                smooth_mpsy = am * qtt_skcubic3d_p(noisey[i],Nscales, eps=eps, epsilon=epsilon)
                smooth_mpsz = am * qtt_skquad3d_p(noisez[i],Nscales, eps=eps, epsilon=epsilon)

            Ax += smooth_mpsx
            Ax = Ax.round(eps)

            Ay += smooth_mpsy
            Ay = Ay.round(eps)

            Az += smooth_mpsz
            Az = Az.round(eps)

        return Az, Ay, Az

    elif field == 'velocity':
        base_scaling = 2.0 ** (-1.0/3.0)
        #pyAx = 0
        #pzAx = 0
        #pxAy = 0
        #pzAy = 0
        #pxAz = 0
        #pyAz = 0
        vx = 0 #epsilon* (base_scaling/2)**Nscales * tntt.randn([8]*Nscales,[1]+[nrank]*(Nscales-1)+[1],var=var)
        vy = 0 #epsilon* (base_scaling/2)**Nscales * tntt.randn([8]*Nscales,[1]+[nrank]*(Nscales-1)+[1],var=var)
        vz = 0 #epsilon* (base_scaling/2)**Nscales * tntt.randn([8]*Nscales,[1]+[nrank]*(Nscales-1)+[1],var=var)
        for i in range(len(noisex)):
            scale = i + 2
            am = (base_scaling**scale) * epsilon 
            # Interpolate to full resolution using linear interpolation.
            if method =='linear':
                raise Exception('Not implemented')

            # Interpolate to full resolution using cuadratic interpolation.
            elif method == 'cuadratic':
                raise Exception('Not implemented')

            # Interpolate to full resolution using splines interpolation

            elif method == 'skc':
                pyAx =  qtt_skcubic3d_p(am *noisex[i],Nscales,  eps=eps, epsilon=epsilon, order=order, p_derivative=[0,1,0])
                pzAx =  qtt_skcubic3d_p(am *noisex[i],Nscales,  eps=eps, epsilon=epsilon, order=order,  p_derivative=[0,0,1])

                pxAy =  qtt_skcubic3d_p(am *noisey[i],Nscales,  eps=eps, epsilon=epsilon, order=order,  p_derivative=[1,0,0])
                pzAy =  qtt_skcubic3d_p(am *noisey[i],Nscales,  eps=eps, epsilon=epsilon, order=order,  p_derivative=[0,0,1])

                pxAz =  qtt_skcubic3d_p(am *noisez[i],Nscales,  eps=eps, epsilon=epsilon, order=order,  p_derivative=[1,0,0])
                pyAz =  qtt_skcubic3d_p(am *noisez[i],Nscales,  eps=eps, epsilon=epsilon, order=order,  p_derivative=[0,1,0])
            elif method == 'skq':
                pyAx =   qtt_skquad3d_p(am *noisex[i],Nscales,  eps=eps, epsilon=epsilon,  p_derivative=[0,1,0])
                pzAx =   qtt_skquad3d_p(am *noisex[i],Nscales,  eps=eps, epsilon=epsilon,  p_derivative=[0,0,1])

                pxAy =  qtt_skquad3d_p(am *noisey[i],Nscales,  eps=eps, epsilon=epsilon,  p_derivative=[1,0,0])
                pzAy =  qtt_skquad3d_p(am *noisey[i],Nscales,  eps=eps, epsilon=epsilon,  p_derivative=[0,0,1])

                pxAz =  qtt_skquad3d_p(am *noisez[i],Nscales,  eps=eps, epsilon=epsilon,  p_derivative=[1,0,0])
                pyAz =  qtt_skquad3d_p(am *noisez[i],Nscales,  eps=eps, epsilon=epsilon,  p_derivative=[0,1,0])


            vx += pyAz - pzAy
            vy += (-1)*pxAz + pzAx
            vz += pxAy - pyAx

        vy = vy.round(eps)
        vx = vx.round(eps)
        vz = vz.round(eps)

        return vx, vy, vz
    else:
        raise Exception('Field not implemented')