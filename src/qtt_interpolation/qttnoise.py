import numpy as np
import torch as tn
import torchtt as tntt
from .utils import reduceg, P_qtt, qtt_polynomial_cores

def qttperlin(n_size, f_scale, octaves=1, persistance = 1/2.0 , r_noise = 5,sorder=3,eps=1e-10):

    next = f_scale - n_size

    if sorder ==3:
        c0 =  [1,0,-3,2]
        c1 = [0,0,3,-2]
    elif sorder == 5:
        c1 = [0,0,0,10,-15,6]
        c0 = [1,0,0,-10,15,-6] 


    #create the random QTT vector
    grads = tntt.randn([2]*n_size, [1]+[r_noise]*(n_size-1)+[1]).round(eps)

    #shift operators
    S1 = P_qtt(2**n_size-1,n_size)

    def mx(coesf):
        return [0] + coesf 
    def mx1(coesf):
        mc = [-x for x in coesf] + [0]
        return np.array(mc) + np.array([0] + coesf )

    
    #fade functions + dot product
    psi1 = tntt.TT( qtt_polynomial_cores(mx1(c1),next) )
    psi0 = tntt.TT( qtt_polynomial_cores(mx(c0),next) )

    # interpolate

    perlinTT = tntt.kron(grads, psi0) + tntt.kron(S1@grads,psi1)
    perlinTT = perlinTT.round(eps)

    
    for i in range(1,octaves):
        if i == 1:
            per  = tntt.kron(tntt.ones([2]),reduceg(perlinTT,0,-1))
        else:
            per = reduceg(per,0,-1)
        
        perlinTT += persistance*per
        perlinTT = perlinTT.round(eps)
    
    return perlinTT

def qttcos(ncores,a=2*np.pi,b=0):

    #first core
    c1 = np.zeros((1,2,2))
    c1[:,0,:] = [1,0]
    c1[:,1,:] = [np.cos(a/2),np.sin(a/2)]
    cores = [c1]

    for i in range(2,ncores):

        c = np.zeros((2,2,2))
        c[:,0,:] = [[1,0],[0,1]]
        c[:,1,:] = [[np.cos(a/2**i),np.sin(a/2**i)],[-np.sin(a/2**i),np.cos(a/2**i)]]
        cores.append(c) 

    #last core
    cl = np.zeros((2,2,1))

    cl[:,0,:] = np.array([np.cos(b),np.sin(b)]).reshape(2,1)
    cl[:,1,:] = np.array([np.cos(a/2**ncores - b),-np.sin(a/2**ncores -b)]).reshape(2,1)
    cores.append(cl)

    return [tn.tensor(core, dtype=tn.float64) for core in cores]

def qttsin(ncores,a=2*np.pi,b=0):

    #first core
    c1 = np.zeros((1,2,2))
    c1[:,0,:] = [1,0]
    c1[:,1,:] = [np.cos(a/2),np.sin(a/2)]
    cores = [c1]

    for i in range(2,ncores):

        c = np.zeros((2,2,2))
        c[:,0,:] = [[1,0],[0,1]]
        c[:,1,:] = [[np.cos(a/2**i),np.sin(a/2**i)],[-np.sin(a/2**i),np.cos(a/2**i)]]
        cores.append(c) 

    #last core
    cl = np.zeros((2,2,1))

    cl[:,0,:] = np.array([np.sin(-b),np.cos(b)]).reshape(2,1)
    cl[:,1,:] = np.array([np.sin(a/2**ncores - b),np.cos(a/2**ncores -b)]).reshape(2,1)
    cores.append(cl)

    return [tn.tensor(core, dtype=tn.float64) for core in cores]



def mid_point_1d(ncores,octaves=None, nrank=10,var=1,eps=1e-10, H = 1):

    if octaves==None:
        octaves = ncores
    depth = ncores
    w = 2**(-H)
    r = nrank

    for i in range(1,octaves+1):
        phi1 = tntt.TT(qtt_polynomial_cores([1,-1], depth-i))
        phi2 = tntt.TT(qtt_polynomial_cores([0,1], depth-i))
        # define shift matrix
        sm = P_qtt( 2**i -1, i)
        if i == 1:
            noise = tntt.TT(tn.rand(2, dtype=tn.float64))
            M = tntt.TT(tn.tensor([[0,1],[1,0]],dtype=tn.float64),[(2,2)])
            mpsm = tntt.kron(noise,phi1) + tntt.kron( (M@noise).round(), phi2)
            #mpsm = mpsm.round(eps)
        
        elif i == ncores :
            delta = tntt.kron(tntt.ones([2]*(ncores-1)),tntt.TT(tn.tensor([0,1])) )
            noise = tntt.randn([2]*ncores,[1]+[r]*(ncores-1)+[1],var=var)
            noise = noise*delta
            mpsm += w**(ncores-1) * level
            #mpsm = (mpsm).round(eps)
        else:
            delta = tntt.kron(tntt.ones([2]*(i-1)),tntt.TT(tn.tensor([0,1])) )
            noise = tntt.randn([2]*i,[1]+[r]*(i-1)+[1],var=var)
            noise = noise*delta
            level = tntt.kron(noise,phi1) + tntt.kron( (sm@noise).round(), phi2)
            mpsm += w**(i-1) * level
    mpsm = (mpsm).round(eps)

    return mpsm