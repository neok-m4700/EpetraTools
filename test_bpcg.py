from PyTrilinos import Epetra, EpetraExt
from EpetraMyTools import subVector
from EpetraBpcg import bpcg
from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
from numpy import load, nonzero
import scipy
import scipy.io
from scipy.sparse import csr_matrix, csc_matrix
import numpy as np

def load_mat(filename):
    dic=np.load(filename)
    indp=dic['arr_0']
    indi=dic['arr_1']
    data=dic['arr_2']
    m=dic['arr_3']
    n=dic['arr_4']
    return csr_matrix((data,indi,indp),shape=(m,n))

def load_mat_matlab(filename):
    from scipy.io import loadmat
    dic=loadmat(filename)
    m = dic['m']
    n = dic['n']
    indp=dic['Ap'][:,0]
    indi=dic['Ai'][:,0]
    data=dic['Ax'][:,0]
    return csc_matrix((data,indi,indp),shape=(m,n)).tocsr()

def load_vec(filename):
    dic=load(filename)
    return dic['arr_0']

def copy_vec(Vp, Vn):
    vmap =  Vp.Map();
    for ii in range(Vp.MyLength()):
        i = vmap.GID(ii)
	Vp[ii] =Vn[i] 

mycomm = Epetra.PyComm()
verbose = (mycomm.MyPID() == 0)
tps = Epetra.Time(mycomm)
tps.ResetStartTime()

# load H and convert it
#Hs=load_mat("H.mat.npz")
Hs=load_mat_matlab("H.mat")
H=scipy_csr_matrix2CrsMatrix(Hs, mycomm)

#load B and convert it
#Bs=load_mat("B.mat.npz")
Bs=load_mat_matlab("B.mat")
B=scipy_csr_matrix2CrsMatrix(Bs, mycomm)

# build Qh diagonal precond
Qh=Epetra.Vector(H.DomainMap())
copy_vec(Qh, 2./(Hs.diagonal()))

# buld Qs diagonal precond
#mpc=load_vec("mpc.mat.npz")
m_dic_mat=scipy.io.loadmat('mpc.mat')
mpc=m_dic_mat['mpc'][:,0]
Qs=Epetra.Vector(B.DomainMap())
copy_vec(Qs, 1./mpc[:,0].todense())

vx=Epetra.Vector(H.DomainMap())
vy=Epetra.Vector(B.DomainMap())

# build RHS
#f = load_vec("F.mat.npz")
F_mat=scipy.io.loadmat('F.mat')
f=F_mat['F']
Fx=Epetra.Vector(H.RangeMap())
# Fy belongs to RangeMap(B.transpose()) = B.DomainMap()
Fy=Epetra.Vector(B.DomainMap())
Nh = H.NumGlobalCols() 
copy_vec(Fx, f[0:Nh])
copy_vec(Fy, f[Nh:])

# solving system
t1 = tps.WallTime()
res, it = bpcg(H, B, Fx, Fy , Qh, Qs, vx, vy , 1e-7, 3000, True)
t2 = tps.WallTime()
if verbose:
   print "res =  %.3e , it = %d" % (res,it) 
   print 'tps bpcg = %.3es' %  (t2-t1)
