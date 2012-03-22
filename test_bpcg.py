from PyTrilinos import Epetra, EpetraExt
from EpetraMyTools import subVector
from EpetraBpcg import bpcg
from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
from numpy import load, nonzero
from scipy.sparse import csr_matrix
import numpy as np

def load_mat(filename):
    dic=load(filename)
    indp=dic['arr_0']
    indi=dic['arr_1']
    data=dic['arr_2']
    return csr_matrix((data,indi,indp))

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
#chargement fichier 
t1 = tps.WallTime()
As=load_mat("A.mat.npz")
t2 = tps.WallTime()
if (mycomm.MyPID() == 0):
   print 'tps load A = %.3es' %  (t2-t1)
t1 = tps.WallTime()
A=scipy_csr_matrix2CrsMatrix(As, mycomm)
t2 = tps.WallTime()
if verbose: 
   print 'size(A) = (%d,%d) nnz(A) = %d '  %(A.NumGlobalRows(), A.NumGlobalCols(), A.NumGlobalNonzeros())
   print 'tps convert A = %.3es' %  (t2-t1)

from scipy.sparse import spdiags

# load H and convert it
Hs=load_mat("H.mat.npz")
H=scipy_csr_matrix2CrsMatrix(Hs, mycomm)

#load B and convert it
Bs=load_mat("B.mat.npz")
B=scipy_csr_matrix2CrsMatrix(Bs, mycomm)

# build Qh diagonal precond
Qh=Epetra.Vector(H.DomainMap())
copy_vec(Qh, 2./(Hs.diagonal()))

# buld Qs diagonal precond
mpc=load_vec("mpc.mat.npz")
Qs=Epetra.Vector(B.DomainMap())
copy_vec(Qs, 1./mpc[:,0])

vx=Epetra.Vector(H.DomainMap())
vy=Epetra.Vector(B.DomainMap())

# build RHS
f = load_vec("F.mat.npz")
Fx=Epetra.Vector(H.RangeMap())
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
