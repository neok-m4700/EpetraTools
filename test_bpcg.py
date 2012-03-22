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

Bs=load_mat("B.mat.npz")
B=scipy_csr_matrix2CrsMatrix(Bs, mycomm)
Hs=load_mat("H.mat.npz")
h=Hs.diagonal()
Qhs=spdiags(2./h, 0, Hs.shape[0], Hs.shape[1]).tocsr()
Qh=scipy_csr_matrix2CrsMatrix(Qhs, mycomm)
H=scipy_csr_matrix2CrsMatrix(Hs, mycomm)

f = load_vec("F.mat.npz")
mpc=load_vec("mpc.mat.npz")
Qss=spdiags(1./mpc[:,0],(0),mpc.shape[0],mpc.shape[0]).tocsr()
Qs=scipy_csr_matrix2CrsMatrix(Qss, mycomm)

vx=Epetra.Vector(H.DomainMap())
vy=Epetra.Vector(B.DomainMap())
Fx=Epetra.Vector(H.RangeMap())
Fy=Epetra.Vector(Qs.RangeMap())

Nh = H.NumGlobalCols() 
fxmap =  Fx.Map()
for ii in range(Fx.MyLength()):
    i = fxmap.GID(ii)
    Fx[ii] =f[0:Nh][i] 
fymap =  Fy.Map()
for ii in range(Fy.MyLength()):
    i = fymap.GID(ii)
    Fy[ii] =f[Nh:][i] 

# solving system
t1 = tps.WallTime()
x,y,res, it = bpcg(H, B, Fx, Fy , Qh, Qs, vx, vy , 1e-7, 100, True)
t2 = tps.WallTime()
if mycomm.MyPID() == 0:
   print 'tps bpcg = %.3es' %  (t2-t1)
