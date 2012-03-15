from PyTrilinos import Epetra, EpetraExt
from EpetraCG import cg
from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
import numpy as np
from scipy.sparse import csr_matrix, spdiags, rand

np.random.seed(12)
N=4
data=np.array([[-1],[2],[-1]])*np.ones((1,N))
diags = np.array([-1,0,1])
Z = rand(N,N, 0.2)
Z = Z + Z.T 
#A_csr=spdiags(data, diags, 4, 4).tocsr()
A_csr = Z.tocsr() 
mycomm = Epetra.PyComm()
A = scipy_csr_matrix2CrsMatrix(A_csr, mycomm)

#initial guess
x0=np.zeros((N),dtype='float')
X = Epetra.Vector(A.DomainMap())
for ii in range(X.MyLength()):
    i=  X.Map().GID(ii)
    X[ii] = x0[i] 
# rhs
F=np.ones((N),dtype='float')
Y = Epetra.Vector(A.RangeMap())
for ii in range(Y.MyLength()):
    i=  Y.Map().GID(ii)
    Y[ii] = F[i]

sol = cg(A, X, Y, 1e-9, 2)

