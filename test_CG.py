from PyTrilinos import Epetra, EpetraExt, TriUtils
from EpetraCG import cg
import numpy as np

N = 10
mycomm = Epetra.PyComm()
gal = TriUtils.CrsMatrixGallery('laplace_3d', mycomm)
gal.Set('nx', N)
gal.Set('ny', N)
gal.Set('nz', N)

A = gal.GetMatrix()
# RHS
# rhs
F = np.ones(A.NumGlobalCols(), dtype='float')
Y = Epetra.Vector(A.RangeMap())
for ii in range(Y.MyLength()):
    i = Y.Map().GID(ii)
    Y[ii] = F[i]

# initial guess
x0 = np.ones(A.NumGlobalRows(), dtype='float')
X = Epetra.Vector(A.DomainMap())
for ii in range(X.MyLength()):
    i = X.Map().GID(ii)
    X[ii] = x0[i]

sol, res, k = cg(A, X, Y, 1e-6, 2000)

if mycomm.MyPID() == 0:
    print('nb_it = ', k)
    print('n_r = %.3e ' % res)
    # print A.DomainMap()
