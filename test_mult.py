from PyTrilinos import Epetra, EpetraExt
import numpy as np
comm=Epetra.PyComm()
[ierr,A]=EpetraExt.MatlabFileToCrsMatrix("bidon.mat",comm)
comm =A.Comm()
x0=np.ones(A.NumGlobalCols(),dtype='float')
X = Epetra.Vector(A.DomainMap())
for ii in range(X.MyLength()):
    i=  X.Map().GID(ii)
    X[ii] = x0[i] 
# rhs
F=np.ones((A.NumGlobalRows()),dtype='float')
Y = Epetra.Vector(A.RangeMap())
for ii in range(Y.MyLength()):
    i=  Y.Map().GID(ii)
    Y[ii] = F[i]
A.Multiply(False, X, Y)
print Y
del A
