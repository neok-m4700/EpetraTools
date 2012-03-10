from PyTrilinos import Epetra, EpetraExt
from EpetraMyTools import subVector 
import numpy as np

comm = Epetra.PyComm()
v = np.linspace(0,1.,10)

xMap=Epetra.Map(v.shape[0], 0 , comm)
X=Epetra.Vector(xMap)
for ii in range(X.MyLength()):
    i = xMap.GID(ii)
    X[ii] = v[i] 
print X
x3=subVector(X, range(6))
print x3 

