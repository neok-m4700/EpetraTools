from PyTrilinos import Epetra, EpetraExt
import numpy as np

def sub_vec(v, ind)
     myelems=vMap.MyGlobalElements()
     #find local_elems which are in indx
     local_elems=[item for item in myelems if item in ind]
     
     xMap = Epetra.Map(-1, local_elems, 0, comm)
     for ii in range(xMap.MyLength()):
           x[ii] = 
 
comm = Epetra.PyComm()
v = np.linspace(0,1.,10)

xMap=Epetra.Map(v.shape[0], 0 , comm)
X=Epetra.Vector(xMap)
for ii in range(X.MyLength()):
    i = xMap.GID(ii)
    X[ii] = v[i] 
recup = X.ExtractCopy()
x3=Epetra.Vector(recup[0:6])
print x3 

