import numpy as np
from PyTrilinos import Epetra


def AddToProfile(G, indx, indy):
 iy=indy.tolist()
 for ix in indx: 
    G.InsertGlobalIndices(ix, iy)  

def AddMatElem(A, indx, indy, s):
  m = indx.__len__()
  n = indy.__len__()
  ix = [i for i in indx for j in range(n)]
  iy = indy.tolist()*m
  A.SumIntoGlobalValues(ix, iy, s)  

def subVector(V, ind):
    """ construct a Vector extracted from a Vector
    s = subVector(V, ind) with  s[k] = V[ind[k]]
    """
    comm =V.Comm()
    vMap = V.Map()
    myelems=vMap.MyGlobalElements()
    
    #find local_elems which are in indx
    local_elems=[item for item in myelems if item in ind]
    l = comm.GatherAll(local_elems.__len__())
    l2=[0]+l.cumsum()[:-1].tolist()
    myid=comm.MyPID()
    
    # the local compopents of the Map of S
    sMyElems=range(l2[myid],l2[myid]+l[myid])
    sMap = Epetra.Map(-1, sMyElems, 0, comm)
    s=Epetra.Vector(sMap)
    for i,el in enumerate(local_elems):
         s[sMap.LID(i)] = V[vMap.LID(i)]
    return s 

def subCrsMatrix(A, indx, indy):
    """ construct a CrsMatrix extracted from a CrsMatrix
    S = subCrsMatrix(A, indx, indy)
    S is such that S(k,l) = A(indx(k), indy(l))
    """
    comm =A.Comm()
    aMap = A.RowMap()
    myelems=aMap.MyGlobalElements()
    
    #find local_elems which are in indx
    local_elems=[item for item in myelems if item in indx]
    l = comm.GatherAll(local_elems.__len__())
    l2=[0]+l.cumsum()[:-1].tolist()
    myid=comm.MyPID()
    
    # the local compopents of the Map of S
    sMyElems=range(l2[myid],l2[myid]+l[myid])
    sMap = Epetra.Map(-1, sMyElems, 0, comm)
    S=Epetra.CrsMatrix(Epetra.Copy, sMap, A.MaxNumEntries())
    
    for it,elem  in enumerate(local_elems): 
	   [val,ind] = A.ExtractGlobalRowCopy(elem)
	   # the row in A is not empty
	   if ind.__len__() > 0:
	       coup=zip(*[(val[i],item) for i,item in enumerate(ind) if item in indy])
	       # the row must contains indices from indy
	       if coup.__len__()>0:
	          S.InsertGlobalValues(sMyElems[it], coup[0], coup[1])
    S.FillComplete()
    return S
