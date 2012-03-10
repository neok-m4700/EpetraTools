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

def subCrsMatrix(A, indx, indy):
    #aliases
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

def useSubCopy(A, indx, indy):
    comm =A.Comm()
    aRMap = A.RowMap()
    aDMap = A.DomainMap()
    row_elems=aRMap.MyGlobalElements()
    col_elems=aDMap.MyGlobalElements()
    #find local_elems which are in indx
  
    row_local_elems=[item for item in row_elems if item in indx]
    col_local_elems=[item for item in col_elems if item in indy]
    
    # the local compopents of the Map of S
    row_Map = Epetra.Map(-1, row_local_elems, 0, comm)
    col_Map = Epetra.Map(-1, col_local_elems, 0, comm)
    
    from EpetraExt import CrsMatrix_SubCopy
    sousmat = CrsMatrix_SubCopy(aRMap,aDMap)
    S=sousmat(A)
    return S
