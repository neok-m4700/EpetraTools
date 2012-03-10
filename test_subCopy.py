from PyTrilinos import Epetra, EpetraExt
comm=Epetra.PyComm()
[ierr,A]=EpetraExt.MatlabFileToCrsMatrix("bidon.mat",comm)
comm =A.Comm()
aRMap = A.RangeMap()
#print "Map=", aRMap
aDMap = A.DomainMap()
row_elems=aRMap.MyGlobalElements()
col_elems=aDMap.MyGlobalElements()
indx=[0,8]
row_local_elems=[item for item in row_elems if item in indx]
new_row_Map = Epetra.Map(-1, row_local_elems , 0, comm)
#print new_row_Map
from EpetraExt import CrsMatrix_SubCopy
sousmat = CrsMatrix_SubCopy(new_row_Map)
s=sousmat(A)
print "A =" , A
print "s = ", s
del A
