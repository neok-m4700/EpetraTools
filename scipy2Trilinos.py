from PyTrilinos import Epetra
def scipy_csr_matrix2CrsMatrix(sp, comm):
    Ap = sp.indptr
    Aj = sp.indices
    Ax = sp.data
    m = Ap.shape[0]-1
    aMap = Epetra.Map(m, 0, comm)
    aGraph = Epetra.CrsGraph( Epetra.Copy, aMap, 0)
    for ii in range(aMap.NumGlobalElements()):
          i = aMap.GID(ii)
          indy = range(Ap[i],Ap[i+1])
          if (indy != []): 
              aGraph.InsertGlobalIndices(i, Aj[indy])
    aGraph.FillComplete() 
    A = Epetra.CrsMatrix(Epetra.Copy, aGraph)
    for ii in range(aMap.NumGlobalElements()):
          i = aMap.GID(ii)
          indy = range(Ap[i],Ap[i+1])
	  if (indy != []): 
	      A.SumIntoGlobalValues(i, Ax[indy], Aj[indy])
    A.FillComplete()
    return A	      
