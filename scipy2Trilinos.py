from PyTrilinos import Epetra
def scipy_csr_matrix2CrsMatrix(sp, comm):
    Ap = sp.indptr
    Aj = sp.indices
    Ax = sp.data
    Ajmax = Aj.max()
    #print Ajmax
    m = Ap.shape[0]-1
    arMap = Epetra.Map(m, 0, comm)
    acMap = Epetra.Map(Ajmax, 0, comm)
    aGraph = Epetra.CrsGraph( Epetra.Copy, arMap, acMap, 0)
    for ii in range(arMap.NumGlobalElements()):
          i = arMap.GID(ii)
          indy = range(Ap[i],Ap[i+1])
          if (indy != []): 
              aGraph.InsertGlobalIndices(i, Aj[indy])
    aGraph.FillComplete()
    #print "Graph", aGraph.NumGlobalCols()
    A = Epetra.CrsMatrix(Epetra.Copy, aGraph)
    for ii in range(arMap.NumGlobalElements()):
          i = arMap.GID(ii)
          indy = range(Ap[i],Ap[i+1])
	  if (indy != []): 
	      A.SumIntoGlobalValues(i, Ax[indy], Aj[indy])
    A.FillComplete()
    return A	      
