import numpy as sp
import scipy as sp
import scipy.sparse
import numpy as np
from PyTrilinos import Epetra

np.set_printoptions(linewidth=np.nan)


def crs2csr(A, **kwargs):
    if A.NumGlobalNonzeros():
        indptr = [0]
        indices, data = [], []
        # column indices for row i are stored in indices[indptr[i]:indptr[i+1]]
        # values are stored in data[indptr[i]:indptr[i+1]]
        for i in A.RowMap().MyGlobalElements():
            val, ind = A.ExtractGlobalRowCopy(i)
            data.extend(val)
            indices.extend(ind)
            indptr.append(len(indices))
            # print(f'row {i} values={val} indices={ind} sz={ind.size}')
        return sp.sparse.csr_matrix((
            np.array(data),
            np.array(indices, dtype='i4'),
            np.array(indptr, dtype='i4')
        ), **kwargs)
    else:
        return None  # return a None if empty matrix on other procs


def locVect2glob(Vloc, vec_map, n_vecs=1, mode=Epetra.Insert):
    rootMap = Epetra.Util_Create_Root_Map(vec_map)
    V = Epetra.Vector(rootMap) if n_vecs == 1 else Epetra.MultiVector(rootMap, n_vecs)
    V.Import(Vloc, Epetra.Import(rootMap, vec_map), mode)
    return V


def locMat2glob(Aloc, row_map, mode=Epetra.Insert):
    rootMap = Epetra.Util_Create_Root_Map(row_map)
    A = Epetra.CrsMatrix(Epetra.Copy, rootMap, 0, False)
    A.Import(Aloc, Epetra.Import(rootMap, row_map), mode)
    return A
