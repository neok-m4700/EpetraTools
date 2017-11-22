from PyTrilinos import Epetra, EpetraExt
from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
from scipy.sparse import csr_matrix, rand
import numpy as np
comm = Epetra.PyComm()
dense = np.array([[0, 0, 1, 4], [1, 0, 0, 0], [0, 8, 9, 7], [0, 1, 0, 0]], dtype='float')
sp_csr = csr_matrix(dense)
# sp_coo=rand(1000,1000,0.1)
#sp_csr = csr_matrix(sp_coo)
A = scipy_csr_matrix2CrsMatrix(sp_csr, comm)
print(A)
