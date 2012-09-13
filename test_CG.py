from PyTrilinos import Epetra, EpetraExt, TriUtils
from EpetraCG import cg
import numpy as np

N=100
mycomm = Epetra.PyComm()
gal = TriUtils.CrsMatrixGallery('laplace_3d', mycomm)
gal.Set('nx', N)
gal.Set('ny', N)
gal.Set('nz', N)

A = gal.GetMatrix()
Y = gal.GetStartingSolution()
X = gal.GetRHS()


sol = cg(A, X, Y, 1e-6, 2000)
