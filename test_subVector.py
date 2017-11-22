from PyTrilinos import Epetra, EpetraExt
from EpetraMyTools import subVector
mycomm = Epetra.PyComm()
rank = mycomm.MyPID()
N = 20
f = list(range(N))
fMap = Epetra.Map(N, 0, mycomm)
F = Epetra.Vector(fMap)
for ii in range(F.MyLength()):
    i = fMap.GID(ii)
    F[ii] = f[i]

# print "rank = ", rank, 'F =', F

V = subVector(F, list(range(0, N / 2)))
print("rank = ", rank, 'V =', V)
W = subVector(F, list(range(N / 2, N)))
print("rank = ", rank, 'W =', W)
T = subVector(F, [8, 12, 2, 17])
print("rank = ", rank, 'T =', T)
