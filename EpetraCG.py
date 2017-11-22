def cg(A, x0, F, prec, maxit, show=False):
    """
    sol, k = cg(A, x0, F,  prec, maxit, show)
    Implements a naive conjugate gradient for solving Ax = F

    Input :
    -------

    A     : operator
    x0    : initial guess
    F     : right hand side
    prec  : relative precision: STOP when ||r|| < prec ||F||
    maxit :  max number of iterations

    Output :
    --------

    sol   : solution of the system
    res   : true residual
    it    : number of iterations
    """

    from PyTrilinos.Epetra import Vector
    from numpy import sqrt

    x = Vector(x0)
    r = Vector(F)
    Ap = Vector(F)
    A.Apply(x, Ap)
    r.Update(-1., Ap, 1.)
    p = Vector(r)
    rsold = r.Dot(r)[0]
    nF = F.Norm2()
    for i in range(maxit):
        A.Apply(p, Ap)
        alpha = rsold / (p.Dot(Ap)[0])
        x.Update(alpha, p, 1.)
        r.Update(-alpha, Ap, 1.)
        rsnew = r.Dot(r)[0]
        n_r = sqrt(rsnew)
        if A.Comm().MyPID() == 0 and show and (i % 1 == 0):
            print('it %d = %.3e' % (i, n_r))
        if n_r < (prec * nF):
            break
        p.Update(1., r, rsnew / rsold, p, 0.)
        rsold = rsnew
    return x, n_r, i
