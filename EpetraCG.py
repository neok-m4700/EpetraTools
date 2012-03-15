def cg(A, x0, F, prec, maxit, show=True):
    """
    sol, res, k = cg(A, x0, F,  prec, maxit, show)
    Implements a naive conjugate gradient
    
    Input :
    -------
    
    A     : discrete laplacian matrix
    x0    : initial guesse 
    F     : source term (right hand side)
    prec  : relative precision: STOP when ||r|| < prec ||F|| 
    maxit :  max number of iterations
    
    Output :
    --------
    
    sol   : solution of the system
    res   : true residual
    it    : number of iterations 
    """ 
   
    from Epetra import Vector
    from numpy import sqrt
    
    x = Vector(x0)
    r = Vector(F)
    Ax = Vector(r)
    A.Multiply(False, x, Ax);
    r -= Ax
    p=Vector(r)
    rsold=r.Dot(r);
    Ap = Vector(Ax) 
    nF = F.Norm2()
    for i in range(maxit):
        A.Multiply(False, p , Ap)
        alpha=rsold/(p.Dot(Ap))
        x += alpha * p
        r -= alpha * Ap
        rsnew = r.Dot(r)
        n_r = sqrt(rsnew)
	if A.Comm().MyPID() == 0:
	   print '||r|| = %.3e' % n_r
	if n_r < (prec * nF):
              break
        p = r + rsnew/rsold * p
        print p
	rsold=Vector(rsnew)
           
    return x 

