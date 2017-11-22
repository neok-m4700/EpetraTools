def bpcg(H, B, Fx, Fy, Qh, Qs, x, y, prec, maxit, show):
    """
    sol, res, k = bpcg(H, B, F, Qh, Qs, v0, prec, maxit, show)
    Implements the Bramble-Pasciak Block preconditionned conjugate gradient
    algorithm taken from the article
    to solve 

    [ H B ] . [ x ] = [ Fx ]
    [ B' 0]   [ y ]   [ Fy ]

    Input :
    -------

    H     : discrete laplacian matrix
    B     : divergence part
    Fx    : laplacian right hand side
    Fy    : schur complement right hand side
    Qh    : vector of diagonal precond for matrix H (H-Qh) must be positive
    Qs    : vector of diagonal schur complement 
    x     : laplacian initial guess 
    y     : schur complement initial guess 
    prec  : relative precision: STOP when ||r|| < prec ||F|| 
    maxit : max number of iterations
    show  : boolean flag for information

    Output :
    --------

    x   : laplacian part of the solution
    y   : schur complement par of the solution
    res : residual (||K*x-F||/||F||)
    it  : number of iterations 
    """

    from Epetra import Vector
    from numpy import sqrt
    verbose = (H.Comm().MyPID() == 0)

    r1 = Vector(x)
    r2 = Vector(y)

    # r1_0 = Fx - H * x - B * y
    tr1 = Vector(Fx)
    H.Multiply(False, x, r1)
    tr1.Update(-1., r1, 1.)
    B.Multiply(False, y, r1)
    tr1.Update(-1., r1, 1.)

    # r2_0 = F_y - B' * x
    tr2 = Vector(Fy)
    B.Multiply(True, x, r2)
    tr2.Update(-1., r2, 1.)

    # r0 = G r_check_0
    # with G = [inv(Qh)     0
    #           B*inv(Qh) - I]
    r1.Multiply(1., Qh, tr1, 0.)

    B.Multiply(True, r1, r2)
    r2.Update(-1., tr2, 1.)

    # norm evaluation
    res = sqrt(r1.Norm2()**2 + r2.Norm2()**2)
    nF = sqrt(Fx.Norm2()**2 + Fy.Norm2()**2)

    # pre-alloc
    z1 = Vector(x)
    z2 = Vector(y)

    w1 = Vector(x)
    w2 = Vector(y)

    q1 = Vector(x)
    q2 = Vector(y)

    d = Vector(x)
    ###########################################
    # MAIN LOOP
    ###########################################
    k = 0
    while ((res > prec * nF) and (k <= maxit)):

        # solve the \tilde{K} z^k = r^k
        z2.Multiply(1., Qs, r2, 0.)
        z1.Update(1., r1, 0.)

        # d = H * r_1^k
        H.Multiply(False, r1, d)

        # beta^n_k = <d,r_1^k> -<r_check_1^k,r_1^k> +<z_2^k,r_2^k>
        bet_n = d.Dot(r1) - tr1.Dot(r1) + z2.Dot(r2)

        if k == 0:
            bet = 0.
            p1 = Vector(z1)
            p2 = Vector(z2)
            s = Vector(d)
        else:
            # beta_k = beta^n_k /beta^n_{k-1}
            bet = bet_n / bet_n1

            # p^k = z^k + beta_k* p^{k-1}
            p1.Update(1., z1, bet)
            p2.Update(1., z2, bet)

            # s^k = d + beta_k* s^{k-1}
            s.Update(1., d, bet)

        # q = [s;0] + [B' p2^k ; B * p1^k]
        B.Multiply(False, p2, q1)
        q1.Update(1., s, 1.)
        B.Multiply(True, p1, q2)

        # w = [Qh^{-1}q1  ; B'Qh^{-1}q1 -q2 ]
        w1.Multiply(1., Qh, q1, 0.)
        #w2 = B.T*w1-q2
        B.Multiply(True, w1, w2)
        w2.Update(-1., q2, 1.)

        # alpha_k^d = <w_1,s^k>-<q_1,p_1^k> + <w_2,p_2^k>
        alp_d = w1.Dot(s) - q1.Dot(p1) + w2.Dot(p2)

        # alpha_k = beta^n_k / alpha_k^d
        alp = bet_n / alp_d

        # v^{k+1} = v^k + alpha_k p^k
        x.Update(alp, p1, 1.)
        y.Update(alp, p2, 1.)

        # r^{k+1} = r^k - alpha_k w
        r1.Update(-alp, w1, 1.)
        r2.Update(-alp, w2, 1.)

        # r_check_1^{k+1} = r_check_1^k - alpha_k q_1
        tr1.Update(-alp, q1, 1.)

        # update
        bet_n1 = bet_n
        k += 1

        res = sqrt(r1.Norm2()**2 + r2.Norm2()**2)
        if show and (k % 10 == 0) and verbose:
            print('%d  %.3e ' % (k, res / nF))

    H.Multiply(False, x, tr1)
    B.Multiply(False, y, r1)
    tr1.Update(1., r1, -1., Fx, 1.)

    B.Multiply(True, x, tr2)
    tr2.Update(-1., Fy, 1.)
    residu = sqrt(tr1.Norm2()**2 + tr2.Norm2()**2)
    return residu / nF, k
