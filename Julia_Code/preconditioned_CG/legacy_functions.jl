function MGCG!(A,b,x,H1,exact, my_solver;smooth_steps = 4,maxiter=length(b)^2,abstol=sqrt(eps(real(eltype(b)))))
    r = b - A * x;
    z = zeros(length(b))
    (M, R, H, I_p, A_2h, I_r, IN) = precond_matrix(A,b;m=smooth_steps,solver="jacobi")
    # Two_level_multigrid!(A,r,z;nu=smooth_steps, solver = my_solver)
    z = M*r #test 
    p = z;
    rzold = r'*z
    
    num_iter_steps = 0
    norms = [sqrt(r'*H1*r)]
 
    diff = x-exact
    err = sqrt(diff'*A*diff)
    
    E = [err]
    for step = 1:maxiter
    # for step = 1:5
        Ap = A*p;
        @show norm(A*p)
        num_iter_steps += 1
        alpha = rzold/(p'*Ap)
        @show alpha
        x .= x .+ alpha * p;

        diff = x-exact
        err = sqrt(diff'*A*diff)
        @show err
        append!(E, err)
        
        r .= r .- alpha * Ap;
      
        norm_r = sqrt(r' * r)
        append!(norms,norm_r)

        if norm_r < abstol
            break
        end

    
        z .= 0
        # Two_level_multigrid!(A,r,z;nu=smooth_steps, solver = my_solver)
        z = M*r

        rznew = r' * z
        beta = rznew/rzold;
        @show beta
        p = z + beta * p;
        rzold = rznew
    end

    return E, num_iter_steps, norms
end