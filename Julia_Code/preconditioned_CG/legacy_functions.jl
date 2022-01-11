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

using Random
Random.seed!(777)
using LinearAlgebra
using SparseArrays
using Plots

function e(i,n)
    out = spzeros(n)
    out[i] = 1.0
    return out 
end

function eyes(n)
    out = spzeros(n,n)
    for i in 1:n
        out[i,i] = 1.0
    end
    return out
end

function u(x, y)
    return sin.(π*x) * (cosh.(π*y))'
end

function u_x(x, y)
    return π * cos.(π*x) * (cosh.(π*y))'
end

function u_y(x, y)
    return π * sin.(π*x) * (sinh.(π*y))'
end

function u_xx(x, y)
    return -π^2 * sin.(π*x) * (cosh.(π*y))'
end

function u_yy(x, y)
    return π^2 * sin.(π*x) * (cosh.(π*y))'
end
function Diag(A)
    # Self defined function that is similar to Matlab Diag
    return Diagonal(A[:])
end


function get_operators(p, h)
    N = Integer(1/h)
    Identity = eyes(N+1)
   
    # Create some random positive definite coefficients
    


    x = range(0,step=h,1);
    y = range(0,step=h,1);
    B = ones(N+1)

    (D1, HI, H1, r1) = diagonal_sbp_D1(p,N,xc=(0,1));
    (D2, BS, HI2, H2, r2) = variable_diagonal_sbp_D2(p,N,B;xc=(0,1));
 

    # Forming 2d Operators
    Hx = kron(Identity, H1)
    HxI = kron(Identity, HI)
    BSx = kron(Identity, BS)

    Hy = kron(H1, Identity)
    HyI = kron(HI, Identity)
    BSy = kron(BS, Identity)

    HH = kron(H1, H1)

    D2xplusD2y = kron(Identity, D2) + kron(D2, Identity)
    e1 = e(1,N+1);
    eN = e(N+1,N+1);
    
    E_N = sparse(kron(Diag(eN), Identity));   
    E_S = sparse(kron(Diag(e1), Identity));
    E_E = sparse(kron(Identity, Diag(eN)));   
    E_W = sparse(kron(Identity, Diag(e1)));


    # Penalty Parameters
    tau_E = -13/h;
    tau_W = -13/h;    
    tau_N = -1;
    tau_S = -1; 
    beta = 1;
    SAT_W = tau_W*HxI*E_W + beta*HxI*BSx'*E_W;
    SAT_E = tau_E*HxI*E_E + beta*HxI*BSx'*E_E;
    #SAT_W_r = tau_W*crr[1]*HyI*E_W*e1 + beta*HyI*BSy'*E_W*e1;
    #SAT_E_r = tau_E*crr[end]*HyI*E_E*eN + beta*HyI*BSy'*E_E*eN;
    
    g_W = 0;
    g_E = 0;
    g_N = u_y(x, 1);
    g_S = u_y(x, 0);



    SAT_N = tau_N*HyI*E_N*BSy;
    SAT_S = tau_S*HyI*E_S*BSy;
    SAT_N_r = tau_N*HyI*kron(eN, Identity);
    SAT_S_r = tau_S*HyI*kron(e1,Identity);
    


    A = D2xplusD2y + SAT_W + SAT_E + SAT_N + SAT_S;
  
    
   
    b =  0 .* u(x, y')[:] + SAT_N_r*g_N  + SAT_S_r*g_S;


    A = -HH*A;
    b = Vector(-HH*b);


    return (A, b, x, y, HH)
end

function prolongation_matrix(N)
    # SBP preserving
    odata = spzeros(2*N-1,N)
    for i in 1:2*N-1
        if i % 2 == 1
            odata[i,div(i+1,2)] = 1
        else
            odata[i,div(i,2)] = 1/2
            odata[i,div(i,2)+1] = 1/2
        end
    end
    return odata
end

function restriction_matrix(N)
    # SBP preserving
    odata = spzeros(div(N+1,2),N)
    odata[1,1] = 1/2
    odata[1,2] = 1/2
    odata[end,end-1] = 1/2
    odata[end,end] = 1/2
    for i in 2:div(N+1,2)-1
        odata[i,2*i-2] = 1/4
        odata[i,2*i-1] = 1/2
        odata[i,2*i] = 1/4
    end
    return odata
end

function standard_prolongation_matrix(N)
    # SBP preserving
    m = (N-1)*2 + 1
    I_r = standard_restriction_matrix(m)
    odata = 2 .* I_r'
    return odata
end

function standard_restriction_matrix(N)
    # SBP preserving
    odata = spzeros(div(N+1,2),N)
    odata[1,1] = 1/2
    odata[1,2] = 1/4
    odata[end,end-1] = 1/4
    odata[end,end] = 1/2
    for i in 2:div(N+1,2)-1
        odata[i,2*i-2] = 1/4
        odata[i,2*i-1] = 1/2
        odata[i,2*i] = 1/4
    end
    return odata
end

function standard_prolongation_matrix_2D(N)
    
    odata = standard_prolongation_matrix(Integer(sqrt(N)))
    return kron(odata, odata)
end

function standard_restriction_matrix_2D(N)
    odata = standard_restriction_matrix(Integer(sqrt(N)))
    return kron(odata, odata)
end


function ssor!(x,A,b;no_iter=4, ω = 1.4)
  
    N = length(b)
    IN = sparse(Matrix(I, N, N))
    P = Diagonal(diag(A))
    L = A - triu(A)
    U = A - tril(A)

    B1 = (P + ω*U)\Matrix(-ω*L + (1-ω)*P)
    B2 = (P + ω*L)\Matrix(-ω*U + (1-ω)*P) 
    X = (P+ω*L)\Matrix(IN)
    C = ω*(2-ω)*(P+ω*U)\Matrix(P*X)

    for j in 1:no_iter 
        x[:] = B1*B2*x[:] + C*b
    end
    
end



function richardson!(x,A,b;no_iter=4, ω =ω_richardson)


    for j in 1:no_iter 
        x[:] = x[:] + ω * (b - A*x[:]) # standard richardson
    end

  
end

function richardson_chebyshev!(x,A,b;no_iter=4, ω =ω_richardson)

 
    M = I - ω*A
    evals = eigen(Matrix(M))
    es1 =  extrema(evals.values)
    aa = es1[1]
    bb = es1[2]
    γ = 1-bb
    Γ = 1-aa 
    κ = Γ/γ
    c = (sqrt(κ)-1)/(sqrt(κ)+1)
    θ = 4*c/(bb-aa)
    ν = -2*c*(aa+bb)/(bb-aa)

    y0 = copy(x)
    y1 = y0 + ω * (b - A*y0)

    x[:] = y1[:]

   for j in 2:no_iter
        x[:] = θ * (y1[:] + ω * (b - A * y1[:]) - y0) + ν * (y1 - y0) + y0
        y0[:] = y1[:]
        y1[:] = x[:]
    end

end

function chebyshev!(x,A,b;no_iter=4)

  
    p = zeros(size(b))
    α = 0*rand(1)

    for j in 1:no_iter 
        r = b - A*x
        if j == 0
            α[1] = 1/d 
        elseif j == 1
            α[1] = 2*d/(2*d^2 - c^2)
        else
            α[1] = 1/(d - α[1]*c^2/4)
        end
        β = α[1]*d - 1
        p = α[1]*r + β*p
        x[:] = x[:] + p[:]


    end
  
end

function jacobi!(x,A,b;no_iter=4, ω = 2/3)

    Pinv = Diagonal(1 ./ diag(A))
    P = Diagonal(diag(A))
    Q = A-P

    for j in 1:no_iter
        x[:] = ω * Pinv*(b .- Q*x[:]) + (1 - ω)*x[:]
    end
    
end





function Two_level_multigrid!(A,b,x0;nu=3, solver = "jacobi")
    N = length(b)
  
    #pre-smoothing with initial guess x0
    if solver == "jacobi"
        jacobi!(x0,A,b;no_iter=nu)
    elseif solver == "ssor"
        ssor!(x0,A,b;no_iter=nu)
    elseif solver == "richardson"
        richardson!(x0,A,b;no_iter=nu)
    elseif solver == "richardson_chebyshev"
        richardson_chebyshev!(x0,A,b;no_iter=nu)
    elseif solver == "chebyshev"
        chebyshev!(x0,A,b;no_iter=nu)
    else

        #add other solvers
    end

   
    #compute residual on fine grid and restrict
    rh = b - A*x0
    r2h = standard_restriction_matrix(N)*rh
   
    
    #Get operators on coarse grid
    (A_2h, b_2h, x_2h, H1_2h) = get_operators(p, 2*h);
   
    #Direct solve on coarse grid and prolong error
    e_2h = A_2h\r2h
    e_h = standard_prolongation_matrix(length(b_2h))*e_2h
  

    #Fine grid correction
    x0[:] .+= e_h

    #post-smoothing
    if solver == "jacobi"
        jacobi!(x0,A,b;no_iter=nu)
    elseif solver == "ssor"
        ssor!(x0,A,b;no_iter=nu)
    elseif solver == "richardson"
        richardson!(x0,A,b;no_iter=nu)
    elseif solver == "richardson_chebyshev"
        richardson_chebyshev!(x0,A,b;no_iter=nu)
    elseif solver == "chebyshev"
        chebyshev!(x0,A,b;no_iter=nu)
            
    else
        #add other solvers
    end
 
    
end