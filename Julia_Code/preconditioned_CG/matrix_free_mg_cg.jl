using CUDA
using Random

include("level_2_multigrid_new.jl")
include("../split_matrix_free.jl")

Random.seed!(123)

function matrix_free_prolongation_2d(idata,odata)
    size_idata = size(idata)
    odata_tmp = zeros(size_idata .* 2)
    for i in 1:size_idata[1]-1
        for j in 1:size_idata[2]-1
            odata[2*i-1,2*j-1] = idata[i,j]
            odata[2*i-1,2*j] = (idata[i,j] + idata[i,j+1]) / 2
            odata[2*i,2*j-1] = (idata[i,j] + idata[i+1,j]) / 2
            odata[2*i,2*j] = (idata[i,j] + idata[i+1,j] + idata[i,j+1] + idata[i+1,j+1]) / 4
        end
    end
    for j in 1:size_idata[2]-1
        odata[end,2*j-1] = idata[end,j]
        odata[end,2*j] = (idata[end,j] + idata[end,j+1]) / 2 
    end
    for i in 1:size_idata[1]-1
        odata[2*i-1,end] = idata[i,end]
        odata[2*i,end] = (idata[i,end] + idata[i+1,end]) / 2
    end
    odata[end,end] = idata[end,end]
    return nothing
end

function matrix_free_prolongation_2d_GPU(idata,odata)
    (Nx,Ny) = size(idata)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx+TILE_DIM_1-1,TILE_DIM_1), div(Ny+TILE_DIM_2-1,TILE_DIM_2))
	blockdim = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim blocks=griddim prolongation_2D_kernel(idata,odata,Nx,Ny,Val(TILE_DIM_1),Val(TILE_DIM_2))
    nothing
end

function prolongation_2D_kernel(idata,odata,Nx,Ny,::Val{TILE_DIM_1},::Val{TILE_DIM_2}) where {TILE_DIM_1,TILE_DIM_2}
    tidx = threadIdx().x
    tidy = threadIdx().y
    i = (blockIdx().x - 1) * TILE_DIM_1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM_2 + tidy


    if 1 <= i <= Nx-1 && 1 <= j <= Ny-1
        odata[2*i-1,2*j-1] = idata[i,j]
        odata[2*i-1,2*j] = (idata[i,j] + idata[i,j+1]) / 2
        odata[2*i,2*j-1] = (idata[i,j] + idata[i+1,j]) / 2
        odata[2*i,2*j] = (idata[i,j] + idata[i+1,j] + idata[i,j+1] + idata[i+1,j+1]) / 4
    end 

    if 1 <= j <= Ny-1
        odata[end,2*j-1] = idata[end,j]
        odata[end,2*j] = (idata[end,j] + idata[end,j+1]) / 2 
    end

    if 1 <= i <= Nx-1
        odata[2*i-1,end] = idata[i,end]
        odata[2*i,end] = (idata[i,end] + idata[i+1,end]) / 2
    end

    odata[end,end] = idata[end,end]
    return nothing
end

function matrix_free_restriction_2d(idata,odata)
    size_idata = size(idata)
    size_odata = div.(size_idata .+ 1,2)
    idata_tmp = zeros(size_idata .+ 2)
    idata_tmp[2:end-1,2:end-1] .= idata

    for i in 1:size_odata[1]
        for j in 1:size_odata[2]
            odata[i,j] = (4*idata_tmp[2*i,2*j] + 
            2 * (idata_tmp[2*i,2*j-1] + idata_tmp[2*i,2*j+1] + idata_tmp[2*i-1,2*j] + idata_tmp[2*i+1,2*j]) +
             (idata_tmp[2*i-1,2*j-1] + idata_tmp[2*i-1,2*j+1] + idata_tmp[2*i+1,2*j-1]) + idata_tmp[2*i+1,2*j+1]) / 16
        end
    end
    return nothing
end

function matrix_free_restriction_2d_GPU(idata,odata)
    (Nx,Ny) = size(idata)
    idata_tmp = CuArray(zeros(Nx+2,Ny+2))
    copyto!(view(idata_tmp,2:Nx+1,2:Ny+1),idata)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx+TILE_DIM_1-1,TILE_DIM_1), div(Ny+TILE_DIM_2-1,TILE_DIM_2))
	blockdim = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim blocks=griddim restriction_2D_kernel(idata_tmp,odata,Nx,Ny,Val(TILE_DIM_1),Val(TILE_DIM_2))
    nothing

end

function restriction_2D_kernel(idata_tmp,odata,Nx,Ny,::Val{TILE_DIM_1},::Val{TILE_DIM_2}) where {TILE_DIM_1,TILE_DIM_2}
    tidx = threadIdx().x
    tidy = threadIdx().y
    i = (blockIdx().x - 1) * TILE_DIM_1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM_2 + tidy

    # idata_tmp = CuArray(zeros(Nx+2,Ny+2))
    # idata_tmp[2:end-1,2:end-1] .= idata

    size_odata = (div(Nx+1,2),div(Ny+1,2))

    if 1 <= i <= size_odata[1] && 1 <= j <= size_odata[2]
        odata[i,j] = (4*idata_tmp[2*i,2*j] + 
        2 * (idata_tmp[2*i,2*j-1] + idata_tmp[2*i,2*j+1] + idata_tmp[2*i-1,2*j] + idata_tmp[2*i+1,2*j]) +
         (idata_tmp[2*i-1,2*j-1] + idata_tmp[2*i-1,2*j+1] + idata_tmp[2*i+1,2*j-1]) + idata_tmp[2*i+1,2*j+1]) / 16
        # odata[i,j] = idata_tmp[2*i,2*j]
        # odata[i,j] = 1
    end
   
    return nothing
end

function matrix_free_richardson(idata_GPU,odata_GPU,b_GPU;maxiter=3,ω=0.15)
    for _ in 1:maxiter
        matrix_free_A_full_GPU(idata_GPU,odata_GPU) # matrix_free_A_full_GPU is -A here, becareful
        odata_GPU .= idata_GPU .+ ω * (b_GPU .+ odata_GPU)
        idata_GPU .= odata_GPU
    end
end


function matrix_free_Two_level_multigrid(b_GPU;nu=3,NUM_V_CYCLES=1,p=2)
    (Nx,Ny) = size(b_GPU)
    level = Int(log(2,Nx-1))
    (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = Assembling_matrix(level-1,p=p)
    v_values_GPU = Dict(1=>CuArray(zeros(Nx,Ny)))
    v_values_out_GPU = Dict(1=>CuArray(zeros(Nx,Ny)))
    Av_values_out_GPU = Dict(1=>CuArray(zeros(Nx,Ny)))
    rhs_values_GPU = Dict(1=>b_GPU)
    N_values = Dict(1=>Nx)
    N_values[2] = div(Nx+1,2)
    f_GPU = Dict(1=>CuArray(zeros(Nx_2h,Ny_2h)))
    e_GPU = Dict(1=>CuArray(zeros(Nx,Ny)))
    
    for cycle_number in 1:NUM_V_CYCLES
        matrix_free_richardson(v_values_GPU[1],v_values_out_GPU[1],rhs_values_GPU[1];maxiter=nu)
        matrix_free_A_full_GPU(v_values_out_GPU[1],Av_values_out_GPU[1])
        r_GPU = b_GPU + Av_values_out_GPU[1]
        matrix_free_restriction_2d(r_GPU,f_GPU[1])
        # v_values_GPU[2] = reshape(CuArray(A_2h) \ f_GPU[1][:],Nx_2h,Ny_2h)
        # v_values_GPU[2] = reshape(CUDA.CUSPARSE.CuSparseMatrixCSC(A_2h) \ f_GPU[1][:],Nx_2h,Ny_2h)
        v_values_GPU[2] = reshape(CuArray(A_2h \ Array(f_GPU[1][:])),Nx_2h,Ny_2h)
        matrix_free_prolongation_2d(v_values_GPU[2],e_GPU[1])
        v_values_GPU[1] .+= e_GPU[1]
        matrix_free_richardson(v_values_GPU[1],v_values_out_GPU[1],rhs_values_GPU[1];maxiter=nu)
    end
    matrix_free_A_full_GPU(v_values_out_GPU[1],Av_values_out_GPU[1])
    return (v_values_out_GPU[1],norm(-Av_values_out_GPU[1]-b_GPU))
end


function matrix_free_MGCG(b_GPU,x_GPU;maxiter=length(b),abstol=sqrt(eps(real(eltype(b)))),NUM_V_CYCLES=1,nu=3,use_galerkin=true,direct_sol=0,H_tilde=0,p=2)
    Ax_GPU = CuArray(zeros(size(x_GPU)))
    matrix_free_A_full_GPU(x_GPU,Ax_GPU)
    r_GPU = b_GPU + Ax_GPU
    z_GPU = matrix_free_Two_level_multigrid(r_GPU)[1]
    p_GPU = copy(z_GPU)
    Ap_GPU = copy(p_GPU)
    num_iter_steps_GPU = 0
    norms_GPU = [norm(r_GPU)]
    errors_GPU = []
    if direct_sol != 0 && H_tilde != 0
        append!(errors,sqrt(direct_sol' * A * direct_sol)) # need to rewrite
    end

    rzold_GPU = sum(r_GPU .* z_GPU)

    for step = 1:maxiter
        num_iter_steps_GPU += 1
        matrix_free_A_full_GPU(p_GPU,Ap_GPU)
        alpha_GPU = - rzold_GPU / sum(p_GPU .* Ap_GPU)
        x_GPU .+= alpha_GPU .* p_GPU
        r_GPU .+= alpha_GPU .* Ap_GPU
        rs_GPU = sum(r_GPU .* r_GPU)
        append!(norms_GPU,sqrt(rs_GPU))
        if direct_sol != 0 && H_tilde != 0
            error = sqrt((x - direct_sol)' * A * (x - direct_sol)) # need to rewrite
            # @show error
            append!(errors,error)
        end
        if sqrt(rs_GPU) < abstol
            break
        end
        z_GPU .=  matrix_free_Two_level_multigrid(r_GPU)[1]
        rznew_GPU = sum(r_GPU .* z_GPU)
        beta_GPU = rznew_GPU / rzold_GPU
        p_GPU .= z_GPU .+ beta_GPU .* p_GPU
        rzold_GPU = rznew_GPU
    end
    return num_iter_steps_GPU, norms_GPU
end


function test_matrix_free_MGCG(;level=6,nu=3,ω=2/3,SBPp=2)
    (A,b,H_tilde,Nx,Ny) = Assembling_matrix(level,p=SBPp);
    direct_sol = A\b
    reltol = sqrt(eps(real(eltype(b))))
    x = zeros(Nx*Ny);
    abstol = norm(A*x-b) * reltol

    x_GPU = CuArray(zeros(Nx,Ny))
    b_GPU = CuArray(reshape(b,Nx,Ny))

    num_iter_steps_GPU, norms_GPU = matrix_free_MGCG(b_GPU,x_GPU;maxiter=length(b_GPU),abstol=abstol)
    iter_mg_cg, norm_mg_cg, error_mg_cg = mg_preconditioned_CG(A,b,x;maxiter=length(b),abstol=abstol,NUM_V_CYCLES=1,nu=nu,use_galerkin=true,direct_sol=direct_sol,H_tilde=H_tilde,p=SBPp)
    @show norms_GPU
    @show norm_mg_cg


    REPEAT = 5

    t_matrix_free_GPU = @elapsed for _ in 1:REPEAT
        x_GPU = CuArray(zeros(Nx,Ny))
        matrix_free_MGCG(b_GPU,x_GPU;maxiter=length(b_GPU),abstol=abstol)
    end

    t_CPU = @elapsed for _ in 1:REPEAT
        x = zeros(Nx*Ny)
        mg_preconditioned_CG(A,b,x;maxiter=length(b),abstol=abstol,NUM_V_CYCLES=1,nu=nu,use_galerkin=true,direct_sol=direct_sol,H_tilde=H_tilde,p=SBPp)
    end

    t_matrix_free_GPU ./ REPEAT
    t_CPU ./ REPEAT

    @show t_matrix_free_GPU
    @show t_CPU

    return nothing
end

let
    level = 6
    N = 2^level + 1
    Random.seed!(0)
    idata = randn(N,N)
    idata_flat = idata[:]
    idata_GPU = CuArray(idata)
    odata_GPU = CuArray(zeros(N,N))
    
    x = zeros(length(idata_flat))
    x_GPU_flat = CuArray(x)
    odata_reshaped = reshape(prolongation_2d(N)*idata_flat,2*N-1,2*N-1)

    size_idata = size(idata)
    odata_prolongation = zeros(2*size_idata[1]-1,2*size_idata[2]-1)
    odata_restriction = zeros(div.(size_idata .+ 1,2))

    odata_prolongation_GPU = CuArray(odata_prolongation)
    odata_restriction_GPU = CuArray(odata_restriction)

    matrix_free_restriction_2d(idata,odata_restriction)
    matrix_free_prolongation_2d(idata,odata_prolongation)

    @assert odata_restriction[:] ≈ restriction_2d(N) * idata_flat
    @assert odata_prolongation[:] ≈ prolongation_2d(N) * idata_flat

    matrix_free_prolongation_2d(idata_GPU,odata_prolongation_GPU)
    matrix_free_prolongation_2d_GPU(idata_GPU,odata_prolongation_GPU)

    matrix_free_restriction_2d(idata_GPU,odata_restriction_GPU)
    matrix_free_restriction_2d_GPU(idata_GPU,odata_restriction_GPU)

    @assert odata_restriction ≈ Array(odata_restriction_GPU)
    @assert odata_prolongation ≈ Array(odata_prolongation_GPU)

    

    (A,b,H,Nx,Ny) = Assembling_matrix(level,p=2)

    maxiter=2
    modified_richardson!(idata_flat,A,b;maxiter=maxiter)
    b_GPU = CuArray(reshape(b,N,N))
    matrix_free_richardson(idata_GPU,odata_GPU,b_GPU;maxiter=maxiter)

    richardson_out_CPU = idata_flat + 0.15 * (b - A*idata_flat)
    matrix_free_A_full_GPU(idata_GPU,odata_GPU) # Be careful, matrix_free_A_GPU is -A here, with minus sign
    richardson_out_GPU = idata_GPU + 0.15 * (b_GPU + odata_GPU) #
end
