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

function matrix_free_richardson(idata_GPU,odata_GPU,b_GPU;maxiter=3,ω=0.15)
    for _ in 1:maxiter
        matrix_free_A_full_GPU(idata,odata) # matrix_free_A_full_GPU is -A here, becareful
        odata .= idata .+ ω * (b .+ odata)
        idata .= odata
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
        v_values_GPU[2] = reshape(CuArray(A_2h) \ f_GPU[1][:],Nx_2h,Ny_2h)
        matrix_free_prolongation_2d(v_values_GPU[2],e_GPU[1])
        v_values_GPU[1] .+= e_GPU[1]
        matrix_free_richardson(v_values_GPU[1],v_values_out_GPU[1],rhs_values_GPU[1];maxiter=nu)
    end
    matrix_free_A_full_GPU(v_values_out_GPU[1],Av_values_out_GPU[1])
    return (v_values_out_GPU[1],norm(-Av_values_out_GPU[1]-b_GPU))
end


function matrix_free_MGCG(b_GPU)

end


let
    level = 2
    N = 2^level + 1
    Random.seed!(0)
    idata = randn(N,N)
    idata_flat = idata[:]
    idata_GPU = CuArray(idata)
    odata_GPU = CuArray(zeros(N,N))
    
    x = zeros(length(idata_flat))
    x_GPU = CuArray(x)
    odata_reshaped = reshape(prolongation_2d(5)*idata_flat,9,9)

    size_idata = size(idata)
    odata_prolongation = zeros(2*size_idata[1]-1,2*size_idata[2]-1)
    odata_restriction = zeros(div.(size_idata .+ 1,2))

    matrix_free_restriction_2d(idata,odata_restriction)
    matrix_free_prolongation_2d(idata,odata_prolongation)

    @assert odata_restriction[:] ≈ restriction_2d(N) * idata_flat
    @assert odata_prolongation[:] ≈ prolongation_2d(N) * idata_flat

    (A,b,H,Nx,Ny) = Assembling_matrix(level,p=2)

    maxiter=2
    modified_richardson!(idata_flat,A,b;maxiter=maxiter)
    b_GPU = CuArray(reshape(b,N,N))
    matrix_free_richardson(idata_GPU,odata_GPU,b_GPU;maxiter=maxiter)

    richardson_out_CPU = idata_flat + 0.15 * (b - A*idata_flat)
    matrix_free_A_full_GPU(idata_GPU,odata_GPU) # Be careful, matrix_free_A_GPU is -A here, with minus sign
    richardson_out_GPU = idata_GPU + 0.15 * (b_GPU + odata_GPU) #
end
