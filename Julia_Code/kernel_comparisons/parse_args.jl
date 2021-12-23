if length(ARGS) != 0
    N = parse(Int,ARGS[1])
else
    N = 1
end

@show typeof(N)

println("N = $N")