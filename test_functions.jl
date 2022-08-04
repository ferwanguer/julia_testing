
function g(x)
    n_dims = size(x)[2]
    #println(n_dims)

    a = 20.0
    b = 0.2
    c = 2 * pi

    auxiliar = -b .* sqrt.((sum(x.^2, dims=2)./ n_dims))
    auxiliar_2 = sum(cos.(c .* x), dims = 2) ./ n_dims
    g = - a .* exp.(auxiliar) .- exp.(auxiliar_2) .+ a .+ exp(1.0)

end


#Pending to add griewank function 