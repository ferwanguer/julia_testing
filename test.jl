
using Random, Distributions, Plots,LinearAlgebra

x = ones(10)
println(size(x))
sum(x, dims = 2)
c = Diagonal(x)
println(c)
include("test_functions.jl")

#g(x)

#rand(truncated(Normal(x, x),n.lower_bound,n.upper_bound))
n_dims = 100
lower_bound = -5*ones(n_dims)
upper_bound = 5*ones(n_dims)

#μ_0 =  rand(Uniform(lower_bound, upper_bound),n_dims)

Uniform.(lower_bound, upper_bound)