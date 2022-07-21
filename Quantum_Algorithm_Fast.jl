
using Random, Distributions, Plots, LinearAlgebra
include("test_functions.jl")

mutable struct NormalFeature
    μ:: Vector{Float64}
    σ:: Vector{Float64}
    lower_bound::Vector{Float64}
    upper_bound::Vector{Float64}
    ρ_μ:: Float64
    ρ_σ:: Float64
 end

 function quantum_init(ρ_μ= 100,ρ_σ = 1.001,n_dims = 1000000)
    lower_bound = -5*ones(n_dims)
    upper_bound = 5*ones(n_dims)
    μ_0 =  rand.(Uniform.(lower_bound, upper_bound))
    σ_0 = upper_bound - lower_bound
    return NormalFeature(μ_0,σ_0, lower_bound,upper_bound, ρ_μ, ρ_σ)
  
 end



 function quantum_sampling(n::NormalFeature,n_samples= 100)

    return reduce(hcat,rand.(truncated.(Normal.(n.μ,n.σ),n.lower_bound,n.upper_bound), n_samples))
    
 end

  
 

 function elitist_sample_evaluation(samples :: Matrix, cost_function::Function, elitism::Int = 5)   
    #sort_order = sortperm(vec(cost_function(samples)))
    return mean(samples[sortperm(vec(cost_function(samples)))[1:elitism],:], dims = 1)
end


function quantum_update(bpi::Vector,features::NormalFeature)
    mu_delta = bpi - features.μ
    features.μ += mu_delta./features.ρ_μ
    
    sigma_decider = abs.(mu_delta)./features.σ
    features.σ = (sigma_decider .>= 1).*features.σ.*features.ρ_σ + 
    (sigma_decider .< 1).*features.σ./features.ρ_σ 

    return nothing
    
end



function training(N_iterations = 2000)
    n = quantum_init(80, 1.004, 20)
    for i in 1:N_iterations
        samples = quantum_sampling(n,20)
        bpi = elitist_sample_evaluation(samples,g,5)
        quantum_update(vec(bpi), n)
        if mod(i,100) == 0
            println(i,g(bpi))
        end
    end
end
@time training()