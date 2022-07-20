
using Random, Distributions, Plots
include("test_functions.jl")

mutable struct NormalFeature
    μ:: Float64
    σ:: Float64
    lower_bound::Float64
    upper_bound::Float64
    ρ_μ:: Float32
    ρ_σ:: Float32
 end

function quantum_sampling(n::NormalFeature, n_samples = 6)
    # println(String("The mean is equal to $(normal_individual.μ)" ))
    d = truncated(Normal(n.μ, n.σ),
    n.lower_bound,n.upper_bound)

    return rand(d, n_samples)
end

function elitist_sample_evaluation(samples :: Matrix, cost_function::Function, elitism::Int = 5)
    costs = cost_function(samples)
    
    sort_order = sortperm(vec(costs))
    return mean(samples[sort_order[1:elitism],:], dims = 1)
end

function quantum_update(bpi_feature,feature::NormalFeature)
    mu_delta = bpi_feature - feature.μ
    feature.μ += mu_delta./feature.ρ_μ
    
    sigma_decider = abs(mu_delta)./feature.σ
    feature.σ = (sigma_decider >= 1)*feature.σ*feature.ρ_σ + 
    (sigma_decider < 1)*feature.σ/feature.ρ_σ 

    return
    
end


@time Individuals = [NormalFeature(2,  5,-5.0,5.0,100,1.0001) for i in 1:100]

function Training(Individuals, N_iterations = 20000)
    for i in 1:N_iterations
        samples = mapreduce(quantum_sampling,hcat,Individuals)
        bpii = elitist_sample_evaluation(samples,g,3)
        map(quantum_update,bpii,Individuals)
        if mod(i,5000) == 0
            println(i,g(bpii))
        end
    end
end
@time Training(Individuals)
print("End")