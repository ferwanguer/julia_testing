
using Random, Distributions, Plots
include("test_functions.jl")

mutable struct NormalFeature
    μ:: Float64
    σ:: Float64
    lower_bound::Float64
    upper_bound::Float64
    ρ_μ:: Float64
    ρ_σ:: Float64
 end

 function quantum_init(n_dims = 100)
    return [NormalFeature(2,  5,-5.0,5.0,80,1.0001) for i in 1:n_dims]
 end

function quantum_sampling(n::NormalFeature)
    # println(String("The mean is equal to $(normal_individual.μ)" ))
    
    return rand(truncated(Normal(n.μ, n.σ),
    n.lower_bound,n.upper_bound))
end

function elitist_sample_evaluation(samples :: Matrix, cost_function::Function, elitism::Int = 5)   
    #sort_order = sortperm(vec(cost_function(samples)))
    return mean(samples[sortperm(vec(cost_function(samples)))[1:elitism],:], dims = 1)
end

function quantum_update(bpi_feature,feature::NormalFeature)
    mu_delta = bpi_feature - feature.μ
    feature.μ += mu_delta./feature.ρ_μ
    
    sigma_decider = abs(mu_delta)./feature.σ
    feature.σ = (sigma_decider >= 1)*feature.σ*feature.ρ_σ + 
    (sigma_decider < 1)*feature.σ/feature.ρ_σ 

    return nothing
    
end


#Individuals = quantum_init()
function sample_mapping(Individuals::Vector{NormalFeature},n_samples = 6)
    samples = Array{Float64}(undef, n_samples, length(Individuals))#zeros(6,length(Individuals))
    for i in 1:n_samples
        for j in 1:length(Individuals)
            samples[i,j] = quantum_sampling(Individuals[j])
        end
   end
    
return samples
end

function update_mapping(Individuals,best_performing_individual)
    for i in 1:length(Individuals)
        quantum_update(best_performing_individual[i],Individuals[i])
    end
    return nothing
end


function Training(Individuals = quantum_init(),N_iterations = 1)
    for i in 1:N_iterations
        samples = sample_mapping(Individuals,10)
        bpii = elitist_sample_evaluation(samples,g,3)
        #map(quantum_update,bpii,Individuals)
        update_mapping(Individuals,bpii)
        if mod(i,5000) == 0
           println(i,g(bpii))
        end
        
    end
    #return samples
end
@time Training(quantum_init(),200000)
print("End")