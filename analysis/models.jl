#=
    models.jl

    Code relating to the mathematical model

=#

using DifferentialEquations

#################################################################
## Two-compartment model
#################################################################

function solve_model_twocompartment(p;tdose=[],tend=:auto,output=:all,verbose=false)

    if tend == :auto
        tend = isempty(tdose) ? 50.0 : maximum(tdose)
    end

    λ,K,γ,ζ,η,ϕ₀ = p

    function ode!(dx,x,p,t)
        V,N = x
        λ,K,γ,ζ,η = p
        dx[1] = λ * V * (1 - V / K) - η * V
        dx[2] = η * V - ζ * N
    end
    function dose!(int)
        λ,K,γ,ζ,η = int.p
        V = int.u[1]
        int.u[1] -= γ * V
        int.u[2] += γ * V
    end

    sol = solve(
        ODEProblem(ode!,[1 - ϕ₀, ϕ₀],(0.0,tend),[λ,K,γ,ζ,η]),
        callback=DiscreteCallback((u,t,int) -> t ∈ tdose,dose!),
        tstops=tdose,verbose=verbose
    )

    if output == :all
        return sol
    elseif output == :total
        return t -> sum(sol(t))
    elseif output == :both
        return t -> [sum(sol(t)) sol(t)[2]]
    else
        error("Unknown model output!")
    end

end

twocompartment_params = ["λ","K","γ","ζ","η","ϕ₀"]
twocompartment_params_expanded = ["λ","K","γ","ζ","η","ϕ₀","α₁","α₂"]