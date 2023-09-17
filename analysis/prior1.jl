#=
    prior1.jl

    First level prior.
    
=#
using Distributions

prior1 = Product(Uniform.(
    [-10.0, 0.0,-10.0,-10.0,-10.0, -5.0],
    [  0.0, 5.0,  0.0,  3.0,  3.0,  0.0]))

prior1α = Product(Uniform.(
    [-10.0, 0.0,-10.0,-10.0,-10.0, -5.0,-10.0,-10.0],
    [  0.0, 5.0,  0.0,  3.0,  3.0,  0.0,  0.0,  0.0]))