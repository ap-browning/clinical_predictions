#=
    Produce the results in table 1.
=#

using CSV
using DataFrames, DataFramesMeta
using Distributions
using LinearAlgebra
using StatsBase

################################################
## LOAD DATA
################################################

posterior1 = CSV.read("analysis/posterior1.csv",DataFrame)
prior2 = CSV.read("analysis/prior2.csv",DataFrame)

################################################
## BREAKDOWN OF CLASSES
################################################

P1 = [count(posterior1.class .== i) / nrow(posterior1) for i = 1:4]
P2 = [count(prior2.class .== i) / nrow(prior2) for i = 1:4]
df2 = DataFrame(class=1:4,posterior1=P1,prior2=P2)

################################################
## STATISTICS (POSTERIOR1)
################################################

# Number of samples per patient
n = nrow(@subset(posterior1,:id .== posterior1.id[1]))

# Mean and std
df2.mean = μ = [mean(@subset(posterior1,:class .== i).init_volume) for i = 1:4]
df2.std = σ = [std(@subset(posterior1,:class .== i).init_volume) for i = 1:4]

# Effective number of patients
df2.n = n = P1 * length(unique(posterior1.id))

################################################
## EXPAND TO "RESPONDERS" (CODE AS CLASS = 5)
################################################

df2 = vcat(df2,DataFrame(
    class = 5,
    posterior1 = sum(P1[[1,3,4]]),
    prior2 = sum(P2[[1,3,4]]),
    mean = mean(@subset(posterior1,:class .!= 2).init_volume),
    std = std(@subset(posterior1,:class .!= 2).init_volume),
    n = sum(df2.n[[1,3,4]])
))

################################################
## HYPOTHESIS TESTING
################################################

# Approximate Welch t-test (accounting for non-integer sample sizes)
function approx_welch_ttest(μ,σ,n)
    s = @. σ / √n
    t = abs(-(μ...) / norm(s))
    ν = norm(s)^4 / (σ[1]^4 / n[1]^2 / (n[1] - 1) + σ[2]^4 / n[2]^2 / (n[2] - 1))
    2*(1 - cdf(TDist(ν),t))
end

# Fast vs poor responders
p1 = approx_welch_ttest(μ[1:2],σ[1:2],n[1:2])
# 0.5821546964874191

# Responders vs poor responders
p2 = approx_welch_ttest(df2.mean[[2,5]],df2.std[[2,5]],df2.n[[2,5]])
# 0.5567106232337893

################################################
## SAVE RESULTS
################################################

CSV.write("analysis/table1.csv",df2)