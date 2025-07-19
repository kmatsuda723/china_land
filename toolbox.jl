using Interpolations
using NLsolve
using Parameters # enable @unpack
# using Printf
using Plots
using Optim
using Roots
# using CSV
# using DataFrames
using Distributions
# using GLM


function tauchen(N, rho, sigma, param)
    """
    ---------------------------------------------------
    === Function to Discretize an AR(1) Process Using the Tauchen Method ===
    ---------------------------------------------------
    Assumes: z' = ρ * z + ε, with ε ~ N(0, σ_ε²), and discretizes it.

    <input>
    ・N: Number of grid points for discretization  
    ・rho: Persistence of the AR(1) process (ρ in the equation above)  
    ・sigma: Standard deviation of the shock term in the AR(1) process (σ_ε above)  
    ・param: Parameter that controls the range of the grid for discretization  

    <output>
    ・Z: Discretized grid of the AR(1) process  
    ・Zprob: Transition matrix across grid points  
    ・Zinv: Stationary distribution over Z  
    """
    Zprob = zeros(N, N) # prob of transition matrix
    Zinv = zeros(N, 1)  # stationary prob

    # Define an equally spaced grid
    # max and min
    zmax = param * sqrt(sigma^2 / (1 - rho^2))
    zmin = -zmax
    # space
    w = (zmax - zmin) / (N - 1)

    Z = collect(range(zmin, zmax, length=N))

    # determining prob given grids
    for j in 1:N # index today
        for jp in 1:N  # index tomorrow
            if jp == 1
                Zprob[j, jp] = cdf_normal((Z[jp] - rho * Z[j] + w / 2) / sigma)
            elseif jp == N
                Zprob[j, jp] = 1 - cdf_normal((Z[jp] - rho * Z[j] - w / 2) / sigma)
            else
                Zprob[j, jp] = cdf_normal((Z[jp] - rho * Z[j] + w / 2) / sigma) - cdf_normal((Z[jp] - rho * Z[j] - w / 2) / sigma)
            end
        end
    end

    # stationary distribution
    dist0 = (1 / N) .* ones(N)
    dist1 = copy(dist0)

    err = 1.0
    errtol = 1e-8
    iter = 1
    while err > errtol

        dist1 = Zprob' * dist0
        err = sum(abs.(dist0 - dist1))
        dist0 = copy(dist1)
        iter += 1

    end

    Zinv = copy(dist1)

    return Z, Zprob, Zinv

end


function cdf_normal(x)
    """
    --------------------------------
    === Cumulative Distribution Function of Standard Normal ===
    --------------------------------
    <input>
    ・x: Value at which to evaluate the CDF  
    <output>
    ・c: Probability that a standard normal random variable X satisfies X ≤ x
    """
    d = Normal(0, 1) # 標準正規分布
    c = cdf(d, x)

    return c

end

function gini_coefficient(X::Vector{Float64})
    n = length(X)
    sorted_X = sort(X)
    # cumulative_X = cumsum(sorted_X)
    gini = (2 * sum((1:n) .* sorted_X)) / (n * sum(sorted_X)) - (n + 1) / n
    return gini
end

function gini_index(values::Vector{T}, probs::Vector{T}) where T
    μ = sum(probs .* values)  # Weighted mean
    n = length(values)
    gini_sum = 0.0

    for i in 1:n
        for j in 1:n
            gini_sum += probs[i] * probs[j] * abs(values[i] - values[j])
        end
    end

    return gini_sum / (2 * μ)
end

# function gini_index(values::Vector{<:Real}, probabilities::Vector{<:Real})
#     # Ensure probabilities sum to 1
#     if abs(sum(probabilities) - 1.0) > 1e-8
#         error("Probabilities must sum to 1.")
#     end
    
#     # Sort values and probabilities based on values
#     sorted_indices = sortperm(values)
#     sorted_values = values[sorted_indices]
#     sorted_probabilities = probabilities[sorted_indices]
    
#     # Compute Gini index
#     cumulative_values = cumsum(sorted_values .* sorted_probabilities)
#     weighted_sum = sum(cumulative_values .* sorted_probabilities)
#     mean_value = sum(sorted_values .* sorted_probabilities)
    
#     return 1.0 - 2.0 * weighted_sum / mean_value
# end

function rank_vector(X::Vector{Float64})
    # Sort X and get the original indices
    sorted_indices = sortperm(X)
    
    # Initialize a vector for ranks
    ranks = similar(X, Int)
    
    # Assign ranks based on the sorted indices
    for rank in 1:length(X)
        ranks[sorted_indices[rank]] = rank
    end
    
    return ranks
end

# making nlsove easier
function nls(func, params...; ini=[0.0])
    if typeof(ini) <: Number
        r = nlsolve((vout, vin) -> vout[1] = func(vin[1], params...), [ini])
        v = r.zero[1]
    else
        r = nlsolve((vout, vin) -> vout .= func(vin, params...), ini)
        v = r.zero
    end
    return v, r.f_converged
end

function interp(x, grid)
    # Find indices of the closest grids and the weights for linear interpolation
    ial = searchsortedlast(grid, x)  # Index of the grid just above or equal to x
    ial = max(1, ial)  # Ensure index is within bounds

    if ial>length(grid)-1
        ial = length(grid)-1  # Handle case where x is beyond the grid
    end

    iar = ial + 1  # The index just below ial

    # Compute the weights for interpolation
    varphi = (grid[iar] - x) / (grid[iar] - grid[ial])
    return ial, iar, varphi
end


# Golden-section search for finding the minimum
function golden_section_search(f, a, b; tol=1e-5, max_iter=1000)
    φ = (sqrt(5) - 1) / 2  # Golden ratio factor

    # Initial points
    c = b - φ * (b - a)
    d = a + φ * (b - a)

    iter = 0
    while (b - a) > tol && iter < max_iter
        if f(c) < f(d)
            b = d
        else
            a = c
        end

        # Update points
        c = b - φ * (b - a)
        d = a + φ * (b - a)

        iter += 1
    end




    # Return the midpoint as the approximate minimum point
    return (a + b) / 2
end

function normal_discrete(mu, sigma, N)
    prob_dist = zeros(N)
    grid = collect(range(mu-2*sigma, stop=mu+2*sigma, length=N))
    step = grid[2] - grid[1]
    for i in 1:N
        if i == 1
            prob_dist[i] = cdf(Normal(mu, sigma), grid[i] + step / 2)
        elseif i == N
            prob_dist[i] = 1 - cdf(Normal(mu, sigma), grid[i] - step / 2)
        else
            prob_dist[i] = cdf(Normal(mu, sigma), grid[i] + step / 2) -
                               cdf(Normal(mu, sigma), grid[i] - step / 2)
        end
    end
    return grid, prob_dist
end

function sample_states_from_distribution(distribution_matrix::AbstractArray, NN::Int)
    # Flatten the distribution to 1D
    distribution_flat = vec(distribution_matrix)

    # Sanity check: probabilities should sum to ≈1
    if abs(sum(distribution_flat) - 1.0) > 1e-6
        @warn "Distribution does not sum to 1.0 (total=$(sum(distribution_flat)))"
    end

    # Generate all possible index tuples for the original shape
    possible_states = CartesianIndices(size(distribution_matrix))

    # Sample indices according to the distribution
    sampled_indices = rand(Categorical(distribution_flat), NN)

    # Map sampled flat indices to multi-dimensional indices (tuples)
    initial_states = Tuple.(possible_states[sampled_indices])

    return initial_states
end

    function sample_with_weights(array, weights)
        return sample(array, Weights(weights))
    end
