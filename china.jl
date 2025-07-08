# ======================================================= #

# ======================================================= #

# import libraries
using Plots
using Optim
using Random
using Distributions
using LaTeXStrings
using Parameters # enable @unpack
using DataFrames
using CSV
using StatsBase

Random.seed!(1234)  # Set random seed

include("toolbox.jl")
include("types.jl")

# p = setParameters(
#         beta=params[1],
#         zeta_rr=params[2],
#         zeta_ua=params[3],
#         zeta_ru=params[4],# sigma_e=params[4]
#         sigma_e=params[5],
#         r_land=params[6]
#     )

# function setParameters(;
#     mu=2.0,             # risk aversion (=3 baseline)             
#     beta=0.4260874772544608,            # subjective discount factor 
#     delta=0.08,            # depreciation
#     alpha=0.36,            # capital'h share of income
#     b=0.0,             # borrowing limit
#     NH=7,             # number of discretized states
#     rho=0.6,           # first-order autoregressive coefficient
#     gamma_h=0.0,
#     gamma_z=0.12,
#     gamma_q=0.1,
#     zeta_ua=0.06394650928365665,
#     r_land=0.11569881101974472,
#     phi_a=0.5,
#     delta_n=0.27,
#     sigma_e=0.19935604769043203,
#     zeta_ru=0.46726019527600726,
#     zeta_rr=-0.25721928801561034,
#     income_thres_top10_base=0.0,
#     sig=1.0           # intermediate value to calculate sigma (=0.4 BASE)
# )

#

function setParameters(;
    mu=2.0,                    # Risk aversion coefficient (baseline = 3)
    beta=0.49777869220275955,  # Subjective discount factor
    delta=0.08,                # Depreciation rate
    alpha=0.36,                # Capital's share of income
    b=0.0,                     # Borrowing limit
    NH=10,                      # Number of discretized labor productivity states
    rho=0.6,                   # AR(1) coefficient for labor productivity process
    gamma_h=0.0,               # Parameter for additional labor-related processes (unused here)
    gamma_z=0.12,              # Parameter for additional shock process (unused here)
    gamma_q=0.1,               # Parameter for another process (unused here)
    zeta_ua=0.09714970304225913,  # Utility discount or cost parameter for urban agricultural group
    r_land=0.047434872244243725,    # Land return rate
    phi_a=0.5,                 # Parameter related to land risk for agricultural hukou
    delta_n=0.27,              # Additional land risk parameter
    sigma_e=0.21509367659714868,  # Scale parameter for idiosyncratic shocks
    zeta_ru=0.31666396394931695,    # Utility discount or cost parameter for rural urban group
    zeta_rr=-0.3479186418550472,   # Utility discount or cost parameter for rural rural group
    income_thres_top10_base=0.0,   # Base threshold for income top 10%
    sig=1.0                    # Std deviation of shock noise for Tauchen method (0.4 baseline)
)

    # Effective land risk for non-agricultural urban hukou
    phi_n = phi_a + delta_n

    # ========================================================= #
    #  COMPUTE TRANSITION MATRIX OF LABOR PRODUCTIVITY (TAUCHEN) #
    # ========================================================= #
    # Using Tauchen method to discretize AR(1) process for labor shocks:
    #   log(s_t) = rho * log(s_{t-1}) + e_t, where e_t ~ N(0, sig^2)
    # Outputs: lz (grid), prob (transition matrix), invdist (stationary distribution)

    M = 2.0  # Width parameter for Tauchen grid
    NZ = 5   # Number of states in labor shock grid

    lz, prob, invdist = tauchen(NZ, rho, sqrt(1.0 - rho^2), M)
    z = exp.(lz)  # Convert from logs to levels

    # Discretized human capital (labor supply) grid
    lh = collect(range(-0.5, stop=0.5, length=NH))
    h = exp.(lh)  # Levels

    # ========================================================= #
    #  HUMAN CAPITAL INVESTMENT GROUPS                          #
    # ========================================================= #
    # Group codes for educational / hukou types:
    #   1: rr (earner + kid in rural)
    #   2: ru (kid in rural, earner in urban)
    #   3: ua (both urban agricultural hukou)
    #   4: un (both urban non-agricultural hukou)

    NI = 4          # Number of groups
    NA = 30         # Grid size for asset state (k)
    NA2 = 30        # Grid size for asset control (k')

    # Log of group-specific quality/skill level (q)
    lq = zeros(NI)
    lq[1] = log(0.72)
    lq[2] = log(0.72)
    lq[3] = log(0.88)
    lq[4] = log(1.0)

    # Mobility cost matrix (currently zeros)
    mv_cost = zeros(NI, NI)

    # Land risk by group
    land_risk = zeros(NI)
    land_risk[3] = phi_a
    land_risk[4] = phi_n

    # Utility discount or cost by group
    dutil = zeros(NI)
    dutil[3] = zeta_ua
    dutil[2] = zeta_ru
    dutil[1] = zeta_rr

    return Params(
        mu, beta, delta, alpha, b, gamma_z, gamma_q, gamma_h, NH,
        h, z, lh, lz, lq, prob, income_thres_top10_base,
        NA, NA2, NZ, NI, mv_cost, land_risk, r_land, sigma_e, dutil, zeta_rr
    )
end

function log_hplus(logz, logq, logh, logxi, p::Params)
    return p.gamma_z * logz + p.gamma_q * logq + p.gamma_h * logh + logxi
end

function set_prices(p::Params, KL, land_lost, avg_income)::Prices
    # Use a fixed interest rate
    r = 1.021^30 - 1.0

    # Set wage vector
    wage = zeros(p.NI)
    wage[1] = 1.0
    wage[2:4] .= 1.35

    # Borrowing constraint (phi)
    phi = p.b

    # Asset grid for state variables
    a_u = 0.5
    a_l = -phi
    curvK = 1.1

    a = zeros(p.NA)
    a[1] = a_l
    for ia in 2:p.NA
        a[ia] = a[1] + (a_u - a_l) * ((ia - 1) / (p.NA - 1))^curvK
    end

    # Asset grid for optimal choice (control variable)
    gridk2 = zeros(p.NA2)
    gridk2[1] = a_l
    for ia in 2:p.NA2
        gridk2[ia] = gridk2[1] + (a_u - a_l) * ((ia - 1) / (p.NA2 - 1))^curvK
    end

    # Land supply inferred from land lost
    ell = 1.0 / (1.0 - land_lost)

    # Tuition by type
    tuition = zeros(p.NI)
    tuition[1] = 13471.0 / 30333.0 / 30 * avg_income
    tuition[2] = 13471.0 / 30333.0 / 30 * avg_income
    tuition[3] = 24151.0 / 30333.0 / 30 * avg_income
    tuition[4] = 54728.0 / 30333.0 / 30 * avg_income

    return Prices(r, wage, phi, a, gridk2, ell, avg_income, tuition, a_u)
end


function solve_household(p::Params, prices::Prices)
    # Initialize some variables
    iaplus = zeros(p.NI, p.NH, p.NA, p.NZ)           # Index of optimal k' choice
    aplus = similar(iaplus)                         # Optimal k' choice
    pplus = zeros(p.NI, p.NH, p.NA, p.NZ, p.NI)      # Probability distribution over category choices

    v = zeros(p.NI, p.NH, p.NA, p.NZ)                # Value function from previous iteration
    tv = similar(iaplus)                            # Updated value function

    err = 20.0
    maxiter = 2000
    iter = 1

    # Cache human capital interpolation
    lhplus_cache = [log_hplus(p.lz[iz], p.lq[ii], p.lh[ih], 0.0, p) for ii in 1:p.NI, ih in 1:p.NH, iz in 1:p.NZ]
    interp_lh_cache = [(interp(lhplus_cache[ii, ih, iz], p.lh)) for ii in 1:p.NI, ih in 1:p.NH, iz in 1:p.NZ]

    while (err > 0.01) & (iter < maxiter)
        @inbounds for ii in 1:p.NI
            vtemp = fill(-1e6, p.NA2, p.NI)
            ptemp = fill(-1e6, p.NA2, p.NI)
            vtemp2 = fill(-1e6, p.NA2)

            for ia in 1:p.NA        # state: asset
                for ih in 1:p.NH    # state: human capital
                    for iz in 1:p.NZ # state: productivity
                        fill!(vtemp, -1e6)
                        fill!(ptemp, -1e6)
                        fill!(vtemp2, -1e6)

                        for iap in 1:p.NA2       # control: next period's asset
                            for iip in 1:p.NI    # control: category choice

                                # Compute consumption
                                cons = prices.wage[ii] * p.h[ih] +
                                       (1.0 + prices.r) * prices.a[ia] -
                                       prices.gridk2[iap] - prices.tuition[iip] -
                                       p.mv_cost[ii, iip] + p.r_land * prices.ell * (1.0 - p.land_risk[ii])

                                if cons <= 0.0
                                    break  # Negative consumption → infeasible
                                end

                                # Compute utility
                                util = (max(cons, 1e-4)^(1.0 - p.mu)) / (1.0 - p.mu) - p.dutil[ii]

                                # Interpolation for next period's value
                                ial, iar, varphi = interp(prices.gridk2[iap], prices.a)
                                ihl, ihr, varphi_h = interp_lh_cache[ii, ih, iz]

                                vpr = 0.0
                                for izp in 1:p.NZ
                                    pz = p.prob[iz, izp]
                                    vpr += pz * (
                                        varphi_h * (varphi * v[iip, ihl, ial, izp] + (1.0 - varphi) * v[iip, ihl, iar, izp]) +
                                        (1.0 - varphi_h) * (varphi * v[iip, ihr, ial, izp] + (1.0 - varphi) * v[iip, ihr, iar, izp])
                                    )
                                end

                                vtemp[iap, iip] = util + p.beta * vpr
                            end

                            maxv = maximum(vtemp[iap, :])
                            sum_exp = sum(exp.((vtemp[iap, :] .- maxv) ./ p.sigma_e))
                            for iip in 1:p.NI
                                denom = sum(exp.((vtemp[iap, :] .- vtemp[iap, iip]) ./ p.sigma_e))
                                ptemp[iap, iip] = 1.0 / denom
                            end
                            vtemp2[iap] = maxv + p.sigma_e * log(sum_exp)
                        end

                        # Bellman maximization step
                        max_val, max_index = findmax(vtemp2)
                        tv[ii, ih, ia, iz] = max_val
                        iaplus[ii, ih, ia, iz] = max_index
                        aplus[ii, ih, ia, iz] = prices.gridk2[max_index]
                        pplus[ii, ih, ia, iz, :] = ptemp[max_index, :]
                    end
                end
            end
        end

        err = maximum(abs.(tv .- v))
        v .= tv
        iter += 1
    end

    if iter == maxiter
        println("WARNING!! @solve_household: iteration reached max: iter=$iter, err=$err")
    end

    return Dec(aplus, iaplus, pplus)
end

function get_distribution(p::Params, dec::Dec, prices::Prices)::Meas
    m = ones(p.NI, p.NH, p.NA, p.NZ) / (p.NI * p.NH * p.NA * p.NZ)  # Initial guess of distribution
    m_new = zeros(p.NI, p.NH, p.NA, p.NZ)                          # Updated distribution

    err = 1.0
    errTol = 1e-5
    maxiter = 2000
    iter = 1
    p_help = zeros(p.NI)

    while (err > errTol) && (iter < maxiter)
        @inbounds for ii in 1:p.NI
            for ia in 1:p.NA
                for ih in 1:p.NH
                    for iz in 1:p.NZ

                        lhplus = log_hplus(p.lz[iz], p.lq[ii], p.lh[ih], 0.0, p)
                        ihl, ihr, varphi_h = interp(lhplus, p.lh)

                        # Interpolate a'
                        ial, iar, varphi = interp(dec.aplus[ii, ih, ia, iz], prices.a)

                        # Probabilities of choosing education type
                        p_help .= dec.pplus[ii, ih, ia, iz, :]

                        for izp in 1:p.NZ
                            for iip in 1:p.NI
                                weight = p_help[iip] * p.prob[iz, izp] * m[ii, ih, ia, iz]
                                m_new[iip, ihl, ial, izp] += varphi_h * varphi * weight
                                m_new[iip, ihl, iar, izp] += varphi_h * (1.0 - varphi) * weight
                                m_new[iip, ihr, ial, izp] += (1.0 - varphi_h) * varphi * weight
                                m_new[iip, ihr, iar, izp] += (1.0 - varphi_h) * (1.0 - varphi) * weight
                            end
                        end

                    end
                end
            end
        end

        err = maximum(abs.(m_new .- m))
        m .= m_new
        iter += 1
        fill!(m_new, 0.0)
    end

    if iter == maxiter
        println("WARNING!! get_distribution: iteration reached max: iter = $iter, err = $err")
    end

    return Meas(m)
end

function aggregation(p::Params, dec::Dec, meas::Meas, prices::Prices)::Agg
    m = meas.m

    meanL = 0.0
    land_lost = 0.0
    avg_income = 0.0

    for ii in 1:p.NI
        for ia in 1:p.NA
            for ih in 1:p.NH
                for iz in 1:p.NZ
                    land_lost += p.land_risk[ii] * m[ii, ih, ia, iz]
                    meanL += p.h[ih] * m[ii, ih, ia, iz]
                    avg_income += prices.wage[ii] * p.h[ih] * m[ii, ih, ia, iz]
                end
            end
        end
    end

    meank = sum(dec.aplus .* m)

    # Marginal distributions
    mass_i = vec(sum(m, dims=(2, 3, 4)))
    mass_z = vec(sum(m, dims=(1, 2, 3)))
    mass_a = vec(sum(m, dims=(1, 2, 4)))
    mass_h = vec(sum(m, dims=(1, 3, 4)))

    return Agg(
        meank,
        meanL,
        land_lost,
        avg_income,
        mass_i,
        mass_z,
        mass_a,
        mass_h
    )
end

function get_Steadystate(p::Params, icase::Int)

    # Initial values
    KL = 6.8     # Capital-labor ratio guess
    land_lost = 0.3
    avg_income = 2.8

    err = 1.0
    errTol = 1e-2
    iter = 0
    maxiter = 50
    adj = 0.2

    # Preallocate
    prices = Prices(0.0, zeros(p.NI), 0.0, zeros(p.NA), zeros(p.NA2), 0.0, 0.0, zeros(p.NI), 0.0)
    dec = Dec(zeros(p.NI, p.NH, p.NA, p.NZ), zeros(p.NI, p.NH, p.NA, p.NZ), zeros(p.NI, p.NH, p.NA, p.NZ, p.NI))
    meas = Meas(zeros(p.NI, p.NH, p.NA, p.NZ))
    agg = Agg(0.0, 0.0, 0.0, 0.0, zeros(p.NI), zeros(p.NZ), zeros(p.NA), zeros(p.NH))

    while err > errTol && iter < maxiter
        iter += 1

        prices = set_prices(p, KL, land_lost, avg_income)
        dec = solve_household(p, prices)
        meas = get_distribution(p, dec, prices)
        agg = aggregation(p, dec, meas, prices)

        # Update values
        KL_new = agg.meank / agg.meanL
        land_lost_new = agg.land_lost
        avg_income_new = agg.avg_income

        # Error metric
        err = abs(avg_income - avg_income_new) / avg_income + abs(land_lost_new - land_lost)

        # Print diagnostics
        println((iter, round(avg_income, digits=4), round(avg_income_new, digits=4),
                      round(land_lost, digits=4), round(land_lost_new, digits=4),
                      round(err, digits=4)))

        # Update guesses with damping
        avg_income += adj * (avg_income_new - avg_income)
        land_lost += adj * (land_lost_new - land_lost)
        KL += adj * (KL_new - KL)
    end

    if iter == maxiter
        println("WARNING: did not converge. err = $err")
    end

    return p, dec, meas, prices, agg
end

function monte_carlo_simulation(p::Params, dec::Dec, meas::Meas, prices::Prices, NN, icase_sim)
    @unpack h, lz, lq, lh, r_land = p
    @unpack a, ell = prices

    ii_sim = zeros(Int, NN)
    iz_sim = zeros(Int, NN)
    ia_sim = zeros(Int, NN)
    ih_sim = zeros(Int, NN)

    # Initialize storage for household trajectories
    initial_states = sample_states_from_distribution(meas.m, NN)
    a_sim, h_sim, z_sim, wage_sim, earnings_sim, lhp_sim = zeros(NN), zeros(NN), zeros(NN), zeros(NN), zeros(NN), zeros(NN)

    Threads.@threads for i in 1:NN

        current_state = initial_states[i]
        ii_sim[i], ih_sim[i], ia_sim[i], iz_sim[i] = current_state[1], current_state[2], current_state[3], current_state[4]
        a_sim[i] = a[ia_sim[i]]
        wage_sim[i] = prices.wage[ii_sim[i]]
        h_sim[i] = h[ih_sim[i]]
        earnings_sim[i] = wage_sim[i] * h_sim[i]
        lhp_sim[i] = log_hplus(lz[iz_sim[i]], lq[ii_sim[i]], lh[ih_sim[i]], 0.0, p)
        if icase_sim==1 && (ii_sim[i]==3 || ii_sim[i]==4)
            lhp_sim[i] = log_hplus(lz[iz_sim[i]], lq[1], lh[ih_sim[i]], 0.0, p)
        end
    end

    # Return all simulated data as a NamedTuple
    return (a_sim=a_sim, h_sim=h_sim, wage_sim=wage_sim, earnings_sim=earnings_sim, ii_sim=ii_sim, lhp_sim=lhp_sim)
end



function output_gen(p::Params, dec::Dec, meas::Meas, prices::Prices, agg, icase)
    @unpack r_land, income_thres_top10_base = p
    @unpack ell = prices

    display(round.(agg.mass_i; digits=4))
    display(round.(agg.mass_h; digits=4))
    display(round.(agg.mass_a; digits=4))
    display(round.(agg.mass_z; digits=4))

    II = 50000

    sim = monte_carlo_simulation(p::Params, dec, meas, prices, II, 0)
    gridk0 = prices.a

    # Determine the threshold for the top 25% (75th percentile)
    income_threshold = quantile(sim.earnings_sim, 0.75)

    # Determine the threshold for the top 10% (90th percentile)
    income_thres_top10 = quantile(sim.lhp_sim, 0.90)

    # Get indices of individuals whose income is above the threshold
    top25_idx = findall(x -> x >= income_threshold, sim.earnings_sim)

    # Among those, count the number whose state is 3 or 4
    count34 = count(i -> sim.ii_sim[i] in (3, 4), top25_idx)
    share34 = count34 / length(top25_idx)

    # Get indices where state is 3 or 4
    idx_12 = findall(i -> sim.ii_sim[i] in (1, 2), sim.ii_sim)
    idx_34 = findall(i -> sim.ii_sim[i] in (3, 4), sim.ii_sim)

    # Count the number in the top 25% among those with state 3 or 4
    count_top25_in_34 = count(i -> sim.earnings_sim[i] >= income_threshold, idx_34)
    share_top25_in_34 = count_top25_in_34 / length(idx_34)


    # # Count the number in the top 25% among those with state 3 or 4
    count_top10_in_34 = count(i -> sim.lhp_sim[i] >= income_thres_top10, idx_34)
    share_top10_in_34 = count_top10_in_34 / length(idx_34)

    # error(sim.lhp_sim[idx_34])

    land_income_sim = r_land * ell ./ (sim.earnings_sim .+ r_land * ell)

    # Take the mean of land income share for state 1
    land_income_share_state1 = mean(land_income_sim[sim.ii_sim.==1])

    r_share = sum(meas.m[1:2, :, :, :]) / sum(meas.m[:, :, :, :])
    un_share = sum(meas.m[4, :, :, :]) / sum(meas.m[:, :, :, :])
    ua_share = sum(meas.m[3, :, :, :]) / sum(meas.m[:, :, :, :])
    ru_share = sum(meas.m[2, :, :, :]) / sum(meas.m[:, :, :, :])
    rr_share = sum(meas.m[1, :, :, :]) / sum(meas.m[:, :, :, :])


    college_thres = percentile(sim.lhp_sim, 90)
    # error(college_thresl)

    col_rate = zeros(p.NI)
    for ii in 1:p.NI
        inds = findall(sim.ii_sim .== ii)  # Get indices where ii_sim equals ii
        # Among those, calculate the share whose lhp_sim exceeds the college threshold
        count_above = count(i -> sim.lhp_sim[i] > college_thres, inds)
        # Proportion (relative to total in group)
        col_rate[ii] = count_above / length(inds)
    end

    # display(income_thres_top10)
    share_top10_in_34_cf = 0.0
    if icase==1
        sim = monte_carlo_simulation(p, dec, meas, prices, II, 1)
            # Count the number in the top 25% among those with state 3 or 4
    count_top10_in_34_cf = count(i -> sim.lhp_sim[i] >= income_thres_top10, idx_34)
    share_top10_in_34_cf = count_top10_in_34_cf / length(idx_34)
    # error([share_top10_in_34, share_top10_in_34_cf])
    end

    avg_earnings_rural = mean(sim.earnings_sim[(sim.ii_sim .== 1)])
    avg_earnings_urban = mean(sim.earnings_sim[(sim.ii_sim .== 2) .| (sim.ii_sim .== 3) .| (sim.ii_sim .== 4)])

    println("MOMENTS")
    println("")
    println("shares of rr, ru, ua, un")
    display(round(rr_share; digits=4))
    display(round(ru_share; digits=4))
    display(round(ua_share; digits=4))
    display(round(un_share; digits=4))
    println("")
    println("urban share of college")
    display(round(share_top10_in_34; digits=4))
    println("")
    println("rural/urban avg income")
    display(round(avg_earnings_rural/avg_earnings_urban; digits=4))

    if icase==1
    println("")
    println("validation: urban share of college with rural q")
    display(round(share_top10_in_34_cf; digits=4))
    end



    return (kfun0=dec.aplus, pplus=dec.pplus, gridk0=gridk0, KK=agg.meank, share_top25_in_34=share_top25_in_34, land_income_share_state1=land_income_share_state1,
        LL=agg.meanL, r_share=r_share, ua_share=ua_share, ru_share=ru_share, share34=share34, rr_share=rr_share, income_thres_top10=income_thres_top10,
        share_top10_in_34=share_top10_in_34, share_top10_in_34_cf=share_top10_in_34_cf)

end

function calibration(params_in)

    println("------------------------------")

    NMOM = 6

    model = zeros(NMOM)
    data = zeros(NMOM)
    dist = zeros(NMOM)
    params = zeros(NMOM)

    # Initialize model and data vectors directly
    params = [
        min(max(params_in[1], 0.0), 0.96),
        params_in[2],
        params_in[3],
        params_in[4],
        max(params_in[5], 1e-4),
        max(params_in[6], 0.0)
    ]

    println("parameters")
    display(params)
    println("")

    data = [
        3.53,  # K/Y
        0.445, # 0.445+0.139
        0.248,
        0.139, # 0.144/(1.0-0.584)# 0.139
        0.734,
        0.067
    ]

    # Set parameters and get steady state results
    p = setParameters(
        beta=params[1],
        zeta_rr=params[2],
        zeta_ua=params[3],
        zeta_ru=params[4],# sigma_e=params[4]
        sigma_e=params[5],
        r_land=params[6]
    )

    p, HHdecisions, meas, prices, agg = get_Steadystate(p, 1)
    output = output_gen(p, HHdecisions, meas, prices, agg, 1)

    # Extract model moments from the output
    model = [
        output.KK / output.LL * 30.0,
        output.rr_share,
        output.ua_share,
        output.ru_share,#output.share34#output.ru_share
        output.share34,
        output.land_income_share_state1
    ]


    # Compute the distance between model and data moments
    for ii in 1:NMOM
        dist[ii] = abs(model[ii] - data[ii]) + 100000 * abs(params_in[ii] - params[ii])
    end
    dist = dist ./ data
    max_dist = sum(dist .^ 2) / NMOM

    println("parameters")
    display(params)
    println("model moments")
    display(model)
    println("Distance")
    display(dist)
    println("Sum of Distance")
    display(max_dist)
    println("")

    # Prepare labels and changes matrix for DataFrame creation
    labels = [
        "K/Y",
        "rural rural share",
        "urban ag share",
        "rural urban share",#"top 25% in urban"
        "top 25% in urban",
        "land income share in rural"
    ]

    changes = hcat(params, model, data)
    d_changes = round.(changes, digits=4)

    # Create DataFrame directly
    df = DataFrame(d_changes, :auto)
    rename!(df, 1 => "p", 2 => "model moments", 3 => "data moments")
    insertcols!(df, 1, :moments => labels)

    # Write DataFrame to CSV
    CSV.write("figures/calbration.csv", df)

    return max_dist
end

function calibration_phi_a(params_in)

    println("------------------------------")

    NMOM = 1

    model = zeros(NMOM)
    data = zeros(NMOM)
    dist = zeros(NMOM)
    params = zeros(NMOM)


    params[1] = max(params_in[1], 0.0)

    println("parameters")
    display(params)
    println("")

    data[1] = 0.13570229260920416 * (1.0 - 0.3)

    p = setParameters(phi_a=params[1]) # check the parameters above

    # output = get_Steadystate(p, 1)

    p, HHdecisions, meas, prices, agg = get_Steadystate(p, 1)
    output = output_gen(p, HHdecisions, meas, prices, agg, 1)

    # model[1] = output.elas_I_y
    model[1] = output.ru_share


    println("MOMENTS")
    println("")

    for ii in 1:NMOM
        dist[ii] = abs(model[ii] - data[ii]) + 100000 * abs(params_in[ii] - params[ii])
    end

    max_dist = sum(dist .^ 2)

    # max_dist = -output.welfare

    println("parameters")
    display(params)
    println("model moments")
    display(model)
    println("Distance")
    display(dist)
    println("Sum of Distance")
    display(max_dist)
    println("")

    return max_dist
end

######################################################
# CALIBRATION
######################################################

# Initial guess for the parameters
initial_guess = [0.49777869220275955,
 -0.3479186418550472,
  0.09714970304225913,
  0.31666396394931695,
  0.21509367659714868,
  0.047434872244243725
]


# res = optimize(calibration, initial_guess)
# display(Optim.minimizer(res))
# error("stop")

# res = optimize(x -> calibration_phi_a(x), [0.5])
# display(Optim.minimizer(res))
# error("stop")


# ======================= #
#  MAIN                   #
# ======================= #

Ncase = 2

# param_help = setPar()
# @unpack NL, NY, NZ, NP, y = param_help

output = Vector{NamedTuple}(undef, Ncase)
income_thres_top10_base = 0.0

for i_case = 1:Ncase
    global income_thres_top10_base

    # set parameters
    if i_case == 1
        println("case: benchmark")
        p = setParameters()

    elseif i_case == 2
        println("case: phi_a = 0")
        p = setParameters(phi_a = 0.0)
    elseif i_case == 3
        println("case: phi_a = 0 and r_land increase by 50%")
        p = setParameters(phi_a = 0.0, r_land=0.04661546513664531*1.5)
    end
    p, dec, meas, prices, agg = get_Steadystate(p, i_case)
    output[i_case] = output_gen(p, dec, meas, prices, agg, i_case)

    if i_case == 1
        income_thres_top10_base = output[i_case].income_thres_top10
    end

end


# plot
plot(output[1].gridk0, output[1].kfun0[1, 1, :, 2], color=:blue, linestyle=:solid, linewidth=2, label=L"hs,l_{low}",
    title="Assets Policy function", xlabel=L"a", ylabel=L"a'=g(a,l)", xlims=(0.0, 1.0 ), ylims=(0.0, 1.0), legend=:topleft)
plot!(output[1].gridk0, output[1].kfun0[1, 4, :, 2], color=:red, linestyle=:solid, linewidth=2, label=L"hs,l_{mid}")
plot!(output[1].gridk0, output[1].kfun0[1, 7, :, 2], color=:black, linestyle=:solid, linewidth=2, label=L"hs,l_{high}")
plot!(output[1].gridk0, output[1].kfun0[1, 1, :, 4], color=:blue, linestyle=:dash, linewidth=2, label=L"cg,l_{mid}")
plot!(output[1].gridk0, output[1].kfun0[1, 4, :, 4], color=:red, linestyle=:dash, linewidth=2, label=L"cg,l_{mid}")
plot!(output[1].gridk0, output[1].kfun0[1, 7, :, 4], color=:black, linestyle=:dash, linewidth=2, label=L"cg,l_{high}")
savefig("figures/fig_kfun.pdf")

plot(output[1].gridk0, output[1].pplus[1, 4, :, 3, 1], color=:blue, linestyle=:solid, linewidth=2, label=L"rural, rural",
    title="Category Policy function", xlabel=L"a", ylabel=L"a'=g(a,l)", xlims=(0.0, 1.0 ), ylims=(0, 1), legend=:topleft)
plot!(output[1].gridk0, output[1].pplus[1, 4, :, 3, 2], color=:red, linestyle=:solid, linewidth=2, label=L"rural, urban")
plot!(output[1].gridk0, output[1].pplus[1, 4, :, 3, 3], color=:black, linestyle=:solid, linewidth=2, label=L"urban, ag")
plot!(output[1].gridk0, output[1].pplus[1, 4, :, 3, 4], color=:blue, linestyle=:dash, linewidth=2, label=L"urban, nonag")
# plot!(a, pplus[1, 4, :, 2], color=:red, linestyle=:dash, linewidth=2, label=L"cg,l_{mid}")
# plot!(a, pplus[1, 7, :, 2], color=:black, linestyle=:dash, linewidth=2, label=L"cg,l_{high}")
savefig("figures/fig_sfun.pdf")



