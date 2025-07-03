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

function setParameters(;
    mu=2.0,             # risk aversion (=3 baseline)             
    beta=0.40987992166424403,            # subjective discount factor 
    delta=0.08,            # depreciation
    alpha=0.36,            # capital'h share of income
    b=0.0,             # borrowing limit
    NH=7,             # number of discretized states
    rho=0.6,           # first-order autoregressive coefficient
    gamma_h=0.0,
    gamma_z=0.12,
    gamma_q=0.1,
    zeta_ua=0.11200676783749941,
    r_land=0.04661546513664531,
    phi_a=0.5,
    delta_n=0.27,
    sigma_e=0.17592346799252628,
    zeta_ru=0.3749448694982676,
    zeta_rr=-0.2990971845888921,
    income_thres_top10_base=0.0,
    sig=1.0           # intermediate value to calculate sigma (=0.4 BASE)
)

    phi_n = phi_a + delta_n
    # ================================================= #
    #  COMPUTE TRANSITION MATRIX OF LABOR PRODUCTIVITY  #
    # ================================================= #

    # ROUTINE tauchen.param TO COMPUTE TRANSITION MATRIX, GRID OF AN AR(1) AND
    # STATIONARY DISTRIBUTION
    # Approximate labor endowment shocks with seven-state Markov chain
    # log(s_{t}) = rho*log(s_{t-1}) + e_{t}
    # e_{t} ~ N(0, sig^2)

    M = 2.0
    NZ = 5

    lz, prob, invdist = tauchen(NZ, rho, sqrt(1.0 - rho^2), M)
    z = exp.(lz)

    lh = collect(range(-3.0, stop=3.0, length=NH))
    h = exp.(lh)

    # ================================================= #
    #  HUMAN CAPITAL INVESTMENT                         #
    # ================================================= #

    # ii = 1: rr (earner+kid in rural), ii = 2: ru (kid in rural, earner in urban)
    # ii = 3: ua (both urban agricultural hukou), ii = 4: un (both urban non-agricultural hukou)

    # Tuition by group
    NI = 4
    NA = 30                                     # Grid size for STATE
    Nk2 = 30                                    # Grid size for CONTROL


    lq = zeros(NI)
    lq[1] = log(0.72)
    lq[2] = log(0.72)
    lq[3] = log(0.88)
    lq[4] = log(1.0)

    mv_cost = zeros(NI, NI)
    # for ii in 1:NI
    #     mv_cost[ii, NI] = zeta_ua
    # end

    land_risk = zeros(NI)

    land_risk[3] = phi_a
    land_risk[4] = phi_n

    dutil = zeros(NI)
    dutil[3] = zeta_ua
    dutil[2] = zeta_ru
    dutil[1] = zeta_rr

    return (mu=mu, beta=beta, delta=delta, alpha=alpha, b=b, gamma_z=gamma_z, dutil=dutil, zeta_rr=zeta_rr,
        gamma_q=gamma_q, gamma_h=gamma_h, NH=NH, h=h, z=z, lh=lh, lz=lz, lq=lq, prob=prob, income_thres_top10_base=income_thres_top10_base,
        NA=NA, Nk2=Nk2, NZ=NZ, NI=NI, mv_cost=mv_cost, land_risk=land_risk, r_land=r_land, sigma_e=sigma_e)
end

function log_hplus(logz, logq, logh, logxi, param)
    @unpack gamma_z, gamma_q, gamma_h = param
    return gamma_z * logz + gamma_q * logq + gamma_h * logh + logxi
end

function set_prices(param, KL, land_lost, avg_income)

    # ================================================= #
    #  SETTING INTEREST, WAGE, and ASSET GRIDS          #
    # ================================================= #

    # r = param.alpha * ((KL)^(param.alpha - 1)) - param.delta # interest rate
    # wage = (1 - param.alpha) * ((param.alpha / (r + param.delta))^param.alpha)^(1 / (1 - param.alpha)) # wage

    r = 1.021^30 - 1.0
    wage = zeros(param.NI)
    wage[1] = 1.0
    wage[2:4] .= 1.35
    # wage = (1 - param.alpha) * ((param.alpha / (r + param.delta))^param.alpha)^(1 / (1 - param.alpha)) # wage

    # -phi is borrowing limit, b is adhoc
    # the second term is natural limit
    # if r <= 0.0
    phi = param.b
    # else
    #     phi = min(param.b, wage * param.h[1] / r)
    # end

    # capital grid (need define in each iteration since it depends on r/phi)
    a_u = 1.0                                    # Maximum value of capital grid
    a_l = -phi                                   # Borrowing constraint
    curvK = 1.1

    # Grid for state
    a = zeros(param.NA)
    a[1] = a_l
    for ia in 2:param.NA
        a[ia] = a[1] + (a_u - a_l) * ((ia - 1) / (param.NA - 1))^curvK
    end

    # Grid for optimal choice
    gridk2 = zeros(param.Nk2)
    gridk2[1] = a_l
    for ia in 2:param.Nk2
        gridk2[ia] = gridk2[1] + (a_u - a_l) * ((ia - 1) / (param.Nk2 - 1))^curvK
    end

    ell = 1.0 / (1.0 - land_lost)

    tuition = zeros(param.NI)
    tuition[1] = 13471.0 / 30333.0 / 30 * avg_income
    tuition[2] = 13471.0 / 30333.0 / 30 * avg_income
    tuition[3] = 24151.0 / 30333.0 / 30 * avg_income
    tuition[4] = 54728.0 / 30333.0 / 30 * avg_income

    return (r=r, wage=wage, phi=phi, a=a, gridk2=gridk2, ell=ell, avg_income=avg_income, tuition=tuition, a_u=a_u)

end


function solve_household(param, prices)
    @unpack NA, NH, NZ, Nk2, NI, mu, h, beta, prob, lz, lh, lq, mv_cost, r_land, land_risk, sigma_e, dutil = param
    @unpack r, wage, a, gridk2, ell, tuition = prices

    # Initialize some variables
    iaplus = zeros(NI, NH, NA, NZ)    # New index of policy function
    aplus = similar(iaplus)           # Policy function
    pplus = zeros(NI, NH, NA, NZ, NI) # Policy function for category choice

    v = zeros(NI, NH, NA, NZ)         # Old value function
    tv = similar(iaplus)              # New value function

    err = 20.0    # Error between old policy index and new policy index
    maxiter = 2000 # Maximum number of iterations
    iter = 1      # Iteration counter

    lhplus_cache = [log_hplus(lz[iz], lq[ii], lh[ih], 0.0, param) for ii in 1:NI, ih in 1:NH, iz in 1:NZ]
    interp_lh_cache = [(interp(lhplus_cache[ii, ih, iz], lh)) for ii in 1:NI, ih in 1:NH, iz in 1:NZ]

    while (err > 0.01) & (iter < maxiter)

        # Tabulate the utility function such that for zero or negative
        # consumption, utility remains a large negative number so that
        # such values will never be chosen as utility maximizing

        # Threads.@threads 
        for ii in 1:NI
            vtemp = fill(-1e6, Nk2, NI)
            ptemp = fill(-1e6, Nk2, NI)
            vtemp2 = fill(-1e6, Nk2)
            # for ii in 1:NI
            for ia in 1:NA # k(STATE)
                for ih in 1:NH # h(STATE)
                    for iz in 1:NZ
                        fill!(vtemp, -1e6)
                        fill!(ptemp, -1e6)
                        fill!(vtemp2, -1e6)
                        for iap in 1:Nk2 # k'(CONTROL)
                            for iip in 1:NI

                                # Amount of consumption given (k, l, k')
                                cons = wage[ii] * h[ih] + (1.0 + r) * a[ia] - gridk2[iap] - tuition[iip] - mv_cost[ii, iip] + r_land * ell * (1.0 - land_risk[ii])

                                if cons <= 0.0
                                    # Penalty for c < 0.0
                                    # Once c becomes negative, vtemp will not be updated (= large negative number)
                                    # kccmax = iap - 1
                                    break
                                end

                                util = (max(cons, 1e-4)^(1.0 - mu)) / (1.0 - mu) - dutil[ii]

                                # Interpolation of next period's value function
                                # Find node and weight for gridk2 (evaluating gridk2 in a)
                                ial, iar, varphi = interp(gridk2[iap], a)
                                ihl, ihr, varphi_h = interp_lh_cache[ii, ih, iz]

                                vpr = 0.0 # Next period's value function given (l, k')
                                for izp in 1:NZ # Expectation of next period's value function
                                    pz = prob[iz, izp]
                                    vpr += pz * (
                                        varphi_h * (varphi * v[iip, ihl, ial, izp] + (1.0 - varphi) * v[iip, ihl, iar, izp]) +
                                        (1.0 - varphi_h) * (varphi * v[iip, ihr, ial, izp] + (1.0 - varphi) * v[iip, ihr, iar, izp])
                                    )
                                end

                                vtemp[iap, iip] = util + beta * vpr
                            end


                            maxv = maximum(vtemp[iap, :])
                            sum_exp = sum(exp.((vtemp[iap, :] .- maxv) ./ sigma_e))
                            for iip in 1:NI
                                denom = sum(exp.((vtemp[iap, :] .- vtemp[iap, iip]) ./ sigma_e))
                                ptemp[iap, iip] = 1.0 / denom
                            end
                            vtemp2[iap] = maxv + sigma_e * log(sum_exp)



                        end

                        # Find k' that solves Bellman equation (subject to k' achieves c > 0)
                        max_val, max_index = findmax(vtemp2)
                        tv[ii, ih, ia, iz] = max_val
                        iaplus[ii, ih, ia, iz] = max_index
                        aplus[ii, ih, ia, iz] = gridk2[max_index]
                        pplus[ii, ih, ia, iz, :] = ptemp[max_index, :]
                    end
                end
            end
        end
        err = maximum(abs.(tv - v))
        v .= tv
        iter += 1
    end

    if iter == maxiter
        println("WARNING!! @aiyagari_vfi2.jl VFI: iteration reached max: iter=$iter,e rr=$err")
    end

    # Return household decisions as a struct
    return (
        aplus=aplus, iaplus=iaplus, pplus=pplus
    )
end


function get_distribution(param, dec, prices)
    @unpack NA, NH, NZ, Nk2, NI, mu, lh, lz, lq, land_risk, h = param
    @unpack aplus, iaplus, pplus = dec
    @unpack r, wage, a, gridk2, tuition = prices

    # Calculate stationary distribution
    m = ones(NI, NH, NA, NZ) / (NI * NH * NA * NZ) # Old distribution
    m_new = zeros(NI, NH, NA, NZ)                   # New distribution
    err = 1
    errTol = 0.00001
    maxiter = 2000
    iter = 1
    p_help = zeros(NI)

    while (err > errTol) & (iter < maxiter)
        for ii in 1:NI
            for ia in 1:NA # k
                for ih in 1:NH # l
                    for iz in 1:NZ # h

                        # iip = iplus[ii, ih, ia, iz] # Index of h'(k, l, h) next generation's education
                        lhplus = log_hplus(lz[iz], lq[ii], lh[ih], 0.0, param)
                        ihl, ihr, varphi_h = interp(lhplus, lh)

                        # Interpolation of policy function
                        # Split to two grids in a
                        ial, iar, varphi = interp(aplus[ii, ih, ia, iz], a)

                        p_help[:] = pplus[ii, ih, ia, iz, :]

                        for izp in 1:NZ # l'
                            for iip in 1:NI
                                m_new[iip, ihl, ial, izp] += p_help[iip] * param.prob[iz, izp] * varphi_h * varphi * m[ii, ih, ia, iz]
                                m_new[iip, ihl, iar, izp] += p_help[iip] * param.prob[iz, izp] * varphi_h * (1.0 - varphi) * m[ii, ih, ia, iz]
                                m_new[iip, ihr, ial, izp] += p_help[iip] * param.prob[iz, izp] * (1.0 - varphi_h) * varphi * m[ii, ih, ia, iz]
                                m_new[iip, ihr, iar, izp] += p_help[iip] * param.prob[iz, izp] * (1.0 - varphi_h) * (1.0 - varphi) * m[ii, ih, ia, iz]
                            end
                        end
                    end
                end
            end
        end

        err = maximum(abs.(m_new - m))
        m = copy(m_new)
        iter += 1
        m_new = zeros(NI, NH, NA, NZ)

    end



    if iter == maxiter
        println("WARNING!! @aiyagari_vfi2.jl INVARIANT DIST: iteration reached max: iter=$iter, err=$err")
    end

    return (
        m=m,
    )
end

function aggregation(param, dec, measures, prices)
    @unpack NA, NH, NZ, Nk2, NI, mu, lh, lz, lq, land_risk, h = param
    @unpack r, wage, a, gridk2, tuition = prices
    @unpack aplus, iaplus, pplus = dec


    m = measures.m

    meanL = 0.0
    land_lost = 0.0
    avg_income = 0.0
    for ii in 1:NI
        for ia in 1:NA # k
            for ih in 1:NH # l
                for iz in 1:NZ # h
                    land_lost += land_risk[ii] * m[ii, ih, ia, iz]
                    meanL += param.h[ih] * m[ii, ih, ia, iz]
                    avg_income += wage[ii] * h[ih] * m[ii, ih, ia, iz]
                end
            end
        end
    end
    meank = sum(sum(m .* aplus))
    sum_A = mapslices(x -> sum(x), m; dims=(1, 2, 4))
    # println(round.(sum_A; digits=2))
    # error("stop")

    mass_i = vec(sum(m, dims=(2, 3, 4)))
    mass_z = vec(sum(m, dims=(1, 2, 3)))
    mass_a = vec(sum(m, dims=(1, 2, 4)))
    mass_h = vec(sum(m, dims=(1, 3, 4)))

    return (
        meank=meank,
        meanL=meanL,
        land_lost=land_lost,
        avg_income=avg_income,
        mass_i=mass_i,
        mass_z=mass_z,
        mass_a=mass_a,
        mass_h=mass_h
    )
end

function get_Steadystate(param, icase)

    # ======================= #
    #  COMPUTE K and r in EQ  #
    # ======================= #

    K0 = 6.8 # Initial guess
    L0 = 1.0

    err2 = 1
    errTol = 0.01
    maxiter = 20
    iter = 1
    adj = 0.2
    # a = zeros(param.NA)

    KL0 = K0 / L0
    land_lost0 = 0.3
    avg_income0 = 2.8

    dec = nothing
    prices = nothing
    measures = nothing
    agg = nothing

    while (err2 > errTol) && (iter < maxiter)

        # Set prices given K/L
        prices = set_prices(param, KL0, land_lost0, avg_income0)

        # Solve household problems for decision rules
        dec = solve_household(param, prices)

        # Solve stationary distribution for aggregates K and L
        measures = get_distribution(param, dec, prices)

        agg = aggregation(param, dec, measures, prices)

        K1 = agg.meank
        L1 = agg.meanL
        land_lost1 = agg.land_lost
        avg_income1 = agg.avg_income

        # K1, L1, land_lost1, avg_income1 = get_distribution(param, dec, prices)

        KL1 = K1 / L1

        # err2 = abs(KL0 - KL1) / abs(KL1) + abs(land_lost1 - land_lost0)
        err2 = abs(avg_income0 - avg_income1) / abs(avg_income0) + abs(land_lost1 - land_lost0)


        # UPDATE GUESS: K0 + adj * (K1 - K0)

        println([
            iter,
            round(avg_income0, digits=4),
            round(avg_income1, digits=4),
            round(land_lost0, digits=4),
            round(land_lost1, digits=4),
            round(err2, digits=4)
        ])
        if err2 > errTol
            # KL0 += adj * (KL1 - KL0)
            avg_income0 += adj * (avg_income1 - avg_income0)
            land_lost0 += adj * (land_lost1 - land_lost0)
            iter += 1
        end

    end

    if iter == maxiter
        println("WARNING!! iter=$maxiter, err=$err2")
    end

    return param, dec, measures, prices, agg
end

function monte_carlo_simulation(param, dec, measures, prices, NN, icase_sim)
    @unpack h, lz, lq, lh, r_land = param
    @unpack wage, a, ell = prices

    ii_sim = zeros(Int, NN)
    iz_sim = zeros(Int, NN)
    ia_sim = zeros(Int, NN)
    ih_sim = zeros(Int, NN)

    # Initialize storage for household trajectories
    initial_states = sample_states_from_distribution(measures.m, NN)
    a_sim, h_sim, z_sim, wage_sim, earnings_sim, lhp_sim = zeros(NN), zeros(NN), zeros(NN), zeros(NN), zeros(NN), zeros(NN)

    Threads.@threads for i in 1:NN

        current_state = initial_states[i]
        ii_sim[i], ih_sim[i], ia_sim[i], iz_sim[i] = current_state[1], current_state[2], current_state[3], current_state[4]
        a_sim[i] = a[ia_sim[i]]
        wage_sim[i] = wage[ii_sim[i]]
        h_sim[i] = h[ih_sim[i]]
        earnings_sim[i] = wage_sim[i] * h_sim[i]
        lhp_sim[i] = log_hplus(lz[iz_sim[i]], lq[ii_sim[i]], lh[ih_sim[i]], 0.0, param)
        if icase_sim==1 && (ii_sim[i]==3 || ii_sim[i]==4)
            lhp_sim[i] = log_hplus(lz[iz_sim[i]], lq[1], lh[ih_sim[i]], 0.0, param)
        end
    end

    # Return all simulated data as a NamedTuple
    return (a_sim=a_sim, h_sim=h_sim, wage_sim=wage_sim, earnings_sim=earnings_sim, ii_sim=ii_sim, lhp_sim=lhp_sim)
end



function output_gen(param, dec, measures, prices, agg, icase)
    @unpack r_land, income_thres_top10_base = param
    @unpack ell = prices

    display(round.(agg.mass_i; digits=4))
    display(round.(agg.mass_h; digits=4))
    display(round.(agg.mass_a; digits=4))
    display(round.(agg.mass_z; digits=4))

    II = 50000

    sim = monte_carlo_simulation(param, dec, measures, prices, II, 0)
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
    idx_34 = findall(i -> sim.ii_sim[i] in (3, 4), sim.ii_sim)

    # Count the number in the top 25% among those with state 3 or 4
    count_top25_in_34 = count(i -> sim.earnings_sim[i] >= income_threshold, idx_34)
    share_top25_in_34 = count_top25_in_34 / length(idx_34)

    # Get indices where state is 1 or 2
    idx_12 = findall(i -> sim.ii_sim[i] in (3, 4), sim.ii_sim)

    

    # # Count the number in the top 25% among those with state 3 or 4
    count_top10_in_34 = count(i -> sim.lhp_sim[i] >= income_thres_top10, idx_34)
    share_top10_in_34 = count_top10_in_34 / length(idx_34)

    # error(sim.lhp_sim[idx_34])

    land_income_sim = r_land * ell ./ (sim.earnings_sim .+ r_land * ell)

    # Take the mean of land income share for state 1
    land_income_share_state1 = mean(land_income_sim[sim.ii_sim.==1])

    r_share = sum(measures.m[1:2, :, :, :]) / sum(measures.m[:, :, :, :])
    ua_share = sum(measures.m[3, :, :, :]) / sum(measures.m[:, :, :, :])
    ru_share = sum(measures.m[2, :, :, :]) / sum(measures.m[:, :, :, :])
    rr_share = sum(measures.m[1, :, :, :]) / sum(measures.m[:, :, :, :])


    college_thres = percentile(sim.lhp_sim, 90)
    # error(college_thresl)

    col_rate = zeros(param.NI)
    for ii in 1:param.NI
        inds = findall(sim.ii_sim .== ii)  # Get indices where ii_sim equals ii
        # Among those, calculate the share whose lhp_sim exceeds the college threshold
        count_above = count(i -> sim.lhp_sim[i] > college_thres, inds)
        # Proportion (relative to total in group)
        col_rate[ii] = count_above / length(inds)
    end

    # display(income_thres_top10)

    if icase==1
        sim = monte_carlo_simulation(param, dec, measures, prices, II, 1)
            # Count the number in the top 25% among those with state 3 or 4
    count_top10_in_34_cf = count(i -> sim.lhp_sim[i] >= income_thres_top10, idx_34)
    share_top10_in_34_cf = count_top10_in_34_cf / length(idx_34)
    # error([share_top10_in_34, share_top10_in_34_cf])
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
    param = setParameters(
        beta=params[1],
        zeta_rr=params[2],
        zeta_ua=params[3],
        zeta_ru=params[4],# sigma_e=params[4]
        sigma_e=params[5],
        r_land=params[6]
    )

    param, HHdecisions, measures, prices, agg = get_Steadystate(param, 1)
    output = output_gen(param, HHdecisions, measures, prices, agg, 1)

    # Extract model moments from the output
    model = [
        output.KK / output.LL * 30.0,
        output.rr_share,
        output.ua_share,
        output.ru_share,#output.share34#output.ru_share
        output.share34,
        output.land_income_share_state1
    ]

    println("MOMENTS")
    println("")

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
    rename!(df, 1 => "param", 2 => "model moments", 3 => "data moments")
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

    param = setParameters(phi_a=params[1]) # check the parameters above

    # output = get_Steadystate(param, 1)

    param, HHdecisions, measures, prices, agg = get_Steadystate(param, 1)
    output = output_gen(param, HHdecisions, measures, prices, agg, 1)

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
initial_guess = [0.40987992166424403,
    -0.2990971845888921,
    0.11200676783749941,
    0.3749448694982676,
    0.17592346799252628,
    0.04661546513664531
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

Ncase = 1

# param_help = setPar()
# @unpack NL, NY, NZ, NP, y = param_help

output = Vector{NamedTuple}(undef, Ncase)
income_thres_top10_base = 0.0

for i_case = 1:Ncase
    global income_thres_top10_base

    # set parameters
    if i_case == 1
        println("case: benchmark")
        param = setParameters()

    elseif i_case == 2
        println("case: validation exercise")
        param = setParameters(income_thres_top10_base=income_thres_top10_base)
    end
    param, dec, measures, prices, agg = get_Steadystate(param, i_case)
    output[i_case] = output_gen(param, dec, measures, prices, agg, i_case)

    if i_case == 1
        income_thres_top10_base = output[i_case].income_thres_top10
    end

end


# plot
plot(output[1].gridk0, output[1].kfun0[1, 1, :, 2], color=:blue, linestyle=:solid, linewidth=2, label=L"hs,l_{low}",
    title="Assets Policy function", xlabel=L"a", ylabel=L"a'=g(a,l)", xlims=(0.0, 1.0 ), ylims=(0.0, prices.a_u), legend=:topleft)
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



