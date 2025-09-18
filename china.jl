# ======================================================= #

# ======================================================= #
ENV["GKSwstype"] = "100"
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
using ThreadsX
using StatsPlots
using Plots.PlotMeasures

gr()
Plots.default(show=false)        # VSCode の自動表示を抑止

Random.seed!(1234)  # Set random seed

include("toolbox.jl")

function setParameters(;
    mu=2.0,                    # Risk aversion coefficient (baseline = 3)
    beta=0.5159784284016175,  # Subjective discount factor
    delta=0.08,                # Depreciation rate
    alpha=0.36,                # Capital's share of income
    b=0.0,                     # Borrowing limit
    NH=10,                      # Number of discretized labor productivity states
    rho=0.16,                   # AR(1) coefficient for labor productivity process
    gamma_h=0.0,               # Parameter for additional labor-related processes (unused here)
    gamma_z=0.09,              # Parameter for additional shock process (unused here)
    gamma_q=0.59,               # Parameter for another process (unused here)
    zeta_ua=-0.11330187603883306,  # Utility discount or cost parameter for urban agricultural group
    r_land=0.5249142987075832,    # Land return rate
    phi_a=0.374,                 # Parameter related to land risk for agricultural hukou
    delta_n=0.53,              # Additional land risk parameter
    sigma_e=0.19336480726308442,  # Scale parameter for idiosyncratic shocks
    zeta_ru=0.25515986986304473,    # Utility discount or cost parameter for rural urban group
    zeta_rr=-0.2825947270401527,   # Utility discount or cost parameter for rural rural group
    income_thres_top10_base=0.0,   # Base threshold for income top 10%
    w_un_r=1.35,
    w_ua_r=1.35,
    z_educ=1.0,
    zeta=0.0,
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
    lh = collect(range(-4.0, stop=-2.0, length=NH))
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
    NA = 20         # Grid size for asset state (k)
    NG = 30        # Grid size for asset control (k')

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

    # z_educ = 2.0

    return (
        mu=mu, beta=beta, delta=delta, alpha=alpha, b=b, gamma_z=gamma_z, gamma_q=gamma_q, gamma_h=gamma_h, NH=NH,
        h=h, z=z, lh=lh, lz=lz, lq=lq, prob=prob, income_thres_top10_base=income_thres_top10_base,
        NA=NA, NG=NG, NZ=NZ, NI=NI, mv_cost=mv_cost, land_risk=land_risk, r_land=r_land, sigma_e=sigma_e, dutil=dutil, zeta_rr=zeta_rr, z_educ=z_educ,
        w_un_r=w_un_r, w_ua_r=w_ua_r, zeta=zeta
    )
end

function log_hplus(logz, educ, logh, logxi, peer, p)
    return log(p.z_educ) + p.gamma_z * logz + p.gamma_q * log(max(educ, 1e-4)) + p.zeta * log(peer) + p.gamma_h * logh + logxi

end

function set_prices(p, KL, land_lost, avg_income, avg_z_r, avg_z_u)
    # Use a fixed interest rate
    r = 1.021^30 - 1.0

    # Set wage vector
    wage = zeros(p.NI)
    # wage[1] = 1.0
    # wage[2:4] .= 1.35
    wage[1] = 1.0
    wage[2:3] .= p.w_ua_r
    wage[4] = p.w_un_r

    # Borrowing constraint (phi)
    phi = p.b

    # Asset grid for state variables
    a_u = 0.3
    a_l = -phi
    curvK = 1.1

    a = zeros(p.NA)
    a[1] = a_l
    for ia in 2:p.NA
        a[ia] = a[1] + (a_u - a_l) * ((ia - 1) / (p.NA - 1))^curvK
    end


    # Land supply inferred from land lost
    ell = 1.0 / (1.0 - land_lost)

    # Tuition by type
    tuition = zeros(p.NI)
    # tuition[1] = 52016.0 / 30333.0 / 30 * avg_income
    # tuition[2] = 52016.0 / 30333.0 / 30 * avg_income
    # tuition[3] = 83223.0 / 30333.0 / 30 * avg_income
    # tuition[4] = 119645.0 / 30333.0 / 30 * avg_income

    tuition[1] = 52400.0 / 30333.0 / 30 * avg_income
    tuition[2] = 52400.0 / 30333.0 / 30 * avg_income
    tuition[3] = 83600.0 / 30333.0 / 30 * avg_income
    tuition[4] = 122500.0 / 30333.0 / 30 * avg_income

    # return Prices(r, wage, phi, a, ell, avg_income, tuition, a_u, KL, land_lost, avg_z_r, avg_z_u)
    return (
        r=r, wage=wage, phi=phi, a=a, ell=ell, avg_income=avg_income, tuition=tuition, a_u=a_u,
        KL=KL, land_lost=land_lost, avg_z_r=avg_z_r, avg_z_u=avg_z_u
    )
end


function solve_household(p, prices)
    # Initialize some variables
    iaplus = zeros(p.NI, p.NH, p.NA, p.NZ)           # Index of optimal k' choice
    aplus = similar(iaplus)                         # Optimal k' choice
    e = similar(iaplus)
    pplus = zeros(p.NI, p.NH, p.NA, p.NZ, p.NI)      # Probability distribution over category choices

    v = zeros(p.NI, p.NH, p.NA, p.NZ)                # Value function from previous iteration
    v_add = zeros(p.NI, p.NH, p.NA, p.NZ)            # The additive utility part of Value function from previous iteration
    tv = similar(iaplus)                            # Updated value function

    err = 20.0
    maxiter = 50
    iter = 1

    NG = 20
    # Grids
    # gridk2 = range(0.01, stop=0.5, length=NG)

    gridk2 = zeros(NG)
    gridk2[1] = 0.0
    for ia in 2:NG
        gridk2[ia] = gridk2[1] + (0.9 - 0.0) * ((ia - 1) / (NG - 1))^1.1
    end
    gride = range(0.0, stop=0.3, length=NG)

    # display(gridk2)

    while (err > 0.05) & (iter < maxiter)
        # Use ThreadsX.foreach for parallelization over asset/human capital indices

        #         Threads.@threads for idx in CartesianIndices((p.NA, p.NH))
        # # For each state (human capital, ability, asset)
        # ia, ih = Tuple(idx)

        ThreadsX.foreach(1:(p.NA*p.NH)) do idx

            ia = div(idx - 1, p.NH) + 1  # asset index
            ih = mod(idx - 1, p.NH) + 1  # human capital index
            @inbounds for ii in 1:p.NI, iz in 1:p.NZ
                local vtemp, v_addtemp, ptemp, vtemp2, v_addtemp2
                vtemp = fill(-1e6, NG, NG, p.NI)
                v_addtemp = fill(-1e6, NG, NG, p.NI)
                ptemp = fill(0.0, NG, NG, p.NI)
                vtemp2 = fill(-1e6, NG, NG)
                v_addtemp2 = fill(-1e6, NG, NG)

                coh = prices.wage[ii] * p.h[ih] - prices.tuition[ii] +
                      (1.0 + prices.r) * prices.a[ia] + p.r_land * prices.ell * (1.0 - p.land_risk[ii])
                if ii in (1, 2)
                    ie_max = 1
                else
                    ie_max = 1
                end
                @inbounds @views for iap in 1:NG, ie in 1:ie_max
                    @inbounds for iip in 1:p.NI
                        a_plus = gridk2[iap] #*coh
                        educ = gride[ie] #*(coh-a_plus)
                        cons = coh - a_plus - educ
                        if cons <= 0.0
                            continue  # Skip infeasible choice
                        end
                        # All code below depends on cons > 0
                        util = (max(cons, 1e-2)^(1.0 - p.mu)) / (1.0 - p.mu) - p.dutil[ii]
                        if ii == 1 || ii == 2
                            ihl, ihr, varphi_h = interp(log_hplus(p.lz[iz], educ + prices.tuition[ii], p.lh[ih], 0.0, prices.avg_z_r, p), p.lh)
                        else
                            ihl, ihr, varphi_h = interp(log_hplus(p.lz[iz], educ + prices.tuition[ii], p.lh[ih], 0.0, prices.avg_z_u, p), p.lh)
                        end
                        # Interpolation for next period's value
                        ial, iar, varphi = interp(a_plus, prices.a)
                        # varphi = max(min(varphi, 1.0), 0.0)
                        # varphi_h = max(min(varphi_h, 1.0), 0.0)
                        vpr = 0.0
                        vpr_add = 0.0
                        @inbounds for izp in 1:p.NZ
                            pz = p.prob[iz, izp]
                            vpr += pz * (
                                varphi_h * (varphi * v[iip, ihl, ial, izp] + (1.0 - varphi) * v[iip, ihl, iar, izp]) +
                                (1.0 - varphi_h) * (varphi * v[iip, ihr, ial, izp] + (1.0 - varphi) * v[iip, ihr, iar, izp])
                            )
                            vpr_add += pz * (
                                varphi_h * (varphi * v_add[iip, ihl, ial, izp] + (1.0 - varphi) * v_add[iip, ihl, iar, izp]) +
                                (1.0 - varphi_h) * (varphi * v_add[iip, ihr, ial, izp] + (1.0 - varphi) * v_add[iip, ihr, iar, izp])
                            )
                        end
                        vtemp[iap, ie, iip] = util + p.beta * vpr
                        v_addtemp[iap, ie, iip] = (max(cons, 1e-4)^(1.0 - p.mu)) / (1.0 - p.mu) + p.beta * vpr_add
                    end
                    # Softmax and log-sum-exp for choice probabilities and value
                    vmax = maximum(vtemp[iap, ie, :])
                    exps = exp.((vtemp[iap, ie, :] .- vmax) ./ p.sigma_e)
                    Z = sum(exps)
                    @inbounds for iip in 1:p.NI
                        ptemp[iap, ie, iip] = exps[iip] / Z
                    end
                    vtemp2[iap, ie] = vmax + p.sigma_e * log(Z)
                    v_addtemp2[iap, ie] = sum(ptemp[iap, ie, :] .* v_addtemp[iap, ie, :])
                end
                iap_opt, ie_opt = Tuple(argmax(vtemp2))
                tv[ii, ih, ia, iz] = vtemp2[iap_opt, ie_opt]
                iaplus[ii, ih, ia, iz] = iap_opt
                aplus[ii, ih, ia, iz] = gridk2[iap_opt] #*coh
                e[ii, ih, ia, iz] = gride[ie_opt] #*(coh-aplus[ii, ih, ia, iz])
                pplus[ii, ih, ia, iz, :] = ptemp[iap_opt, ie_opt, :]
                v_add[ii, ih, ia, iz] = v_addtemp2[iap_opt, ie_opt]
            end
        end


        err = maximum(abs.(tv .- v))
        v .= tv
        iter += 1
    end

    # display(e)
    # error("check")
    if iter == maxiter
        println("WARNING!! @solve_household: iteration reached max: iter=$iter, err=$err")
    end

    # return Dec(aplus, e, pplus, v, v_add)
    return (aplus=aplus, e=e, pplus=pplus, v=v, v_add=v_add)
end

function get_distribution(p, dec, prices)
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
                        if ii == 1 || ii == 2
                            lhplus = log_hplus(p.lz[iz], dec.e[ii, ih, ia, iz] + prices.tuition[ii], p.lh[ih], 0.0, prices.avg_z_r, p)
                        else
                            lhplus = log_hplus(p.lz[iz], dec.e[ii, ih, ia, iz] + prices.tuition[ii], p.lh[ih], 0.0, prices.avg_z_u, p)
                        end

                        ihl, ihr, varphi_h = interp(lhplus, p.lh)
                        ial, iar, varphi = interp(dec.aplus[ii, ih, ia, iz], prices.a)

                        varphi = max(min(varphi, 1.0), 0.0)
                        varphi_h = max(min(varphi_h, 1.0), 0.0)

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

        # display("check")
        # display(minimum(m_new))

        err = maximum(abs.(m_new .- m))
        m .= m_new
        iter += 1
        fill!(m_new, 0.0)
    end
    # error(vec(sum(m, dims=(1, 3, 4))))

    # display(minimum(m))

    if iter == maxiter
        println("WARNING!! get_distribution: iteration reached max: iter = $iter, err = $err")
    end

    # return Meas(m)
    return (m=m,)
end

function aggregation(p, dec, meas, prices)
    m = meas.m

    meanL = 0.0
    land_lost = 0.0
    avg_income = 0.0
    avg_z_r = 0.0
    avg_z_u = 0.0
    mea_r = 0.0
    mea_u = 0.0
    # EE = 0.0

    for ii in 1:p.NI
        for ia in 1:p.NA
            for ih in 1:p.NH
                for iz in 1:p.NZ
                    land_lost += p.land_risk[ii] * m[ii, ih, ia, iz]
                    meanL += p.h[ih] * m[ii, ih, ia, iz]
                    avg_income += prices.wage[ii] * p.h[ih] * m[ii, ih, ia, iz]
                    # EE += dec.e[ii, ih, ia, iz] * m[ii, ih, ia, iz]
                    if iz == 1 || iz == 2
                        avg_z_r += p.z[iz] * m[ii, ih, ia, iz]
                        mea_r += m[ii, ih, ia, iz]
                    else
                        avg_z_u += p.z[iz] * m[ii, ih, ia, iz]
                        mea_u += m[ii, ih, ia, iz]
                    end
                end
            end
        end
    end

    avg_z_r = avg_z_r / max(mea_r, 1e-4)
    avg_z_u = avg_z_u / max(mea_u, 1e-4)

    meank = sum(dec.aplus .* m)

    # Marginal distributions
    mass_i = vec(sum(m, dims=(2, 3, 4)))
    mass_z = vec(sum(m, dims=(1, 2, 3)))
    mass_a = vec(sum(m, dims=(1, 2, 4)))
    mass_h = vec(sum(m, dims=(1, 3, 4)))

    # error(mass_h)

    return (
        meank=meank,
        meanL=meanL,
        land_lost=land_lost,
        avg_income=avg_income,
        mass_i=mass_i,
        mass_z=mass_z,
        mass_a=mass_a,
        mass_h=mass_h,
        avg_z_r=avg_z_r,
        avg_z_u=avg_z_u
    )
end

function get_Steadystate(p, icase::Int; guess_base=nothing)
    # Initial values
    KL = 6.8     # Capital-labor ratio guess
    land_lost = 0.2443
    avg_income = 0.0642
    avg_z_r = 0.3175
    avg_z_u = 2.2185

    # if icase == 3
    #     if guess_base === nothing
    #         error("guess_base must be provided for icase == 3")
    #     end
    #     KL = guess_base.KL
    #     land_lost = guess_base.land_lost
    #     avg_income = guess_base.avg_income
    # end

    err = 1.0
    errTol = 1e-2
    iter = 0
    maxiter = 50
    adj = 0.2

    # # Preallocate
    # prices = Prices(0.0, zeros(p.NI), 0.0, zeros(p.NA), 0.0, 0.0, zeros(p.NI), 0.0, 0.0, 0.0, 0.0, 0.0)
    # dec = Dec(zeros(p.NI, p.NH, p.NA, p.NZ), zeros(p.NI, p.NH, p.NA, p.NZ), zeros(p.NI, p.NH, p.NA, p.NZ, p.NI), zeros(p.NI, p.NH, p.NA, p.NZ), zeros(p.NI, p.NH, p.NA, p.NZ))
    # meas = Meas(zeros(p.NI, p.NH, p.NA, p.NZ))
    # agg = Agg(0.0, 0.0, 0.0, 0.0, zeros(p.NI), zeros(p.NZ), zeros(p.NA), zeros(p.NH), 0.0, 0.0)

    prices = nothing
    dec = nothing
    meas = nothing
    agg = nothing

    while err > errTol && iter < maxiter
        iter += 1

        if icase == 1
            prices = set_prices(p, KL, land_lost, avg_income, avg_z_r, avg_z_u)
        else
            prices = set_prices(p, KL, land_lost, guess_base.avg_income, avg_z_r, avg_z_u)
        end
        dec = solve_household(p, prices)
        meas = get_distribution(p, dec, prices)
        agg = aggregation(p, dec, meas, prices)



        # Update values
        KL_new = agg.meank / agg.meanL
        land_lost_new = agg.land_lost
        avg_income_new = agg.avg_income
        avg_z_r_new = agg.avg_z_r
        avg_z_u_new = agg.avg_z_u

        # Error metric
        err = abs(avg_income - avg_income_new) / avg_income + abs(land_lost_new - land_lost)
        err += abs(avg_z_r - avg_z_r_new) / avg_z_r + abs(avg_z_u_new - avg_z_u) / avg_z_u

        # Print diagnostics
        println((iter, round(avg_income_new, digits=4), round(land_lost_new, digits=4),
            round(avg_z_r_new, digits=4), round(avg_z_u_new, digits=4),
            round(err, digits=4)))

        # Partial eq
        # if icase == 3
        #     break
        # end

        # Update guesses with damping
        avg_income += adj * (avg_income_new - avg_income)
        land_lost += adj * (land_lost_new - land_lost)
        KL += adj * (KL_new - KL)
        avg_z_r += adj * (avg_z_r_new - avg_z_r)
        avg_z_u += adj * (avg_z_u_new - avg_z_u)


    end

    if iter == maxiter
        println("WARNING: did not converge. err = $err")
    end

    return p, dec, meas, prices, agg
end

function monte_carlo_simulation(rng::AbstractRNG, p, dec, meas, prices, NN, icase_sim)
    @unpack h, lz, lq, lh, z, r_land = p
    @unpack a, ell = prices

    ii_sim = zeros(Int, NN)
    iip_sim = zeros(Int, NN)
    iz_sim = zeros(Int, NN)
    ia_sim = zeros(Int, NN)
    ih_sim = zeros(Int, NN)
    ipeer_sim = zeros(NN)

    # Initialize storage for household trajectories
    initial_states = sample_states_from_distribution(rng, meas.m, NN)
    a_sim, h_sim, e_sim, lq_sim, z_sim, wage_sim, wagep_sim, earnings_sim, earningsp_sim, hp_sim, lhp_sim = zeros(NN), zeros(NN), zeros(NN), zeros(NN), zeros(NN), zeros(NN), zeros(NN), zeros(NN), zeros(NN), zeros(NN), zeros(NN)

    e_pub_sim = zeros(NN)
    for i in 1:NN

        current_state = initial_states[i]
        ii_sim[i], ih_sim[i], ia_sim[i], iz_sim[i] = current_state[1], current_state[2], current_state[3], current_state[4]
        a_sim[i] = a[ia_sim[i]]
        z_sim[i] = z[iz_sim[i]]
        wage_sim[i] = prices.wage[ii_sim[i]]
        h_sim[i] = h[ih_sim[i]]
        e_sim[i] = dec.e[ii_sim[i], ih_sim[i], ia_sim[i], iz_sim[i]]
        earnings_sim[i] = wage_sim[i] * h_sim[i]
        lq_sim[i] = log(p.z_educ * e_sim[i])

        # lhp_sim[i] = log_hplus(lz[iz_sim[i]], log(p.z_educ * e_sim[i]), lh[ih_sim[i]], 0.0, p)
        # if icase_sim == 1 && (ii_sim[i] == 3 || ii_sim[i] == 4)
        #     lhp_sim[i] = log_hplus(lz[iz_sim[i]], log(p.z_educ * e_sim[i]), lh[ih_sim[i]], 0.0, p)
        # end


        if icase_sim == 1 && (ii_sim[i] == 3 || ii_sim[i] == 4) # validation exercise
            ipeer_sim[i] = prices.avg_z_r
            lhp_sim[i] = log_hplus(lz[iz_sim[i]], prices.tuition[1], lh[ih_sim[i]], 0.0, prices.avg_z_r, p)
        elseif icase_sim == 2 && (ii_sim[i] == 1 || ii_sim[i] == 2) # validation exercise
            ipeer_sim[i] = prices.avg_z_u
            ua_share_help = 0.248
            un_share_help = 1.0 - 0.445 - 0.248 - 0.139
            exp_help = prices.tuition[3] * ua_share_help / (ua_share_help + un_share_help) + prices.tuition[4] * un_share_help / (ua_share_help + un_share_help)
            lhp_sim[i] = log_hplus(lz[iz_sim[i]], exp_help, lh[ih_sim[i]], 0.0, prices.avg_z_u, p)
        else

            if ii_sim[i] == 1 || ii_sim[i] == 2
                ipeer_sim[i] = prices.avg_z_r
                lhp_sim[i] = log_hplus(lz[iz_sim[i]], e_sim[i] + prices.tuition[ii_sim[i]], lh[ih_sim[i]], 0.0, prices.avg_z_r, p)
            else
                ipeer_sim[i] = prices.avg_z_u
                lhp_sim[i] = log_hplus(lz[iz_sim[i]], e_sim[i] + prices.tuition[ii_sim[i]], lh[ih_sim[i]], 0.0, prices.avg_z_u, p)
            end
        end

        hp_sim[i] = exp(lhp_sim[i])
        e_pub_sim[i] = prices.tuition[ii_sim[i]]

        iip_sim[i] = sample_with_weights(rng, 1:p.NI, dec.pplus[ii_sim[i], ih_sim[i], ia_sim[i], iz_sim[i], :])
        wagep_sim[i] = prices.wage[iip_sim[i]]
        earningsp_sim[i] = prices.wage[iip_sim[i]] * hp_sim[i]

    end

    # Return all simulated data as a NamedTuple
    return (a_sim=a_sim, h_sim=h_sim, wage_sim=wage_sim, wagep_sim=wagep_sim, e_pub_sim=e_pub_sim,
        earnings_sim=earnings_sim, earningsp_sim=earningsp_sim, ii_sim=ii_sim, ipeer_sim=ipeer_sim,
        iip_sim=iip_sim, lhp_sim=lhp_sim, hp_sim=hp_sim, lq_sim=lq_sim, e_sim=e_sim, z_sim=z_sim)
end



function output_gen(p, dec, meas, prices, agg, icase)
    @unpack r_land, income_thres_top10_base = p
    @unpack ell = prices

    display(round.(agg.mass_i; digits=4))
    display(round.(agg.mass_h; digits=4))
    display(round.(agg.mass_a; digits=4))
    display(round.(agg.mass_z; digits=4))

    II = 50000

    rng = MersenneTwister(1234)
    # sim = monte_carlo_simulation(rng, p::Params, dec, meas, prices, II, 0)
    sim = monte_carlo_simulation(rng, p, dec, meas, prices, II, 0)
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
    idx_12 = findall(x -> x in (1, 2), sim.ii_sim)
    idx_34 = findall(x -> x in (3, 4), sim.ii_sim)

    # Count the number in the top 25% among those with state 3 or 4
    count_top25_in_34 = count(i -> sim.earnings_sim[i] >= income_threshold, idx_34)
    share_top25_in_34 = count_top25_in_34 / length(idx_34)

    # # Count the number in the top 25% among those with state 3 or 4
    count_top10_in_12 = count(i -> sim.lhp_sim[i] >= income_thres_top10, idx_12)
    share_top10_in_12 = count_top10_in_12 / length(idx_12)
    # # Count the number in the top 25% among those with state 3 or 4
    count_top10_in_34 = count(i -> sim.lhp_sim[i] >= income_thres_top10, idx_34)
    share_top10_in_34 = count_top10_in_34 / length(idx_34)


    land_income_sim = r_land * ell ./ (sim.earnings_sim .+ r_land * ell)

    # Take the mean of land income share for state 1
    land_income_share_state1 = mean(land_income_sim[sim.ii_sim.==1])

    r_share = sum(meas.m[1:2, :, :, :]) / sum(meas.m[:, :, :, :])
    un_share = sum(meas.m[4, :, :, :]) / sum(meas.m[:, :, :, :])
    ua_share = sum(meas.m[3, :, :, :]) / sum(meas.m[:, :, :, :])
    ru_share = sum(meas.m[2, :, :, :]) / sum(meas.m[:, :, :, :])
    rr_share = sum(meas.m[1, :, :, :]) / sum(meas.m[:, :, :, :])

    welfare = sum(meas.m .* dec.v)
    welfare_add = sum(meas.m .* dec.v_add)


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



    avg_earnings_rural = mean(sim.earnings_sim[(sim.ii_sim.==1)])
    avg_earnings_urban = mean(sim.earnings_sim[(sim.ii_sim.==2).|(sim.ii_sim.==3).|(sim.ii_sim.==4)])


    avg_lq_r = mean(sim.lq_sim[(sim.ii_sim.==2).|(sim.ii_sim.==1)])
    avg_lq_ua = mean(sim.lq_sim[(sim.ii_sim.==3)])
    avg_lq_un = mean(sim.lq_sim[(sim.ii_sim.==4)])

    avg_e_r = mean(sim.e_sim[(sim.ii_sim.==2).|(sim.ii_sim.==1)]) / mean(sim.earnings_sim)
    avg_e_ua = mean(sim.e_sim[(sim.ii_sim.==3)]) / mean(sim.earnings_sim)
    avg_e_un = mean(sim.e_sim[(sim.ii_sim.==4)]) / mean(sim.earnings_sim)
    avg_e_u = mean(sim.e_sim[(sim.ii_sim.==3).|(sim.ii_sim.==4)]) / mean(sim.earnings_sim)

    avg_earnings_r = mean(sim.earnings_sim[(sim.ii_sim.==1)])
    avg_earnings_ua = mean(sim.earnings_sim[(sim.ii_sim.==2).|(sim.ii_sim.==3)])
    avg_earnings_un = mean(sim.earnings_sim[(sim.ii_sim.==4)])

    gini_earnings = gini_coefficient(sim.earnings_sim)

    reg_cons = ones(length(sim.earnings_sim))

    reg_X = hcat(reg_cons, log.(sim.earnings_sim))
    reg_Y = log.(sim.earningsp_sim)
    reg_beta = (reg_X' * reg_X) \ (reg_X' * reg_Y)
    IGE = reg_beta[2]

    # display([log(sim.earnings_sim[1]), log(sim.earningsp_sim[1]), IGE])
    # error("stop")

    top10_sim = Int.(sim.lhp_sim .≥ quantile(sim.lhp_sim, 0.9))
    reg_X = hcat(reg_cons, log.(sim.e_sim + prices.tuition[sim.ii_sim]), log.(sim.z_sim), log.(sim.ipeer_sim))
    reg_Y = top10_sim
    reg_beta = (reg_X' * reg_X) \ (reg_X' * reg_Y)
    coef_peer = reg_beta[4]

    # error(coef_peer)

    # reg_X = hcat(reg_cons, log.(sim.h_sim))
    # reg_Y = log.(sim.hp_sim)
    # reg_beta = (reg_X' * reg_X) \ (reg_X' * reg_Y)
    # IGE2 = reg_beta[2]

    # reg_X = hcat(reg_cons, log.(sim.wage_sim))
    # reg_Y = log.(sim.wagep_sim)
    # reg_beta = (reg_X' * reg_X) \ (reg_X' * reg_Y)
    # IGE3 = reg_beta[2]


    reg_X = hcat(reg_cons, log.(sim.e_sim + prices.tuition[sim.ii_sim]), log.(sim.z_sim))
    reg_Y = log.(sim.earningsp_sim)
    reg_beta = (reg_X' * reg_X) \ (reg_X' * reg_Y)
    coef_e = reg_beta[2]
    coef_z = reg_beta[3]

    corr_lepub_lz = cor(log.(sim.e_pub_sim), log.(sim.z_sim))

    println("MOMENTS")
    println("")
    println("shares of rr, ru, ua, un")
    display(round(rr_share; digits=4))
    display(round(ru_share; digits=4))
    display(round(ua_share; digits=4))
    display(round(un_share; digits=4))
    println("")
    println("rural share of college")
    display(round(share_top10_in_12; digits=4))
    println("urban share of college")
    display(round(share_top10_in_34; digits=4))
    println("")
    println("rural/urban avg income")
    display(round(avg_earnings_rural / avg_earnings_urban; digits=4))
    # println("")
    # println("IGE of ability")
    # display(round(IGE2; digits=4))
    # println("")
    # println("IGE of location")
    # display(round(IGE3; digits=4))
    println("e, r, ua, un")
    display(round(avg_e_r; digits=4))
    display(round(avg_e_ua; digits=4))
    display(round(avg_e_un; digits=4))

    # println("quality of edc, r, ua, un")
    # display(round(avg_lq_r; digits=4))
    # display(round(avg_lq_ua; digits=4))
    # display(round(avg_lq_un; digits=4))

    println("avg earnings, r, ua, un")
    display(round(avg_earnings_r; digits=4))
    display(round(avg_earnings_ua; digits=4))
    display(round(avg_earnings_un; digits=4))

    #     println("avg tuition/earnings, r, ua, un")
    # display(round(prices.tuition[1]/avg_earnings_r; digits=4))
    # display(round(prices.tuition[3]/avg_earnings_ua; digits=4))
    # display(round(prices.tuition[4]/avg_earnings_un; digits=4))

    println("avg tuition/earnings, r, ua, un")
    display(round(prices.tuition[1] / mean(sim.earnings_sim); digits=4))
    display(round(prices.tuition[3] / mean(sim.earnings_sim); digits=4))
    display(round(prices.tuition[4] / mean(sim.earnings_sim); digits=4))



    #     lq[1] = log(0.72)
    # lq[2] = log(0.72)
    # lq[3] = log(0.88)
    # lq[4] = log(1.0)

    # display(income_thres_top10)
    # share_top10_in_12_cf = 0.0
    # share_top10_in_34_cf = 0.0
    # if icase == 1
        sim = monte_carlo_simulation(rng, p, dec, meas, prices, II, 2)
        # Count the number in the top 25% among those with state 3 or 4
        count_top10_in_12_cf = count(i -> sim.lhp_sim[i] >= income_thres_top10, idx_12)
        share_top10_in_12_cf = count_top10_in_12_cf / length(idx_12)
        # error([share_top10_in_34, share_top10_in_34_cf])
        println("")
        println("validation: rural share of college with rural q")
        display(round(share_top10_in_12_cf; digits=4))

        sim = monte_carlo_simulation(rng, p, dec, meas, prices, II, 1)
        # Count the number in the top 25% among those with state 3 or 4
        count_top10_in_34_cf = count(i -> sim.lhp_sim[i] >= income_thres_top10, idx_34)
        share_top10_in_34_cf = count_top10_in_34_cf / length(idx_34)
        # error([share_top10_in_34, share_top10_in_34_cf])
        println("")
        println("validation: urban share of college with rural q")
        display(round(share_top10_in_34_cf; digits=4))


    # end



    return (kfun0=dec.aplus, pplus=dec.pplus, gridk0=gridk0, KK=agg.meank, share_top25_in_34=share_top25_in_34, land_income_share_state1=land_income_share_state1,
        LL=agg.meanL, r_share=r_share, ua_share=ua_share, ru_share=ru_share, share34=share34, rr_share=rr_share, income_thres_top10=income_thres_top10,
        share_top10_in_34=share_top10_in_34, share_top10_in_34_cf=share_top10_in_34_cf, m=meas.m, v=dec.v, v_add=dec.v_add,
        share_top10_in_12=share_top10_in_12, share_top10_in_12_cf=share_top10_in_12_cf, validation=share_top10_in_12_cf-share_top10_in_12, 
        welfare=welfare, welfare_add=welfare_add, avg_income=agg.avg_income, corr_lepub_lz=corr_lepub_lz,
        gini_earnings=gini_earnings, meanL=agg.meanL, IGE=IGE, avg_earnings_r=avg_earnings_r, avg_e_u=avg_e_u,
        avg_earnings_ua=avg_earnings_ua, avg_earnings_un=avg_earnings_un, avg_lq_r=avg_lq_r, avg_lq_ua=avg_lq_ua, avg_lq_un=avg_lq_un,
        avg_e_r=avg_e_r, avg_e_ua=avg_e_ua, avg_e_un=avg_e_un, coef_peer=coef_peer, coef_e=coef_e, coef_z=coef_z, ell_eq=ell)

end

function calibration(params_in)

    println("------------------------------")

    NMOM = 8

    model = zeros(NMOM)
    data = zeros(NMOM)
    dist = zeros(NMOM)
    params = zeros(NMOM)

    # Initialize model and data vectors directly
    params = [
        min(max(params_in[1], 0.0), 0.7),
        params_in[2],
        params_in[3],
        params_in[4],
        max(params_in[5], 1e-4),
        max(params_in[6], 0.0),
        max(params_in[7], 0.0),
        max(params_in[8], 0.0)
    ]

    println("parameters")
    display(params)
    println("")

    data_w_r = (15.7 * 0.012 + 7.6 * 0.402 + 17.5 * 0.011 + 11.9 * 0.026) / (0.012 + 0.402 + 0.011 + 0.026)
    data_w_ua = (23.0 * 0.014 + 11.7 * 0.128) / (0.014 + 0.128)
    data_w_un = (21.9 * 0.169 + 13.5 * 0.239) / (0.169 + 0.239)


    data = [
        3.53,  # K/Y
        0.472, # 0.445+0.139
        0.224,
        0.126, # 0.144/(1.0-0.584)# 0.139
        0.734,
        0.43,
        1.64,
        2.17,
    ]

    # display(data_w_ua / data_w_r)
    # display(data_w_un / data_w_r)
    # error("check")

    p = setParameters(
        beta=params[1],
        zeta_rr=params[2],
        zeta_ua=params[3],
        zeta_ru=params[4],# sigma_e=params[4]
        sigma_e=params[5],
        r_land=params[6],
        w_ua_r=params[7],
        w_un_r=params[8]
    )

    p, HHdecisions, meas, prices, agg = get_Steadystate(p, 1)
    output = output_gen(p, HHdecisions, meas, prices, agg, 1)

    # Extract model moments from the output
    model = [
        output.KK / output.avg_income * 30.0,
        output.rr_share,
        output.ua_share,
        output.ru_share,#output.share34#output.ru_share
        output.share34,
        output.land_income_share_state1,
        output.avg_earnings_ua / output.avg_earnings_r,
        output.avg_earnings_un / output.avg_earnings_r,
    ]


    # Compute the distance between model and data moments
    for ii in 1:NMOM
        dist[ii] = abs(model[ii] - data[ii])
    end
    dist = dist ./ data
    max_dist = sqrt(sum(dist .^ 2)) / NMOM
    for ii in 1:8
        max_dist += 100000 * abs(params_in[ii] - params[ii])
    end


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
        "land income share in rural",
        "wage ua/r",
        "wage un/r",
    ]

    changes = hcat(model, data)
    d_changes = round.(changes, digits=4)

    # Create DataFrame directly
    df = DataFrame(d_changes, :auto)
    rename!(df, 1 => "model moments", 2 => "data moments")
    insertcols!(df, 1, :moments => labels)

    # Write DataFrame to CSV
    CSV.write("figures/calbration.csv", df)

    in_param = params

    # Prepare labels and changes matrix for DataFrame creation
    labels = [
        "beta",
        "zeta_rr",
        "zeta_ua",
        "zeta_ru",#"top 25% in urban"
        "sigma_e",
        "r_land",
        "w_ua",
        "w_un",
    ]


    # ラベル数に合わせて値を取り、丸める
    n = length(labels)
    vals = round.(in_param[1:n]; digits=2)

    # 素直に2列のDataFrameを作る
    df = DataFrame(moments=labels, values=vals)

    # CSVへ書き出し
    CSV.write("figures/in_param.csv", df)

    ex_param = [
        p.mu,
        0.16, #p.rho,
        p.gamma_z,
        p.gamma_q,
        0.374,
        0.53,
        52400.0,
        83600.0,
        122500.0
    ]

    labels = [
        "mu",
        "rho (persistence of z)",
        "gamma_z",
        "gamma_e",#"top 25% in urban"
        "phi_a",
        "phi_a - phi_n",
        "tuition r (yuan)",
        "tuition ua (yuan)",
        "tuition un (yuan)"
    ]

    # ラベル数に合わせて値を取り、丸める
    n = length(labels)
    vals = round.(ex_param[1:n]; digits=2)

    # 素直に2列のDataFrameを作る
    df = DataFrame(moments=labels, values=vals)
    CSV.write("figures/ex_param.csv", df)

    return max_dist
end

function calibration_phi_a(params_in, param_vec)

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

    data[1] = 0.139 * 2.0 / 3.0

    p = setParameters(
        beta=param_vec[1],
        zeta_rr=param_vec[2],
        zeta_ua=param_vec[3],
        zeta_ru=param_vec[4], # sigma_e=params[4]
        sigma_e=param_vec[5],
        r_land=param_vec[6],
        w_ua_r=param_vec[7],
        w_un_r=param_vec[8],
        phi_a=params[1]) # check the parameters above
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
# initial_guess = [
#     0.47779143674495533
#     -4.221095334210588
#     -0.05474569770102006
#     2.9858810337785684
#     1.831184218637923
#     0.028976820622901113
#     1.6085058822725786
#     1.8990798135220495
# ]

initial_guess = [
  0.4812302192713535
 -4.832374962387025
 -0.055335557291067976
  3.139710947266862
  1.830218788934081
  0.026864627128996912
  1.5838614464182614
  1.9322557193322103
]


# res = optimize(calibration, initial_guess, NelderMead())
# display(Optim.minimizer(res))
# error("stop")

# res = optimize(x -> calibration_phi_a(x, initial_guess), [0.5])
# display(Optim.minimizer(res))
# error("stop")


# ======================= #
#  MAIN                   #
# ======================= #


function main(param_vec)  # ← 引数名を変更（元: base_params）
    Ncase = 3
    output = Vector{NamedTuple}(undef, Ncase)
    income_thres_top10_base = 0.0
    guess_base = nothing
    a = nothing

    for i_case in 1:Ncase
        # set parameters
        p = if i_case == 1
            println("case: benchmark")
            # Set parameters and get steady state results
            setParameters(
                beta=param_vec[1],
                zeta_rr=param_vec[2],
                zeta_ua=param_vec[3],
                zeta_ru=param_vec[4], # sigma_e=params[4]
                sigma_e=param_vec[5],
                r_land=param_vec[6],
                w_ua_r=param_vec[7],
                w_un_r=param_vec[8],
            )
        elseif i_case == 2
            println("case: phi_a = 0.0")
            setParameters(
                beta=param_vec[1],
                zeta_rr=param_vec[2],
                zeta_ua=param_vec[3],
                zeta_ru=param_vec[4], # sigma_e=params[4]
                sigma_e=param_vec[5],
                r_land=param_vec[6],
                w_ua_r=param_vec[7],
                w_un_r=param_vec[8],
                phi_a=0.0
            )
            # setParameters(phi_a=0.0)
            # elseif i_case == 3
            #     println("case: phi_a = 0 PE")
            #     setParameters(phi_a=0.0)
        elseif i_case == 3
            println("case: phi's = 0.0")
            setParameters(
                beta=param_vec[1],
                zeta_rr=param_vec[2],
                zeta_ua=param_vec[3],
                zeta_ru=param_vec[4], # sigma_e=params[4]
                sigma_e=param_vec[5],
                r_land=param_vec[6],
                w_ua_r=param_vec[7],
                w_un_r=param_vec[8],
                phi_a=0.0,
                delta_n=0.0 #0.09931640625000003
            )
            # setParameters(phi_a=0.0)
            # elseif i_case == 3
            #     println("case: phi_a = 0 PE")
        end

        if i_case == 1
            p, dec, meas, prices, agg = get_Steadystate(p, i_case)
        else
            p, dec, meas, prices, agg = get_Steadystate(p, i_case; guess_base=guess_base)
        end
        output[i_case] = output_gen(p, dec, meas, prices, agg, i_case)

        if i_case == 1
            income_thres_top10_base = output[i_case].income_thres_top10
            # guess_base = Guess_base(prices.KL, prices.land_lost, prices.avg_income)
            guess_base = (KL=prices.KL, land_lost=prices.land_lost, avg_income=prices.avg_income)
            a = prices.a
        end
    end



    return output, income_thres_top10_base, a
end

# Run main simulation to get outputs
output, income_thres_top10_base, a = main(initial_guess)

base_params = initial_guess

p = setParameters(
    beta=base_params[1],
    zeta_rr=base_params[2],
    zeta_ua=base_params[3],
    zeta_ru=base_params[4],# sigma_e=params[4]
    sigma_e=base_params[5],
    r_land=base_params[6],
    w_ua_r=base_params[7],
    w_un_r=base_params[8],
)

# Number of cases
Ncase = 3

# Initialize welfare change container
changes = zeros(Ncase, 11)

for icase = 1:Ncase

    num_ru = zeros(2)
    den_ru = zeros(2)
    num_ru_az = zeros(2, p.NZ, p.NA)
    den_ru_az = zeros(2, p.NZ, p.NA)
    num_az = zeros(p.NZ, p.NA)
    den_az = zeros(p.NZ, p.NA)
    for ii in 1:p.NI
        for ia in 1:p.NA
            for ih in 1:p.NH
                for iz in 1:p.NZ
                    wel_change = ((sum(output[icase].v[ii, ih, ia, iz]) - (sum(output[1].v[ii, ih, ia, iz]) - sum(output[1].v_add[ii, ih, ia, iz]))) / sum(output[1].v_add[ii, ih, ia, iz]))^(1.0 / (1.0 - p.mu)) - 1.0
                    if ii == 1 || ii == 2
                        num_ru[1] += output[1].m[ii, ih, ia, iz] * wel_change
                        den_ru[1] += output[1].m[ii, ih, ia, iz]
                        num_ru_az[1, iz, ia] += output[1].m[ii, ih, ia, iz] * wel_change
                        den_ru_az[1, iz, ia] += output[1].m[ii, ih, ia, iz]
                    else
                        num_ru[2] += output[1].m[ii, ih, ia, iz] * wel_change
                        den_ru[2] += output[1].m[ii, ih, ia, iz]
                        num_ru_az[2, iz, ia] += output[1].m[ii, ih, ia, iz] * wel_change
                        den_ru_az[2, iz, ia] += output[1].m[ii, ih, ia, iz]
                    end
                    num_az[iz, ia] += output[1].m[ii, ih, ia, iz] * wel_change
                    den_az[iz, ia] += output[1].m[ii, ih, ia, iz]
                end
            end
        end
    end
    wel_ru = num_ru ./ max.(den_ru, 0.0)
    wel_ru_az = num_ru_az ./ max.(den_ru_az, 0.0)
    wel_az = num_az ./ max.(den_az, 0.0)



    if icase == 1 || icase == 2 || icase == 3
        if icase == 1
            changes[icase, 1] = ((output[icase].welfare - (output[1].welfare - output[1].welfare_add)) / output[1].welfare_add)^(1.0 / (1.0 - p.mu)) - 1.0
            changes[icase, 2] = 0.0
            changes[icase, 3] = output[icase].gini_earnings
            changes[icase, 4] = output[icase].IGE
            changes[icase, 5] = 0.0
            changes[icase, 6] = output[icase].corr_lepub_lz
            changes[icase, 7] = output[icase].ru_share
            changes[icase, 8] = 0.0
            changes[icase, 9] = 0.0
            changes[icase, 10] = 0.0
            changes[icase, 11] = output[icase].validation
        else
            changes[icase, 1] = ((output[icase].welfare - (output[1].welfare - output[1].welfare_add)) / output[1].welfare_add)^(1.0 / (1.0 - p.mu)) - 1.0
            changes[icase, 2] = output[icase].avg_income / output[1].avg_income - 1.0
            changes[icase, 3] = output[icase].gini_earnings / output[1].gini_earnings - 1.0
            changes[icase, 4] = output[icase].IGE / output[1].IGE - 1.0
            changes[icase, 5] = output[icase].meanL / output[1].meanL - 1.0
            changes[icase, 6] = output[icase].corr_lepub_lz - output[1].corr_lepub_lz
            changes[icase, 7] = output[icase].ru_share - output[1].ru_share
            changes[icase, 8] = wel_ru[1]
            changes[icase, 9] = wel_ru[2]
            changes[icase, 10] = output[icase].ell_eq / output[1].ell_eq - 1.0
            changes[icase, 11] = output[icase].validation
        end
        changes[icase, :] = changes[icase, :] * 100.0
    end


    # plt1 = surface(a, p.lz, wel_ru_az[1, :, :];
    # xlabel = L"\ell_z", ylabel = L"a", zlabel = "weighted avg welfare gain",
    # title = "ii = 1 or 2", camera = (45, 30), colorbar = true)

    if icase > 1
        x = p.lz
        y = a
        Z12 = wel_ru_az[1, :, :]                 # (NZ, NA)
        Z34 = wel_ru_az[2, :, :]
        Z = wel_az
        # Plots の heatmap(x, y, Z) は Z のサイズが (length(y), length(x)) なので転置する
        Z12p = permutedims(Z12, (2, 1))          # (NA, NZ)
        Z34p = permutedims(Z34, (2, 1))
        Zp = permutedims(Z, (2, 1))

        plt1 = heatmap(x, y, Z12p; xlabel="log ability", ylabel="assets", right_margin = 10mm)
        savefig(plt1, "figures/welfare_heatmap_rr_ru_$(icase).png")

        plt2 = heatmap(x, y, Z34p; xlabel="log ability", ylabel="assets", right_margin = 10mm)
        savefig(plt2, "figures/welfare_heatmap_ua_un_$(icase).png")

plt3 = heatmap(x, y, Zp;
    xlabel = "log ability",
    ylabel = "assets",
    right_margin = 10mm
)
savefig(plt3, "figures/welfare_heatmap_$(icase).png")
    end

end



# Labels and column names
row_labels = ["Welfare (CEV %)", "GDP (%)", "gini earnings (%)", "IGE (%)", 
"Human capital (%)", "corr log educ, log z (ppt)", "ru share (ppt)", 
"Welfare r (CEV %)", "Welfare u (CEV %)", "land per capita", "validation (%, change in rural share of college with urban q, level for each case)"]
col_labels = ["Benchmark", "phi_a=0, phi_n>0", "phi_a=phi_n=0"]

# Transpose and round the matrix for DataFrame creation
rounded_changes = round.(changes', digits=4)

# Create DataFrame with labeled rows and columns
df = DataFrame(rounded_changes, col_labels)
insertcols!(df, 1, :variables => row_labels)

# Export to CSV
CSV.write("figures/outcome.csv", df)



# plot
plot(output[1].gridk0, output[1].kfun0[1, 1, :, 2], color=:blue, linestyle=:solid, linewidth=2, label=L"hs,l_{low}",
    title="Assets Policy function", xlabel=L"a", ylabel=L"a'=g(a,l)", xlims=(0.0, 1.0), ylims=(0.0, 1.0), legend=:topleft)
plot!(output[1].gridk0, output[1].kfun0[1, 4, :, 2], color=:red, linestyle=:solid, linewidth=2, label=L"hs,l_{mid}")
plot!(output[1].gridk0, output[1].kfun0[1, 7, :, 2], color=:black, linestyle=:solid, linewidth=2, label=L"hs,l_{high}")
plot!(output[1].gridk0, output[1].kfun0[1, 1, :, 4], color=:blue, linestyle=:dash, linewidth=2, label=L"cg,l_{mid}")
plot!(output[1].gridk0, output[1].kfun0[1, 4, :, 4], color=:red, linestyle=:dash, linewidth=2, label=L"cg,l_{mid}")
plot!(output[1].gridk0, output[1].kfun0[1, 7, :, 4], color=:black, linestyle=:dash, linewidth=2, label=L"cg,l_{high}")
savefig("figures/fig_kfun.pdf")

plot(output[1].gridk0, output[1].pplus[1, 4, :, 3, 1], color=:blue, linestyle=:solid, linewidth=2, label=L"rural, rural",
    title="Category Policy function", xlabel=L"a", ylabel=L"a'=g(a,l)", xlims=(0.0, 1.0), ylims=(0, 1), legend=:topleft)
plot!(output[1].gridk0, output[1].pplus[1, 4, :, 3, 2], color=:red, linestyle=:solid, linewidth=2, label=L"rural, urban")
plot!(output[1].gridk0, output[1].pplus[1, 4, :, 3, 3], color=:black, linestyle=:solid, linewidth=2, label=L"urban, ag")
plot!(output[1].gridk0, output[1].pplus[1, 4, :, 3, 4], color=:blue, linestyle=:dash, linewidth=2, label=L"urban, nonag")
# plot!(a, pplus[1, 4, :, 2], color=:red, linestyle=:dash, linewidth=2, label=L"cg,l_{mid}")
# plot!(a, pplus[1, 7, :, 2], color=:black, linestyle=:dash, linewidth=2, label=L"cg,l_{high}")
savefig("figures/fig_sfun.pdf")


cases = 1:3
rr = [output[i].rr_share for i in cases]
ru = [output[i].ru_share for i in cases]
ua = [output[i].ua_share for i in cases]
un = clamp.(1 .- (rr .+ ru .+ ua), 0.0, 1.0)

# データ行列（行=ケース, 列=カテゴリ）
data = hcat(rr, ru, ua, un)

groupedbar(col_labels, data;
    bar_position=:stack,
    label=["rr" "ru" "ua" "un"],
    xlabel="",
    ylabel="Share",
    ylims=(0, 1),
    legend=:topright,
    title="Shares by Location"
)

isdir("figures") || mkpath("figures")
savefig("figures/share_barplot.pdf")

