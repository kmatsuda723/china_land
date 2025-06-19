# ======================================================= #
# Model of Aiyagari (1994)                                #
# By Sagiri Kitao (Translated in Julia by Taiki Ono)      #
# ======================================================= #

# import libraries
using Plots
using Optim
using Random
using Distributions
using LaTeXStrings
using Parameters # enable @unpack
Random.seed!(1234)  # シードを固定

# test

function tauchen(N, rho, sigma, param)
    """
    ---------------------------------------------------
    === AR(1)過程をtauchenの手法によって離散化する関数 ===
    ---------------------------------------------------
    ※z'= ρ*z + ε, ε~N(0,σ_{ε}^2) を離散化する

    <input>
    ・N: 離散化するグリッドの数
    ・rho: AR(1)過程の慣性(上式のρ)
    ・sigma: AR(1)過程のショック項の標準偏差(上式のσ_{ε})
    ・m: 離散化するグリッドの範囲に関するパラメータ
    <output>
    ・Z: 離散化されたグリッド
    ・Zprob: 各グリッドの遷移行列
    ・Zinv: Zの定常分布
    """
    Zprob = zeros(N, N) # 遷移確率の行列
    Zinv = zeros(N, 1)  # 定常分布

    # 等間隔のグリッドを定める
    # 最大値と最小値
    zmax = param * sqrt(sigma^2 / (1 - rho^2))
    zmin = -zmax
    # グリッド間の間隔
    w = (zmax - zmin) / (N - 1)

    Z = collect(range(zmin, zmax, length=N))

    # グリッド所与として遷移確率を求める
    for j in 1:N # 今期のZのインデックス
        for jp in 1:N  # 来期のZのインデックス
            if jp == 1
                Zprob[j, jp] = cdf_normal((Z[jp] - rho * Z[j] + w / 2) / sigma)
            elseif jp == N
                Zprob[j, jp] = 1 - cdf_normal((Z[jp] - rho * Z[j] - w / 2) / sigma)
            else
                Zprob[j, jp] = cdf_normal((Z[jp] - rho * Z[j] + w / 2) / sigma) - cdf_normal((Z[jp] - rho * Z[j] - w / 2) / sigma)
            end
        end
    end

    # 定常分布を求める
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
    === 標準正規分布の累積分布関数 ===
    --------------------------------
    <input>
    ・x: 
    <output>
    ・c: 標準正規分布にしたがう確率変数Xがx以下である確率
    """
    d = Normal(0, 1) # 標準正規分布
    c = cdf(d, x)

    return c

end

function interp(x, grid)
    # Find indices of the closest grids and the weights for linear interpolation
    ial = searchsortedlast(grid, x)  # Index of the grid just above or equal to x
    ial = max(1, ial)  # Ensure index iz within bounds

    if ial > length(grid) - 1
        ial = length(grid) - 1  # Handle case where x iz beyond the grid
    end

    iar = ial + 1  # The index just below ial

    # Compute the weights for interpolation
    varphi = (grid[iar] - x) / (grid[iar] - grid[ial])
    return ial, iar, varphi
end


function setParameters(;
    mu=2.0,             # risk aversion (=3 baseline)             
    beta=0.96^30,            # subjective discount factor 
    delta=0.08,            # depreciation
    alpha=0.36,            # capital'h share of income
    b=0.0,             # borrowing limit
    NH=7,             # number of discretized states
    rho=0.6,           # first-order autoregressive coefficient
    gamma_h=0.08,
    gamma_z=0.57,
    gamma_q=0.1,
    zeta=0.0,
    r_land=1.04^30-1.0,
    phi_a=0.5,
    phi_n=0.77,
    sigma_e=0.1,
    sig=1.0           # intermediate value to calculate sigma (=0.4 BASE)
)

    # ================================================= #
    #  COMPUTE TRANSITION MATRIX OF LABOR PRODUCTIVITY  #
    # ================================================= #

    # ROUTINE tauchen.param TO COMPUTE TRANSITION MATRIX, GRID OF AN AR(1) AND
    # STATIONARY DISTRIBUTION
    # approximate labor endowment shocks with seven states Markov chain
    # log(s_{t}) = rho*log(s_{t-1})+e_{t} 
    # e_{t}~ N(0,sig^2)

    M = 2.0
    NZ = 5

    lz, prob, invdist = tauchen(NZ, rho, sqrt(1.0 - rho^2), M)
    z = exp.(lz)

    lh = collect(range(-3.0, stop=3.0, length=NH))
    h = exp.(lh)

    # ================================================= #
    #  HUMAN CAPITAL INVESTMENT                         #
    # ================================================= #

    # ii = 1: rr (earner+kid in rural), ii = 2: rn (kid in rural, earner in urban)
    # ii = 3: ua (both urban agricultural hukou), ii = 4: un (both urban non-agricultural hukou)

    # tuition
    NI = 4
    NA = 30                                     # grid size for STATE 
    Nk2 = 30                                     # grid size for CONTROL


    lq = zeros(NI)
    lq[1] = 0.0
    lq[2] = 0.0
    lq[3] = 0.1
    lq[4] = 0.2

    mv_cost = zeros(NI, NI)
    for ii in 1:NI
        mv_cost[ii, NI] = zeta
    end

    land_risk = zeros(NI)

    land_risk[3] = phi_a
    land_risk[4] = phi_n

    return (mu=mu, beta=beta, delta=delta, alpha=alpha, b=b, gamma_z=gamma_z,
        gamma_q=gamma_q, gamma_h=gamma_h, NH=NH, h=h, z=z, lh=lh, lz=lz, lq=lq, prob=prob,
        NA=NA, Nk2=Nk2, NZ=NZ, NI=NI, mv_cost=mv_cost, land_risk=land_risk, r_land=r_land, sigma_e=sigma_e)
end

function log_hplus(logz, logq, logh, logxi, param)
    @unpack gamma_z, gamma_q, gamma_h = param
    return gamma_z*logz + gamma_q*logq + gamma_h*logh + logxi
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

    # -phi iz borrowing limit, b iz adhoc
    # the second term iz natural limit
    # if r <= 0.0
    phi = param.b
    # else
    #     phi = min(param.b, wage * param.h[1] / r)
    # end

    # capital grid (need define in each iteration since it depends on r/phi)
    a_u = 1.0                                    # maximum value of capital grid
    a_l = -phi                                  # borrowing constraint
    curvK = 1.1

    # grid for state
    a = zeros(param.NA)
    a[1] = a_l
    for ia in 2:param.NA
        a[ia] = a[1] + (a_u - a_l) * ((ia - 1) / (param.NA - 1))^curvK
    end

    # grid for optimal choice
    gridk2 = zeros(param.Nk2)
    gridk2[1] = a_l
    for ia in 2:param.Nk2
        gridk2[ia] = gridk2[1] + (a_u - a_l) * ((ia - 1) / (param.Nk2 - 1))^curvK
    end

    ell = 1.0 / (1.0 - land_lost)

    tuition = zeros(param.NI)
    tuition[1] = 2222.0 / 30333.0 * avg_income
    tuition[2] = 2222.0 / 30333.0 * avg_income
    tuition[3] = 3666.0 / 30333.0 * avg_income
    tuition[4] = 5250.0 / 30333.0 * avg_income

    return (r=r, wage=wage, phi=phi, a=a, gridk2=gridk2, ell=ell, avg_income=avg_income, tuition=tuition, a_u=a_u)

end


function solve_household(param, prices)
    @unpack NA, NH, NZ, Nk2, NI, mu, h, beta, prob, lz, lh, lq, mv_cost, r_land, land_risk, sigma_e = param
    @unpack r, wage, a, gridk2, ell, tuition = prices

    # initialize some variables
    iaplus = zeros(NI, NH, NA, NZ)    # new index of policy function 
    aplus = similar(iaplus)     # policy function   
    pplus = zeros(NI, NH, NA, NZ, NI) # old policy function 

    v = zeros(NI, NH, NA, NZ)        # old value function
    tv = similar(iaplus)       # new value function

    err = 20.0   # error between old policy index and new policy index
    maxiter = 2000 # maximum number of iteration 
    iter = 1    # iteration counter

    while (err > 0.01) & (iter < maxiter)

        # tabulate the utility function such that for zero or negative
        # consumption utility remains a large negative number so that
        # such values will never be chosen as utility maximizing

        for ii in 1:NI
            for ia in 1:NA # k(STATE)
                for ih in 1:NH # h(STATE)
                    for iz in 1:NZ
                        vtemp = -1000000 .* ones(Nk2, NI) # initizalization
                        ptemp = -1000000 .* ones(Nk2, NI) # initizalization
                        vtemp2 = -1000000 .* ones(Nk2) # initizalization

                        for iap in 1:Nk2 # k'(CONTROL)
                            for iip in 1:NI

                                # amount of consumption given (k,l,k')
                                cons = wage[ii] * h[ih] + (1.0 + r) * a[ia] - gridk2[iap] - tuition[iip] - mv_cost[ii, iip] + r_land * ell * (1.0 - land_risk[ii])

                                if cons <= 0.0
                                    # penalty for c<0.0
                                    # once c becomes negative, vtemp will not be updated(=large negative number)
                                    # kccmax = iap - 1
                                    break
                                end

                                util = (cons^(1.0 - mu)) / (1.0 - mu)

                                # interpolation of next period'h value function
                                # find node and weight for gridk2 (evaluating gridk2 in a) 
                                ial, iar, varphi = interp(gridk2[iap], a)

                                lhplus = log_hplus(lz[iz], lq[ii], lh[ih], 0.0, param)
                                ihl, ihr, varphi_h = interp(lhplus, lh)


                                vpr = 0.0 # next period'h value function given (l,k')
                                for izp in 1:NZ # expectation of next period'h value function
                                    vpr += prob[iz, izp] * varphi_h * (varphi * v[iip, ihl, ial, izp] + (1.0 - varphi) * v[iip, ihl, iar, izp])
                                    vpr += prob[iz, izp] * (1.0 - varphi_h) * (varphi * v[iip, ihr, ial, izp] + (1.0 - varphi) * v[iip, ihr, iar, izp])
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

                        # find k' that  solves bellman equation
                        max_val, max_index = findmax(vtemp2) # subject to k' achieves c>0
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

        # println([iter,err])
        #flush(stdout)

        iter += 1

    end

    if iter == maxiter
        println("WARNING!! @aiyagari_vfi2.jl VFI: iteration reached max: iter=$iter,e rr=$err")
    end

    # error("stop")

    # Return household decisions as a struct
    return (
        aplus=aplus, iaplus=iaplus, pplus=pplus
    )
end


function get_distribution(param, dec, prices)
    @unpack NA, NH, NZ, Nk2, NI, mu, lh, lz, lq, land_risk, h = param
    @unpack aplus, iaplus, pplus = dec
    @unpack r, wage, a, gridk2, tuition = prices

    # calculate stationary distribution
    m = ones(NI, NH, NA, NZ) / (NI * NH * NA * NZ) # old distribution
    mea1 = zeros(NI, NH, NA, NZ) # new distribution
    err = 1
    errTol = 0.00001
    maxiter = 2000
    iter = 1
    meanL = 0.0
    land_lost = 0.0
    p_help = zeros(NI)
    avg_income = 0.0


    while (err > errTol) & (iter < maxiter)
        for ii in 1:NI
            for ia in 1:NA # k
                for ih in 1:NH # l
                    for iz in 1:NZ # h

                        # iip = iplus[ii, ih, ia, iz] # index of h'(k,l,h) next gen'h education
                        lhplus = log_hplus(lz[iz], lq[ii], lh[ih], 0.0, param)
                        ihl, ihr, varphi_h = interp(lhplus, lh)

                        # interpolation of policy function 
                        # split to two grids in a
                        ial, iar, varphi = interp(aplus[ii, ih, ia, iz], a)

                        p_help[:] = pplus[ii, ih, ia, iz, :]

                        for izp in 1:NZ # l'
                            for iip in 1:NI
                                mea1[iip, ihl, ial, izp] += p_help[iip] * param.prob[iz, izp] * varphi_h * varphi * m[ii, ih, ia, iz]
                                mea1[iip, ihl, iar, izp] += p_help[iip] * param.prob[iz, izp] * varphi_h * (1.0 - varphi) * m[ii, ih, ia, iz]
                                mea1[iip, ihr, ial, izp] += p_help[iip] * param.prob[iz, izp] * (1.0 - varphi_h) * varphi * m[ii, ih, ia, iz]
                                mea1[iip, ihr, iar, izp] += p_help[iip] * param.prob[iz, izp] * (1.0 - varphi_h) * (1.0 - varphi) * m[ii, ih, ia, iz]
                            end
                        end
                    end
                end
            end
        end

        err = maximum(abs.(mea1 - m))
        m = copy(mea1)
        iter += 1
        mea1 = zeros(NI, NH, NA, NZ)

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
    sum_A = mapslices(x -> sum(x), m; dims=(1,2,4))
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
    )end

function get_Steadystate(param, icase)

    # ======================= #
    #  COMPUTE K and r in EQ  #
    # ======================= #

    K0 = 6.8 # initial guess
    L0 = 1.0

    err2 = 1
    errTol = 0.01
    maxiter = 20
    iter = 1
    adj = 0.2
    ell = 1.0
    # a = zeros(param.NA)

    KL0 = K0 / L0
    land_lost0 = 0.3
    avg_income0 = 2.8

    dec = nothing
    prices = nothing
    measures = nothing
    agg = nothing

    while (err2 > errTol) && (iter < maxiter)

        # set prices given K/L
        prices = set_prices(param, KL0, land_lost0, avg_income0)

        # solve household problems for decision rules
        dec = solve_household(param, prices)

        # solve stationary distribution for aggregates K and L
        measures = get_distribution(param, dec, prices)

        agg = aggregation(param, dec, measures, prices)

        K1 = agg.meank
        L1 = agg.meanL
        land_lost1 = agg.land_lost
        avg_income1 = agg.avg_income

        # K1, L1, land_lost1, avg_income1 = get_distribution(param, dec, prices)

        KL1 = K1 / L1

        # err2 =abs(KL0 -  KL1) / abs(KL1) + abs(land_lost1 - land_lost0)
        err2 = abs(avg_income0 - avg_income1) / abs(avg_income0) + abs(land_lost1 - land_lost0)


        # UPDATE GUESS AS K0+adj*(K1-K0)

        println([iter, avg_income0, avg_income1, land_lost0, land_lost1, err2])

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

    # prices = set_prices(param, KL0, land_lost0, avg_income0)
    # kfun0, iaplus, pplus = solve_household(param, prices)
    # gridk0 = prices.a

    return param, dec, measures, prices, agg

end

function sample_states_from_distribution(distribution_matrix, NN)
    NI, NH, NA, NZ = size(distribution_matrix)  # Get the dimensions of the distribution matrix

    # Flatten the distribution matrix into a 1D array of probabilities
    distribution_flat = vec(distribution_matrix)

    # Create a list of possible states, corresponding to the positions in the distribution matrix
    possible_states = [(ii, ih, ia, iz) for ii in 1:NI, ih in 1:NH, ia in 1:NA, iz in 1:NZ]

    # Sample initial states for each household based on the probabilities
    sampled_indices = rand(Categorical(distribution_flat), NN)

    # Map sampled indices back to household states (ih, iz, ia)
    initial_states = [possible_states[idx] for idx in sampled_indices]

    return initial_states
end

function monte_carlo_simulation(param, dec, measures, prices, NN)
    @unpack h = param
    @unpack wage, a = prices

    ii_sim = zeros(Int, NN)
    iz_sim = zeros(Int, NN)
    ia_sim = zeros(Int, NN)
    ih_sim = zeros(Int, NN)

    # Initialize storage for household trajectories
    initial_states = sample_states_from_distribution(measures.m, NN)
    a_sim, h_sim, z_sim, wage_sim, earnings_sim = zeros(NN), zeros(NN), zeros(NN), zeros(NN), zeros(NN)

    Threads.@threads for i in 1:NN

        current_state = initial_states[i]
        ii_sim[i], ih_sim[i], ia_sim[i], iz_sim[i] = current_state[1], current_state[2], current_state[3], current_state[4]
        a_sim[i] = a[ia_sim[i]]
        wage_sim[i] = wage[ii_sim[i]]
        h_sim[i] = h[ih_sim[i]]
        earnings_sim[i] = wage_sim[i]*h_sim[i]
    end

    # Return all simulated data as a NamedTuple
    return (a_sim=a_sim, h_sim=h_sim, wage_sim=wage_sim, earnings_sim=earnings_sim, ii_sim=ii_sim)
end



function output_gen(param, dec, measures, prices, agg, icase)

    display(agg.mass_i)
    display(agg.mass_h)
    display(agg.mass_a)
    display(agg.mass_z)

    II = 50000

    sim = monte_carlo_simulation(param, dec, measures, prices, II)
    gridk0 = prices.a

    return (kfun0=dec.aplus, pplus=dec.pplus, gridk0=gridk0)

end

# ======================= #
#  MAIN                   #
# ======================= #

Ncase = 1

# param_help = setPar()
# @unpack NL, NY, NZ, NP, y = param_help

output = Vector{NamedTuple}(undef, Ncase)

# set parameters
param = setParameters()
param, dec, measures, prices, agg = get_Steadystate(param, 1)
output[1] = output_gen(param, dec, measures, prices, agg, 1)


# plot
plot(output[1].gridk0, output[1].kfun0[1, 1, :, 1], color=:blue, linestyle=:solid, linewidth=2, label=L"hs,l_{low}",
    title="Policy function", xlabel=L"a", ylabel=L"a'=g(a,l)", xlims=(0.0, prices.a_u), ylims=(0.0, prices.a_u), legend=:topleft)
plot!(output[1].gridk0, output[1].kfun0[1, 4, :, 1], color=:red, linestyle=:solid, linewidth=2, label=L"hs,l_{mid}")
plot!(output[1].gridk0, output[1].kfun0[1, 7, :, 1], color=:black, linestyle=:solid, linewidth=2, label=L"hs,l_{high}")
plot!(output[1].gridk0, output[1].kfun0[1, 1, :, 2], color=:blue, linestyle=:dash, linewidth=2, label=L"cg,l_{mid}")
plot!(output[1].gridk0, output[1].kfun0[1, 4, :, 2], color=:red, linestyle=:dash, linewidth=2, label=L"cg,l_{mid}")
plot!(output[1].gridk0, output[1].kfun0[1, 7, :, 2], color=:black, linestyle=:dash, linewidth=2, label=L"cg,l_{high}")
savefig("figures/fig_kfun.pdf")

plot(output[1].gridk0, output[1].pplus[1, 4, :, 1, 1], color=:blue, linestyle=:solid, linewidth=2, label=L"rural rural",
    title="Policy function", xlabel=L"a", ylabel=L"a'=g(a,l)", xlims=(0.0, prices.a_u), ylims=(0, 1), legend=:topleft)
plot!(output[1].gridk0, output[1].pplus[1, 4, :, 1, 2], color=:red, linestyle=:solid, linewidth=2, label=L"rural urban")
plot!(output[1].gridk0, output[1].pplus[1, 4, :, 1, 3], color=:black, linestyle=:solid, linewidth=2, label=L"urban ag")
plot!(output[1].gridk0, output[1].pplus[1, 4, :, 1, 4], color=:blue, linestyle=:dash, linewidth=2, label=L"urban non-ag")
# plot!(a, pplus[1, 4, :, 2], color=:red, linestyle=:dash, linewidth=2, label=L"cg,l_{mid}")
# plot!(a, pplus[1, 7, :, 2], color=:black, linestyle=:dash, linewidth=2, label=L"cg,l_{high}")
savefig("figures/fig_sfun.pdf")



