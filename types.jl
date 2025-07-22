struct Params
    mu::Float64
    beta::Float64
    delta::Float64
    alpha::Float64
    b::Float64
    gamma_z::Float64
    gamma_q::Float64
    gamma_h::Float64
    NH::Int
    h::Vector{Float64}
    z::Vector{Float64}
    lh::Vector{Float64}
    lz::Vector{Float64}
    lq::Vector{Float64}
    prob::Matrix{Float64}
    income_thres_top10_base::Float64
    NA::Int
    NA2::Int
    NZ::Int
    NI::Int
    mv_cost::Matrix{Float64}
    land_risk::Vector{Float64}
    r_land::Float64
    sigma_e::Float64
    dutil::Vector{Float64}
    zeta_rr::Float64
end

struct Prices
    r::Float64
    wage::Vector{Float64}
    phi::Float64
    a::Vector{Float64}
    ell::Float64
    avg_income::Float64
    tuition::Vector{Float64}
    a_u::Float64
    KL::Float64
    land_lost::Float64
end

struct Dec
    aplus::Array{Float64, 4}
    e::Array{Float64, 4}
    pplus::Array{Float64, 5}
    v::Array{Float64, 4}
    v_add::Array{Float64, 4}
end

struct Meas
    m::Array{Float64, 4}  # NI × NH × NA × NZ
end

struct Agg
    meank::Float64
    meanL::Float64
    land_lost::Float64
    avg_income::Float64
    mass_i::Vector{Float64}
    mass_z::Vector{Float64}
    mass_a::Vector{Float64}
    mass_h::Vector{Float64}
end

struct Guess_base
    KL::Float64
    land_lost::Float64
    avg_income::Float64
end