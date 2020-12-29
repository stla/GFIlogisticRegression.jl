module GFIlogisticRegression

import Polyhedra
using CDDLib
import LinearAlgebra
import Distributions
import Optim
import StatsBase
import StatsModels
import DataFrames
import DataFramesMeta # to use @with for confidence interval of a parameter: @with(df, :groupA - :groupB)

export fidSummary
export fidConfInt
export fidQuantile
export fidProb
export fidSampleLR


function logit(u) # = qlogis
  return log(u / (1-u))
end

function expit(x) # = plogis
  return 1 / (1+exp(-x))
end

function rtlogis1(x)
  b = expit(x)
  if b == 0
    return x
  end
  return logit(b * rand())
end

function rtlogis2(x)
  a = expit(x)
  if a == 1
    return x
  end
  return logit(a + (1-a) * rand())
end

function from01(u)
  return map(log, u ./ (1 .- u))
end

function from01scalar(u)
  return log(u / (1 - u))
end

function to01(x)
  return 1 / (1 + exp(-x))
end

function dfrom01(u)
  return 1 / (u * (1 - u))
end

function dlogis(x)
  expminusx = map(exp, -x)
  oneplusexpminusx = 1 .+ expminusx
  return expminusx ./ (oneplusexpminusx .* oneplusexpminusx)
end

function ldlogis(x)
  return -x - 2 * map(log1p, map(exp, -x))
end

function dldlogis(x)
  return 1 .- 2 ./ (1 .+ map(exp, -x))
end

function forig(x, P, b)
  return prod(dlogis(P * x + b))
end

function f(u, P, b)
  return prod(dlogis(P * from01(u) + b))
end

function logf(u, P, b)
  return sum(ldlogis(P * from01(u) + b))
end

function df(ui, Pi, y1, y2)
  return y1 * dfrom01(ui) * sum(Pi .* y2)
end

function dlogf(ui, Pi, y2)
  return dfrom01(ui) * sum(Pi .* y2)
end

function get_umax(P, b)
  d = size(P, 2)
  fn = function(u)
    return -logf(u, P, b)
  end
  grfn! = function(storage, u)
    y2 = dldlogis(P * from01(u) + b)
    for i in 1:d
      storage[i] = -dlogf(u[i], P[:, i], y2)
    end
  end
  eta = sqrt(eps())
  lower = eta * ones(d)
  upper = 1.0 .- lower
  init = 0.5 * ones(d)
  opt = Optim.optimize(
    fn, grfn!, lower, upper, init, Optim.Fminbox(Optim.LBFGS())
  )
  return (
    mu = from01(opt.minimizer),
    umax = exp(-opt.minimum)^(2 / (d + 2))
  )
end

function get_vmin_i(P, b, j, mu)
  d = size(P, 2)
  alpha = 1 / (d + 2)
  fn = function(u)
    return f(u, P, b)^alpha * (from01scalar(u[j]) - mu[j])
  end
  grfn! = function(storage, u)
    y1alpha = f(u, P, b)^alpha
    y2 = dldlogis(P * from01(u) + b)
    diff = from01scalar(u[j]) - mu[j]
    storage[j] = y1alpha * dfrom01(u[j]) *
      (alpha * sum(P[:, j] .* y2) * diff + 1)
    others = deleteat!(collect(1:d), j)
    for i in others
      storage[i] = y1alpha * dfrom01(u[i]) * alpha * sum(P[:, i] .* y2) * diff
    end
  end
  eta = sqrt(eps())
  lower = eta * ones(d)
  upper = (1-eta) * ones(d)
  upper[j] = to01(mu[j])
  init = 0.5 * ones(d)
  init[j] = to01(mu[j]) / 2.0
  opt = Optim.optimize(
    fn, grfn!, lower, upper, init, Optim.Fminbox(Optim.LBFGS())
  )
  return opt.minimum
end

function get_vmin(P, b, mu)
  d = size(P, 2)
  vmin = Vector{Float64}(undef, d)
  for i in 1:d
    vmin[i] = get_vmin_i(P, b, i, mu)
  end
  return vmin
end

function get_vmax_i(P, b, j, mu)
  d = size(P, 2)
  alpha = 1 / (d + 2)
  fn = function(u)
    return -f(u, P, b)^alpha * (from01scalar(u[j]) - mu[j])
  end
  grfn! = function(storage, u)
    y1alpha = f(u, P, b)^alpha
    y2 = dldlogis(P * from01(u) + b)
    diff = from01scalar(u[j]) - mu[j]
    storage[j] = -y1alpha * dfrom01(u[j]) *
      (alpha * sum(P[:, j] .* y2) * diff + 1)
    others = deleteat!(collect(1:d), j)
    for i in others
      storage[i] = -y1alpha * dfrom01(u[i]) * alpha * sum(P[:, i] .* y2) * diff
    end
  end
  eta = sqrt(eps())
  lower = eta * ones(d)
  lower[j] = to01(mu[j])
  upper = (1-eta) * ones(d)
  init = 0.5 * ones(d)
  init[j] = (to01(mu[j]) + 1) / 2.0
  opt = Optim.optimize(
    fn, grfn!, lower, upper, init, Optim.Fminbox(Optim.LBFGS())
  )
  return -opt.minimum
end

function get_vmax(P, b, mu)
  d = size(P, 2)
  vmax = Vector{Float64}(undef, d)
  for i in 1:d
    vmax[i] = get_vmax_i(P, b, i, mu)
  end
  return vmax
end

function get_bounds(P, b)
  (mu, umax) = get_umax(P, b)
  return (
    mu = mu,
    umax = umax,
    vmin = get_vmin(P, b, mu),
    vmax = get_vmax(P, b, mu)
  )
end

function rcd(n, P, b)
  d = size(P, 2)
  sims = Vector{Vector{Float64}}(undef, n)
  (mu, umax, vmin, vmax) = get_bounds(P, b)
  if any(vmin .>= vmax)
    error("vmin .>= vmax")
  end
  k = 0
  while k < n
    u = umax * rand()
    v = vmin + (vmax - vmin) .* rand(d)
    x = v / sqrt(u) + mu
    if u < forig(x, P, b)^(2 / (d+2))
      k += 1
      sims[k] = x
    end
  end
  return sims
end

"""
    fidSampleLR(formula, data, N[, gmp][, thresh])

Fiducial sampling of the parameters of the logistic regression model.

# Arguments
- `formula`: a formula describing the model
- `data`: data frame in which the variables of the model can be found
- `N`: number of simulations
- `gmp`: whether to use exact arithmetic in the algorithm
- `thresh`: the threshold used in the sequential sampler; the default `N/2` should not be changed

# Example

    using GFIlogisticRegression, DataFrames, StatsModels
    data = DataFrame(
      y = [0, 0, 1, 1, 1],
      x = [-2, -1, 0, 1, 2]
    )
    fidsamples = fidSampleLR(@formula(y ~ x), data, 3000)
"""
function fidSampleLR(formula, data, N, gmp = false, thresh = N/2)
  y = StatsModels.response(formula, data)
  mf = StatsModels.ModelFrame(formula, data)
  mm = StatsModels.ModelMatrix(mf)
  X = mm.m
  (_, colnames) = StatsModels.coefnames(mf.f)
  (n, p) = size(X)
  weight = ones(n, N)
  local WTnorm
  ESS = N .* ones(n)
  T = gmp ? Rational{BigInt} : Float64
  cdd = gmp ? :exact : :float
  CC = Vector{Array{T,2}}(undef, N)
  cc = Vector{Vector{T}}(undef, N)
  # Kstart ####
  Kstart = [1]
  i = 1
  rk = 1
  while rk < p
    i += 1
    Kstart_plus_i = vcat(Kstart, i)
    if LinearAlgebra.rank(X[Kstart_plus_i, :]) == rk + 1
      Kstart = Kstart_plus_i
      rk += 1
    end
  end
  @inbounds Xstart = X[Kstart, :]
  qXstart = convert(Array{T,2}, Xstart)
  ystart = y[Kstart]
  K = setdiff(1:n, Kstart)
  @inbounds XK = X[K, :]
  qXK = convert(Array{T,2}, XK)
  @inbounds yK = y[K]
  # t = 1 to p ####
  At = Array{Float64}(undef, p, N)
  for i in 1:N
    a = map(logit, rand(p))
    @inbounds At[:, i] = a
    C = Array{T,2}(undef, p, p)
    c = Vector{T}(undef, p)
    for j in 1:p
      @inbounds if ystart[j] == 0
        @inbounds C[j, :] = qXstart[j, :]
        @inbounds c[j] = convert(T, a[j])
      else
        @inbounds C[j, :] = -qXstart[j, :]
        @inbounds c[j] = convert(T, -a[j])
      end
    end
    @inbounds CC[i] = C
    @inbounds cc[i] = c
  end
  # t from p+1 to n ####
  for t in 1:(n-p)
    #println(t)
    At = vcat(At, Array{Float64,2}(undef, 1, N))
    @inbounds qXt = qXK[t, :]
    qXt_row = reshape(qXt, 1, :)
    qXtt = LinearAlgebra.transpose(qXt)

    @inbounds if yK[t] == 0
      for i in 1:N
        @inbounds H = Polyhedra.hrep(CC[i], cc[i])
        plyhdrn = Polyhedra.polyhedron(H, CDDLib.Library(cdd))
        pts = collect(Polyhedra.points(plyhdrn))
        MIN = convert(
          Float64, minimum(qXtt * hcat(pts...))
        )
        atilde = rtlogis2(MIN)
        @inbounds weight[t, i] = 1 - expit(MIN)
        @inbounds CC[i] = vcat(CC[i], qXt_row)
        @inbounds cc[i] = vcat(cc[i], convert(T, atilde))
        @inbounds At[p+t, i] = atilde
      end
    else
      for i in 1:N
        @inbounds H = Polyhedra.hrep(CC[i], cc[i])
        plyhdrn = Polyhedra.polyhedron(H, CDDLib.Library(cdd))
        pts = collect(Polyhedra.points(plyhdrn))
        MAX = convert(
          Float64, maximum(qXtt * hcat(pts...))
        )
        atilde = rtlogis1(MAX)
        @inbounds weight[t, i] = expit(MAX)
        @inbounds CC[i] = vcat(CC[i], -qXt_row)
        @inbounds cc[i] = vcat(cc[i], convert(T, -atilde))
        @inbounds At[p+t, i] = atilde
      end
    end
    WT = prod(weight; dims = 1)[1, :]
    WTnorm = WT ./ sum(WT)
    @inbounds ESS[p+t] = 1.0 / sum(WTnorm .* WTnorm)
    @inbounds if ESS[p+t] < thresh || t == n-p
      #println("alteration...")
      Nsons = rand(Distributions.Multinomial(N, WTnorm))
      counter = 1
      At_new = Array{Float64, 2}(undef, p+t, 0)
      @inbounds D = vcat(Xstart, XK[1:t, :])
      QR = LinearAlgebra.qr(D)
      @inbounds P = QR.Q[:, 1:p]#orth(D)
      Pt = LinearAlgebra.transpose(P)
      QQt = LinearAlgebra.I - P*Pt
      M = inv(LinearAlgebra.transpose(D)*D) * LinearAlgebra.transpose(D) * P
      CCtemp = Vector{Array{T,2}}(undef, N)
      cctemp = Vector{Vector{T}}(undef, N)
      for i in 1:N
        @inbounds ncopies = Nsons[i]
        if ncopies >= 1
          @inbounds CCtemp[counter] = CC[i]
          @inbounds cctemp[counter] = cc[i]
          @inbounds At_new = hcat(At_new, At[:, i])
          if ncopies > 1
            @inbounds H = Polyhedra.hrep(CC[i], cc[i])
            plyhdrn = Polyhedra.polyhedron(H, CDDLib.Library(cdd))
            pts = collect(Polyhedra.points(plyhdrn))
            lns = collect(Polyhedra.lines(plyhdrn))
            rys = collect(Polyhedra.rays(plyhdrn))
            @inbounds b = QQt * At[:, i]
            @inbounds B = Pt * At[:, i]
            BTILDES = rcd(ncopies-1, P, b)
            for j in 2:ncopies
              VT_new = Vector{Vector{T}}(undef, length(pts))
              @inbounds Btilde = BTILDES[j-1]
              At_tilde = P * Btilde .+ b
              At_new = hcat(At_new, At_tilde)
              for k in 1:length(pts)
                pt = convert(Vector{Float64}, pts[k])
                VT_new[k] =
                  convert(Vector{T}, pt .- M * (B .- Btilde))
              end
              V = Polyhedra.vrep(VT_new, lns, rys)
              plyhdrn = Polyhedra.polyhedron(V, CDDLib.Library(cdd))
              #HlfSpcs = Polyhedra.hrep(plyhdrn).halfspaces
              #A = map(x -> x.a, HlfSpcs)
              H = Polyhedra.hrep(plyhdrn)
              Hmat = convert(Array{T,2}, CDDLib.extractA(unsafe_load(H.matrix)))
              @inbounds CCtemp[counter + j - 1] = -Hmat[:, 2:end] #LinearAlgebra.transpose(hcat(A...))
              @inbounds cctemp[counter + j - 1] = Hmat[:, 1]#map(x -> x.β, HlfSpcs)
            end
          end
          counter += ncopies
        end
      end
      CC = CCtemp
      cc = cctemp
      At = At_new
      if t < n-p
        weight = ones(n, N)
      end
    end
  end
  Beta = Array{Float64, 2}(undef, N, p)
  for i in 1:N
    @inbounds H = Polyhedra.hrep(CC[i], cc[i])
    plyhdr = Polyhedra.polyhedron(H, CDDLib.Library(cdd))
    vertices = collect(Polyhedra.points(plyhdr))
    pts = hcat(vertices...)
    for j in 1:p
      pt = convert(Vector{Float64}, pts[j, :])
      if rand() < 0.5
        @inbounds Beta[i, j] = minimum(pt)
      else
        @inbounds Beta[i, j] = maximum(pt)
      end
    end
  end
  dfBeta = DataFrames.DataFrame(Beta, Symbol.(colnames))
  return (Beta = dfBeta, Weights = WTnorm)
end # fidSampleLR

"""
    fidSummary(fidsamples)

Summary of the fiducial simulations.

# Argument
- `fidsamples`: an output of `fidSampleLR`

# Example

    using GFIlogisticRegression, DataFrames, StatsModels
    data = DataFrame(
      y = [0, 0, 1, 1, 1],
      x = [-2, -1, 0, 1, 2]
    )
    fidsamples = fidSampleLR(@formula(y ~ x), data, 3000)
    fidSummary(fidsamples)
"""
function fidSummary(fidsamples)
  (_, p) = size(fidsamples.Beta)
  gdf = DataFrames.groupby(DataFrames.stack(fidsamples.Beta, 1:p), :variable)
  wghts = fidsamples.Weights
  function fsummary(x)
    return (
      mean = sum(wghts .* x),
      median = StatsBase.quantile(x, StatsBase.weights(wghts), 0.5),
      lwr = StatsBase.quantile(x, StatsBase.weights(wghts), 0.025),
      upr = StatsBase.quantile(x, StatsBase.weights(wghts), 0.975)
    )
  end
  return DataFrames.combine(gdf, DataFrames.valuecols(gdf) .=> fsummary => DataFrames.AsTable)
end

"""
    fidConfInt(parameter, fidsamples, conf)

Fiducial confidence interval of a parameter of interest.

# Arguments
- `parameter`: an expression of the parameter of interest given as a string; see the example
- `fidsamples`: an output of `fidSampleLR`
- `conf`: confidence level

# Example

    using GFIlogisticRegression, DataFrames, StatsModels
    data = DataFrame(
      y = [0, 0, 1, 1, 1, 1],
      group = ["A", "A", "A", "B", "B", "B"]
    )
    fidsamples = fidSampleLR(@formula(y ~ 0 + group), data, 3000)
    fidConfInt(":\\\"group: A\\\" - :\\\"group: B\\\"", fidsamples, 0.95)
"""
function fidConfInt(parameter, fidsamples, conf = 0.95)
  x = eval(:(DataFramesMeta.@with($(fidsamples.Beta), $(Meta.parse(parameter)))))
  wghts = fidsamples.Weights
  halpha = (1.0 - conf) / 2.0
  qntls = StatsBase.quantile(x, StatsBase.weights(wghts), [halpha; 1.0-halpha])
  return (lower = qntls[1], upper = qntls[2])
end

"""
    fidQuantile(parameter, fidsamples, p)

Fiducial quantile of a parameter of interest.

# Arguments
- `parameter`: an expression of the parameter of interest given as a string; see the example
- `fidsamples`: an output of `fidSampleLR`
- `p`: quantile level, between 0 and 1

# Example

    using GFIlogisticRegression, DataFrames, StatsModels
    data = DataFrame(
      y = [0, 0, 1, 1, 1, 1],
      group = ["A", "A", "A", "B", "B", "B"]
    )
    fidsamples = fidSampleLR(@formula(y ~ 0 + group), data, 3000)
    fidQuantile(":\\\"group: A\\\" ./ :\\\"group: B\\\"", fidsamples, 0.5)
"""
function fidQuantile(parameter, fidsamples, p)
  x = eval(:(DataFramesMeta.@with($(fidsamples.Beta), $(Meta.parse(parameter)))))
  wghts = fidsamples.Weights
  return StatsBase.quantile(x, StatsBase.weights(wghts), p)
end

"""
    fidProb(parameter, fidsamples, q)

Fiducial non-exceedance probability of a parameter of interest.

# Arguments
- `parameter`: an expression of the parameter of interest given as a string; see the example
- `fidsamples`: fiducial simulations, an output of `fidSampleLR`
- `q`: the non-exceedance threshold

# Example

    using GFIlogisticRegression, DataFrames, StatsModels
    data = DataFrame(
      y = [0, 0, 1, 1, 1],
      x = [-2, -1, 0, 1, 2]
    )
    fidsamples = fidSampleLR(@formula(y ~ x), data, 3000)
    fidProb("map(exp, :x)", fidsamples, 1) # this is Pr(exp(x) <= 1)
"""
function fidProb(parameter, fidsamples, q)
  x = eval(:(DataFramesMeta.@with($(fidsamples.Beta), $(Meta.parse(parameter)))))
  wghts = fidsamples.Weights
  return sum(wghts[findall(x .<= q)])
end

end # module

#=
P = [
  -0.82 -0.18;
  -0.41 -0.37;
  0.0   -0.55;
  0.41  -0.73
]
b = [-0.50; 1.58; -1.66; 0.58]
=#

#=
y = [0, 0, 1, 1, 1]
X = [1 -2; 1 -1; 1 0; 1 1; 1 2]
fidsamples = fidSampleLR(y, X, 5)
=#

#=
using DataFrames, StatsModels
data = DataFrame(
  y = [0, 0, 1, 1, 1],
  x = [-2, -1, 0, 1, 2]
)
fidsamples = fidSampleLR(@formula(y ~ x), data, 5)

fidSummary(fidsamples)
fidConfInt("map(exp, :x)", fidsamples)
fidConfInt(":x ./ :\"(Intercept)\"", fidsamples)
=#

#=
data = DataFrame(
  y = [0, 0, 1, 1, 1, 1],
  group = ["A", "A", "A", "B", "B", "B"]
)
fidsamples = fidSampleLR(@formula(y ~ 0 + group), data, 3000)
fidConfInt(":\"group: A\" - :\"group: B\"", fidsamples, 0.95)
=#

#=
    for i in 1:N
      @inbounds H = Polyhedra.hrep(CC[i], cc[i])
      plyhdrn = Polyhedra.polyhedron(H, CDDLib.Library(cdd))
      pts = collect(Polyhedra.points(plyhdrn))
      @inbounds if yK[t] == 0
        MIN = convert(
          Float64, minimum(qXtt * hcat(pts...))
        )
        atilde = rtlogis2(MIN)
        @inbounds weight[t, i] = 1 - expit(MIN)
        @inbounds CC[i] = vcat(CC[i], qXt_row)
        @inbounds cc[i] = vcat(cc[i], convert(T, atilde))
      else
        MAX = convert(
          Float64, maximum(qXtt * hcat(pts...))
        )
        atilde = rtlogis1(MAX)
        @inbounds weight[t, i] = expit(MAX)
        @inbounds CC[i] = vcat(CC[i], -qXt_row)
        @inbounds cc[i] = vcat(cc[i], convert(T, -atilde))
      end
      @inbounds At[p+t, i] = atilde
    end
=#
