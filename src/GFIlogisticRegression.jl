module GFIlogisticRegression

import Polyhedra
using CDDLib
import LinearAlgebra
import Distributions
import Optim
import StatsBase
import StatsModels

export summary
export fidSampleLR


function logit(u) # = qlogis
  return log(u / (1-u))
end

function expit(x) # = plogis
  return 1 / (1+exp(-x))
end

function rtlogis1(x)
  b = expit(b)
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

###########################################################

P = [
  -0.82 -0.18;
  -0.41 -0.37;
  0.0   -0.55;
  0.41  -0.73
]
b = [-0.50; 1.58; -1.66; 0.58]

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

function fidSampleLR(formula, data, N, thresh = N/2)
  println("start")
  y = StatsModels.response(formula, data)
  mf = StatsModels.ModelFrame(formula, data)
  mm = StatsModels.ModelMatrix(mf)
  X = mm.m
  #coefnames(mf.f)
  (n, p) = size(X)
  weight = ones(n, N)
  local WTnorm
  ESS = N .* ones(n)
  CC = Vector{Array{Rational{BigInt},2}}(undef, N)
  cc = Vector{Vector{Rational{BigInt}}}(undef, N)
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
  qXstart = convert(Array{Rational{BigInt},2}, convert(Array{Rational{BigInt},2}, Xstart))
  ystart = y[Kstart]
  K = setdiff(1:n, Kstart)
  @inbounds XK = X[K, :]
  qXK = convert(Array{Rational{BigInt},2}, convert(Array{Rational{BigInt},2}, XK))
  @inbounds yK = y[K]
  # t = 1 to p ####
  At = Array{Float64}(undef, p, N)
  for i in 1:N
    a = map(logit, rand(p))
    @inbounds At[:, i] = a
    C = Array{Rational{BigInt},2}(undef, p, p)
    c = Vector{Rational{BigInt}}(undef, p)
    for j in 1:p
      @inbounds if ystart[j] == 0
        @inbounds C[j, :] = qXstart[j, :]
        @inbounds c[j] = convert(Rational{BigInt}, convert(Rational{BigInt}, a[j]))
      else
        @inbounds C[j, :] = -qXstart[j, :]
        @inbounds c[j] = convert(Rational{BigInt}, convert(Rational{BigInt}, -a[j]))
      end
    end
    @inbounds CC[i] = C
    @inbounds cc[i] = c
  end
  # t from p+1 to n ####
  for t in 1:(n-p)
    println(t)
    At = vcat(At, Array{Float64,2}(undef, 1, N))
    @inbounds qXt = qXK[t, :]
    qXt_row = reshape(qXt, 1, :)
    qXtt = LinearAlgebra.transpose(qXt)
    for i in 1:N
      @inbounds H = Polyhedra.hrep(CC[i], cc[i])
      plyhdrn = Polyhedra.polyhedron(H, CDDLib.Library(:exact))
      pts = collect(Polyhedra.points(plyhdrn))
      @inbounds if yK[t] == 0
        MIN = convert(
          Float64, minimum(qXtt * hcat(pts...))
        )
        atilde = rtlogis2(MIN)
        @inbounds weight[t, i] = 1 - expit(MIN)
        @inbounds CC[i] = vcat(CC[i], qXt_row)
        @inbounds cc[i] = vcat(cc[i], convert(Rational{BigInt}, convert(Rational{BigInt}, atilde)))
      else
        MAX = convert(
          Float64, maximum(qXtt * hcat(pts...))
        )
        atilde = rtlogis1(MAX)
        @inbounds weight[t, i] = expit(MAX)
        @inbounds CC[i] = vcat(CC[i], -qXt_row)
        @inbounds cc[i] = vcat(cc[i], convert(Rational{BigInt}, convert(Rational{BigInt}, -atilde)))
      end
      @inbounds At[p+t, i] = atilde
    end
    WT = prod(weight; dims = 1)[1, :]
    WTnorm = WT ./ sum(WT)
    @inbounds ESS[p+t] = 1.0 / sum(WTnorm .* WTnorm)
    @inbounds if ESS[p+t] < thresh || t == n-p
      println("alteration...")
      Nsons = rand(Distributions.Multinomial(N, WTnorm))
      counter = 1
      At_new = Array{Float64, 2}(undef, p+t, 0)
      @inbounds D = vcat(Xstart, XK[1:t, :])
      QR = LinearAlgebra.qr(D)
      @inbounds P = QR.Q[:, 1:p]#orth(D)
      Pt = LinearAlgebra.transpose(P)
      QQt = LinearAlgebra.I - P*Pt
      M = inv(LinearAlgebra.transpose(D)*D) * LinearAlgebra.transpose(D) * P
      CCtemp = Vector{Array{Rational{BigInt},2}}(undef, N)
      cctemp = Vector{Vector{Rational{BigInt}}}(undef, N)
      for i in 1:N
        @inbounds ncopies = Nsons[i]
        if ncopies >= 1
          @inbounds CCtemp[counter] = CC[i]
          @inbounds cctemp[counter] = cc[i]
          @inbounds At_new = hcat(At_new, At[:, i])
          if ncopies > 1
            @inbounds H = Polyhedra.hrep(CC[i], cc[i])
            plyhdrn = Polyhedra.polyhedron(H, CDDLib.Library(:exact))
            pts = collect(Polyhedra.points(plyhdrn))
            lns = collect(Polyhedra.lines(plyhdrn))
            rys = collect(Polyhedra.rays(plyhdrn))
            @inbounds b = QQt * At[:, i]
            @inbounds B = Pt * At[:, i]
            BTILDES = rcd(ncopies-1, P, b)
            for j in 2:ncopies
              VT_new = Vector{Vector{Rational{BigInt}}}(undef, length(pts))
              @inbounds Btilde = BTILDES[j-1]
              At_tilde = P * Btilde .+ b
              At_new = hcat(At_new, At_tilde)
              for k in 1:length(pts)
                pt = convert(Vector{Float64}, pts[k])
                VT_new[k] =
                  convert(Vector{Rational{BigInt}}, convert(Vector{Rational{BigInt}}, pt .- M * (B .- Btilde)))
              end
              V = Polyhedra.vrep(VT_new, lns, rys)
              plyhdrn = Polyhedra.polyhedron(V, CDDLib.Library(:exact))
              #HlfSpcs = Polyhedra.hrep(plyhdrn).halfspaces
              #A = map(x -> x.a, HlfSpcs)
              H = Polyhedra.hrep(plyhdrn)
              Hmat = convert(Array{Rational{BigInt},2}, CDDLib.extractA(unsafe_load(H.matrix)))
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
    plyhdr = Polyhedra.polyhedron(H)
    vertices = collect(Polyhedra.points(plyhdr))
    pts = hcat(vertices...)
    for j in 1:p
      if rand() < 0.5
        @inbounds Beta[i, j] = minimum(pts[j, :])
      else
        @inbounds Beta[i, j] = maximum(pts[j, :])
      end
    end
  end
  return (Beta = Beta, Weights = WTnorm)
end # fidSampleLR

function summary(fidsamples)
  println("beta1:")
  println(sum(fidsamples.Beta[:, 1] .* fidsamples.Weights))
  println(
    StatsBase.quantile(fidsamples.Beta[:, 1], StatsBase.weights(fidsamples.Weights), [0.025,0.975])
  )
  println("beta2:")
  println(sum(fidsamples.Beta[:, 2] .* fidsamples.Weights))
  println(
    StatsBase.quantile(fidsamples.Beta[:, 2], StatsBase.weights(fidsamples.Weights), [0.025,0.975])
  )
end

end # module

#=
y = [0, 0, 1, 1, 1]
X = [1 -2; 1 -1; 1 0; 1 1; 1 2]
fidsamples = fidSampleLR(y, X, 5)
=#

using DataFrames, StatsModels
data = DataFrame(
  y = [0, 0, 1, 1, 1],
  x = [-2, -1, 0, 1, 2]
)

fidsamples = fidSampleLR(@formula(y ~ x), data, 5)
summary(fidsamples)
