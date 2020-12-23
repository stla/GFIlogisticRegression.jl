module GFIlogisticRegression

import Polyhedra
#import CDDLib
import LinearAlgebra
import Distributions
import Optim
import StatsBase

export summary
export fidSampleLR


function logit(u) # = qlogis
  return log(u / (1-u))
end

function expit(x) # = plogis
  return 1 / (1+exp(-x))
end

function rtlogis1(b)
  out = logit(expit(b) * rand())
  if out == -Inf
    qmin = logit(1e-16)
    if b > qmin
      out = qmin + (b-qmin) * rand()
    else
      out = b
    end
  end
  return out
end

function rtlogis2(b)
  p = expit(b)
  out = logit(p + (1-p) * rand())
  if out == Inf
    qmax = logit(1-1e-16)
    if b < qmax
      out = b + (qmax-b) * rand()
    else
      out = b
    end
  end
  return out
end

function dlogis(x)
  exp_x = exp(x)
  return exp_x / (1+exp_x) / (1+exp_x)
end

function dlogit(u)
  return 1.0 / (u * (1.0 - u))
end

function ldlogit(U)
  return map(u -> -log(u * (1.0-u)), U)
end

function ldlogis(X)
  return X - 2.0 * map(x -> log1p(exp(x)), X)
end

function dldlogis(X)
  return 1.0 .- 2.0 ./ (1.0 .+ map(exp, -X))
end

function log_f(u, P, b)
  x = P * map(logit, u) + b;
  return sum(ldlogis(x)) + sum(ldlogit(u))
end

function dlog_f(ui, Pi, y)
  return dlogit(ui) * sum(Pi .* y) + (2.0 * ui - 1.0) / (ui * (1.0 - ui))
end

P = [
  -0.82 -0.18;
  -0.41 -0.37;
  0.0   -0.55;
  0.41  -0.73
]
b = [-0.50; 1.58; -1.66; 0.58]

function get_umax0(P, b, init)
  d = size(P, 2)
  fn = function(u)
    return -log_f(u, P, b)
  end
  grfn! = function(storage, u)
    y = dldlogis(P * map(logit, u) .+ b)
    for i in 1:d
      storage[i] = -dlog_f(u[i], P[:, i], y)
    end
  end
  eta = sqrt(eps())
  lower = eta * ones(d)
  upper = 1.0 .- lower
  od = Optim.OnceDifferentiable(fn, grfn!, init)
  results = Optim.optimize(
    fn, grfn!, lower, upper, init, Optim.Fminbox(Optim.LBFGS())
  )
  return (results.minimizer, results.minimum)
end

function get_umax(P, b)
  d = size(P, 2)
  cp = Iterators.product(ntuple(i -> [0.01, 0.5, 0.99], d)...)
  inits = hcat(map(collect, collect(cp))...)
  n = size(inits, 2)
  mins = Vector{Float64}(undef, n)
  ats = Vector{Vector{Float64}}(undef, n)
  for i in 1:n
    (ats[i], mins[i]) = get_umax0(P, b, inits[:, i])
  end
  imin = argmin(mins)
  return (
    mu = ats[imin],
    umax = exp(-mins[imin])^(2.0 / (2.0 + d))
  )
end

function get_vmin_i(P, b, i, mu)
  d = size(P, 2)
  fn = function(u)
    return -log_f(u, P, b) - (d + 2) * log(mu[i] - u[i])
  end
  grfn! = function(storage, u)
    y = dldlogis(P * map(logit, u) .+ b)
    storage[i] = -dlog_f(u[i], P[:, i], y) + (d+2) / (mu[i] - u[i])
    others = deleteat!(collect(1:d), i)
    for j in others
      storage[j] = -dlog_f(u[j], P[:, j], y)
    end
  end
  eta = sqrt(eps())/3
  lower = eta * ones(d)
  upper = ones(d)
  upper[i] = mu[i]
  upper .-= eta
  init = 0.5 * ones(d)
  init[i] = mu[i] / 2.0
  od = Optim.OnceDifferentiable(fn, grfn!, init)
  opt = Optim.optimize(
    fn, grfn!, lower, upper, init, Optim.Fminbox(Optim.LBFGS())
  )
  return -exp(-opt.minimum / (d+2))
end

function get_vmin(P, b, mu)
  d = size(P, 2)
  vmin = Vector{Float64}(undef, d)
  for i in 1:d
    vmin[i] = get_vmin_i(P, b, i, mu)
  end
  return vmin
end

function get_vmax_i(P, b, i, mu)
  d = size(P, 2)
  fn = function(u)
    return -log_f(u, P, b) - (d + 2) * log(u[i] - mu[i])
  end
  grfn! = function(storage, u)
    y = dldlogis(P * map(logit, u) .+ b)
    storage[i] = -dlog_f(u[i], P[:, i], y) + (d+2) / (mu[i] - u[i])
    others = deleteat!(collect(1:d), i)
    for j in others
      storage[j] = -dlog_f(u[j], P[:, j], y)
    end
  end
  eta = sqrt(eps())/3
  lower = zeros(d)
  lower[i] = mu[i]
  lower .+= eta
  upper = ones(d) .- eta
  init = 0.5 * ones(d)
  init[i] = (mu[i] + 1.0) / 2.0
  od = Optim.OnceDifferentiable(fn, grfn!, init)
  opt = Optim.optimize(
    fn, grfn!, lower, upper, init, Optim.Fminbox(Optim.LBFGS())
  )
  return exp(-opt.minimum / (d+2))
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
    if all(x .> 0) && all(x .< 1) && ((d+2) * log(u) < 2.0 * log_f(x, P, b))
      k += 1
      sims[k] = map(logit, x)
    end
  end
  return sims
end

function fidSampleLR(y, X, N, thresh = N/2)
  println("start")
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
  qXstart = convert(Array{Rational{BigInt},2}, Xstart)
  ystart = y[Kstart]
  K = setdiff(1:n, Kstart)
  @inbounds XK = X[K, :]
  qXK = convert(Array{Rational{BigInt},2}, XK)
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
        @inbounds c[j] = convert(Rational{BigInt}, a[j])
      else
        @inbounds C[j, :] = -qXstart[j, :]
        @inbounds c[j] = convert(Rational{BigInt}, -a[j])
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
      plyhdrn = Polyhedra.polyhedron(H) #, CDDLib.Library(:exact))
      pts = collect(Polyhedra.points(plyhdrn))
      @inbounds if yK[t] == 0
        MIN = convert(
          Float64, minimum(qXtt * hcat(pts...))
        )
        atilde = rtlogis2(MIN)
        @inbounds weight[t, i] = 1 - expit(MIN)
        @inbounds CC[i] = vcat(CC[i], qXt_row)
        @inbounds cc[i] = vcat(cc[i], convert(Rational{BigInt}, atilde))
      else
        MAX = convert(
          Float64, maximum(qXtt * hcat(pts...))
        )
        atilde = rtlogis1(MAX)
        @inbounds weight[t, i] = expit(MAX)
        @inbounds CC[i] = vcat(CC[i], -qXt_row)
        @inbounds cc[i] = vcat(cc[i], convert(Rational{BigInt}, -atilde))
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
            plyhdrn = Polyhedra.polyhedron(H) #, CDDLib.Library(:exact))
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
                  convert(Vector{Rational{BigInt}}, pt .- M * (B .- Btilde))
              end
              V = Polyhedra.vrep(VT_new, lns, rys)
              plyhdrn = Polyhedra.polyhedron(V)
              HlfSpcs = Polyhedra.hrep(plyhdrn).halfspaces
              A = map(x -> x.a, HlfSpcs)
              @inbounds CCtemp[counter + j - 1] = LinearAlgebra.transpose(hcat(A...))
              @inbounds cctemp[counter + j - 1] = map(x -> x.β, HlfSpcs)
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
