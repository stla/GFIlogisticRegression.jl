module GFIlogisticRegression

import Junuran
import Polyhedra
import LinearAlgebra

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
  exp_x / (1+exp_x) / (1+exp_x)
end

function rcd(n, P, b, B)
  d = length(B)
  pdf = function(u)
    logit_u = map(log, u ./ (1 .- u))
    dlogit_u = 1 ./ (u .* (1 .- u))
    x = map(dlogis, P * logit_u + b)
    return prod(x) * prod(dlogit_u)
  end
  ctr = map(expit, B)
  gen = Junuran.urgen_vnrou(d, pdf, ctr, nothing, zeros(d), ones(d))
  sims = Junuran.ursample(gen, n)
  return map(logit, sims)
end

function fidSampleLR(y, X, N, thresh = N/2)
  (n, p) = size(X)
  weight = ones(n, N)
  ESS = N .* ones(n)
#  H = Vector{Polyhedra.MixedMatHRep{Rational{BigInt}{Int64},Array{Rational{BigInt}{Int64},2}}}(undef, N)
  CC = Vector{Array{Rational{BigInt},2}}(undef, N)
  cc = Vector{Vector{Rational{BigInt}}}(undef, N)
  # Kstart ####
  Kstart = [1]
  i = 1
  rk = 1
  while rk < p
    i += 1
    Kstart_plus_i = vcat(Kstart, i)
    if LinearAlgebra.rank(X[Kstart_plus_i, 1:end]) == rk + 1
      Kstart = Kstart_plus_i
      rk += 1
    end
  end
  Xstart = convert(Array{Rational{BigInt},2}, X[Kstart, 1:end])
  ystart = y[Kstart]
  K = setdiff(1:n, Kstart)
  XK = convert(Array{Rational{BigInt},2}, X[K, 1:end])
  yK = y[K]
  # t = 1 to p ####
  At = Array{Float64}(undef, p, N)
  for i in 1:N
    a = map(logit, rand(p))
    At[1:end, i] = a
    C = Array{Rational{BigInt},2}(undef, p, p)
    c = Vector{Rational{BigInt}}(undef, p)
    for j in 1:p
      if ystart[j] == 0
        C[j, 1:end] = Xstart[j, 1:end]
        c[j] = convert(Rational{BigInt}, a[j])
      else
        C[j, 1:end] = -Xstart[j, 1:end]
        c[j] = convert(Rational{BigInt}, -a[j])
      end
    end
    CC[i] = C
    cc[i] = c
  end
  # t from p+1 to n ####
  for t in 1:(n-p)
    At = vcat(At, Array{Float64,2}(undef, 1, N))
    Xt = XK[t, 1:end]
    for i in 1:N
      H = Polyhedra.hrep(CC[i], cc[i])
      plyhdrn = Polyhedra.polyhedron(H)
      pts = collect(Polyhedra.points(plyhdrn))
      if yK[t] == 0
        MIN = convert(Float64, minimum(LinearAlgebra.transpose(Xt) * hcat(pts...)))
        atilde = rtlogis2(MIN)
        weight[t, i] = 1 - expit(MIN)
        CC[i] = vcat(CC[i], reshape(Xt, 1, :))
        cc[i] = vcat(cc[i], convert(Rational{BigInt}, atilde))
      else
        MAX = convert(Float64, maximum(LinearAlgebra.transpose(Xt) * hcat(pts...)))
        atilde = rtlogis1(MAX)
        weight[t, i] = expit(MAX)
        CC[i] = vcat(CC[i], reshape(-Xt, 1, :))
        cc[i] = vcat(cc[i], convert(Rational{BigInt}, -atilde))
      end
      At[p+t, i] = atilde
    end
    WT = prod(weight; dims = 1)[1, 1:end]
    WTnorm = WT ./ sum(WT)
    ESS[p+t] = 1 / sum(WTnorm .* WTnorm)
  end
  return ESS
end # fidSampleLR

end # module
