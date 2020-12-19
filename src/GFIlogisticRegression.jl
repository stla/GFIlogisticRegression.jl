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
  Xstart = X[Kstart, 1:end]
  ystart = y[Kstart]
  K = setdiff(1:n, Kstart)
  XK = X[K, 1:end]
  yK = y[K]
  # t = 1 to p ####
  At = Array{Float64}(undef, p, n)
  for i in 1:N

  end

end # fidSampleLR

end # module
