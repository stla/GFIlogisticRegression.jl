module GFIlogisticRegression

import Junuran
import Polyhedra

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

end # module
