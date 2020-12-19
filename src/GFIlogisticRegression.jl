module GFIlogisticRegression

import Junuran
import Polyhedra
import LinearAlgebra
import Distributions

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

function orth(A)
  if (isempty(A))
    retval = []
  else
    (U, S, V) = LinearAlgebra.svd(A)
    (rows, cols) = size(A)
    tol = maximum(size(A)) * S[1] * eps()
    r = sum(S .> tol)
    if (r > 0)
      retval = -U[:, 1:r]
    else
      retval = zeros(rows, 0)
    end
  end
  return (retval)
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
  Xstart = X[Kstart, :]
  qXstart = convert(Array{Rational{BigInt},2}, Xstart)
  ystart = y[Kstart]
  K = setdiff(1:n, Kstart)
  XK = X[K, :]
  qXK = convert(Array{Rational{BigInt},2}, XK)
  yK = y[K]
  # t = 1 to p ####
  At = Array{Float64}(undef, p, N)
  for i in 1:N
    a = map(logit, rand(p))
    At[:, i] = a
    C = Array{Rational{BigInt},2}(undef, p, p)
    c = Vector{Rational{BigInt}}(undef, p)
    for j in 1:p
      if ystart[j] == 0
        C[j, 1:end] = qXstart[j, :]
        c[j] = convert(Rational{BigInt}, a[j])
      else
        C[j, 1:end] = -qXstart[j, :]
        c[j] = convert(Rational{BigInt}, -a[j])
      end
    end
    CC[i] = C
    cc[i] = c
  end
  # t from p+1 to n ####
  for t in 1:(n-p)
    At = vcat(At, Array{Float64,2}(undef, 1, N))
    qXt = qXK[t, 1:end]
    for i in 1:N
      H = Polyhedra.hrep(CC[i], cc[i])
      plyhdrn = Polyhedra.polyhedron(H)
      pts = collect(Polyhedra.points(plyhdrn))
      if yK[t] == 0
        MIN = convert(
          Float64, minimum(LinearAlgebra.transpose(qXt) * hcat(pts...))
        )
        atilde = rtlogis2(MIN)
        weight[t, i] = 1 - expit(MIN)
        CC[i] = vcat(CC[i], reshape(qXt, 1, :))
        cc[i] = vcat(cc[i], convert(Rational{BigInt}, atilde))
      else
        MAX = convert(
          Float64, maximum(LinearAlgebra.transpose(qXt) * hcat(pts...))
        )
        atilde = rtlogis1(MAX)
        weight[t, i] = expit(MAX)
        CC[i] = vcat(CC[i], reshape(-qXt, 1, :))
        cc[i] = vcat(cc[i], convert(Rational{BigInt}, -atilde))
      end
      At[p+t, i] = atilde
    end
    WT = prod(weight; dims = 1)[1, 1:end]
    WTnorm = WT ./ sum(WT)
    ESS[p+t] = 1.0 / sum(WTnorm .* WTnorm)
    if ESS[p+t] < thresh || t == n-p
      Nsons = rand(Multinomial(N, WTnorm))
      counter = 1
      At_new = Array{Float64, 2}(undef, p+t, 0)
      D = vcat(Xstart, XK[1:t, :])
      P = orth(D)
      Pt = LinearAlgebra.transpose(P)
      QQt = LinearAlgebra.I - P*Pt
      M = inv(LinearAlgebra.transpose(D)*D) * LinearAlgebra.transpose(D) * P
      CCtemp = Vector{Array{Rational{BigInt},2}}(undef, N)
      cctemp = Vector{Vector{Rational{BigInt}}}(undef, N)
      for i in 1:N
        ncopies = Nsons[i]
        if ncopies >= 1
          CCtemp[counter] = CC[i]
          cctemp[counter] = cc[i]
          At_new = vcat(AT_new, At[:, i])
          if ncopies > 1
            H = Polyhedra.hrep(CC[i], cc[i])
            plyhdrn = Polyhedra.polyhedron(H)
            pts = collect(Polyhedra.points(plyhdrn))
            lns = collect(Polyhedra.lines(plyhdrn))
            rys = collect(Polyhedra.rays(plyhdrn))
            b = QQt * At[:, i]
            B = Pt * At[:, i]
            BTILDES = rcd(ncopies-1, P, b, B)
            for j in 2:ncopies
              VT_new = Vector{Vector{Float64}}(undef, length(pts))
              Btilde = BTILDES[j-1]
              At_tilde = P * Btilde + b
              At_new = vcat(AT_new, At_tilde)
              for k in 1:length(pts)
                pt = convert(Vector{Float64}, pts[k])
                VT_new[k] =
                  convert(Vector{Rational{BigInt}}, pt .- M * (B .- Btilde))
              end
              V = Polyhedra.vrep(VT_new, lns, rys)
              plyhdrn = Polyhedra.polyhedron(V)
              H = plyhdrn.hrep
              CCtemp[counter + j - 1] = H.A
              cctemp[counter + j - 1] = H.b
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
  return ESS
end # fidSampleLR

end # module
