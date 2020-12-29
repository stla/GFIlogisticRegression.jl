# Generalized fiducial inference for logistic regression

The main function of the 'GFIlogisticRegression' package is `fidSampleLR`. 
It simulates the fiducial distribution of the parameters of a logistic 
regression model.

## Example

To illustrate it, we will consider a logistic dose-response model for inference
on the median lethal dose. The median lethal dose (LD50) is the amount of a 
substance, such as a drug, that is expected to kill half of its users.

The results of LD50 experiments can be modeled using the relation
```math
\textrm{logit}(p_i) = \beta_1(x_i - \mu)
```
where ``p_i`` is the probability of death at the dose administration ``x_i``, 
and ``\mu`` is the median lethal dose, i.e. the dosage at which the probability 
of death is ``0.5``. The ``x_i`` are known while ``\beta_1`` and ``\mu`` are 
fixed effects that are unknown. 

This relation can be written in the form 
```math
\textrm{logit}(p_i) = \beta_0 + \beta_1 x_i
```
with ``\mu = -\beta_0 / \beta_1``.

We will perform the fiducial inference in this model with the following data:

```@example 1
using DataFrames
data = DataFrame(
  x = [
    -2, -2, -2, -2, -2, 
    -1, -1, -1, -1, -1, 
     0,  0,  0,  0,  0,
     1,  1,  1,  1,  1,
     2,  2,  2,  2,  2
  ],
  y = [
    1, 0, 0, 0, 0,
    1, 1, 1, 0, 0,
    1, 1, 0, 0, 0,
    1, 1, 1, 1, 0,
    1, 1, 1, 1, 1
  ]
)
```

Let's go with ``5000`` fiducial simulations: 

```@example 1
using StatsModels, GFIlogisticRegression
fidsamples = fidSampleLR(@formula(y ~ x), data, 5000)
```

Here are the fiducial estimates and ``95\%``-confidence intervals of the 
parameters ``\beta_0`` and ``\beta_1``:

```@example 1
fidSummary(fidsamples)
```

Now let us draw the fiducial ``95\%``-confidence interval about our parameter of 
interest ``\mu``:

```@example 1
fidConfInt("-:x ./ :\"(Intercept)\"", fidsamples, 0.95)
```


## Member functions

```@autodocs
Modules = [GFIlogisticRegression]
Order   = [:type, :function]
```