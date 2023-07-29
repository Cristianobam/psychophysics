#%%
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

#%%
with pm.Model() as model:
    X = pm.MutableData('x')
    Y = pm.MutableData('y')
    N = pm.MutableData('N')
    mu = pm.Normal('mu_alpha', 0, 0.001, shape=(2,))
    sigma = pm.HalfNormal('sigma_alpha', 1, shape=(2,))

    alpha = pm.Normal('alpha', mu[0], sigma[0])
    beta = pm.Normal('beta', mu[1], sigma[1])

    theta = pm.invprobit(alpha + beta * X)

    trace = pm.Binomial('likelihood', n=N, p=theta, obs=Y)