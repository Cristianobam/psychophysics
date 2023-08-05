#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import scipy.stats as ss

import jax
jax.config.update('jax_platform_name', 'cpu')

from pymc.sampling_jax import sample_numpyro_nuts
from scipy.optimize import curve_fit

#%%
df = pd.read_csv(f'data/P003.csv', sep='\t', index_col=False)
df['Delta'] = np.abs(df['Current_Time'] - df['Previous_Time'])
blocks = {}
for iBlock in df.groupby('Previous_Time'):
    blocks[iBlock[0]] = iBlock[1]

resp1 = {}
resp2 = {}
for iResp in blocks[357.1429].groupby('Delta'):
    resp1[round(iResp[0])] = iResp[1]['Accuracy'].sum()/len(iResp[1]['Accuracy'])
    resp2[round(iResp[0])] = len(iResp[1]['Accuracy'])

# %%
df = pd.read_csv('data/P005.csv')
plt.rcParams['font.size'] = 8
X = df[['staircase.intensity', 'staircase.response']].dropna().values[:,0]
Y = df[['staircase.intensity', 'staircase.response']].dropna().values[:,1]

mosaic = '''A'''

fig, axs = plt.subplot_mosaic(mosaic, figsize=(15,3))
axs = [axs[k] for k in axs.keys()]

axs[0].plot(np.arange(len(X)), X, color='k')
axs[0].scatter(np.arange(len(X)), X, edgecolor='k', facecolor=np.where(Y, 'k', 'w'), s=10, zorder=2)
axs[0].grid(linestyle='dashed', alpha=.3)

idx = X.argsort()
splits = np.split(Y[idx], np.unique(X[idx], return_index=True)[1][1:])

X = np.log10(np.unique(X[idx]))
Y = np.array(list(map(lambda x: sum(x), splits)))
N = np.array(list(map(lambda x: len(x), splits)))

#%%
with pm.Model() as model:
    x_hat = pm.MutableData('x_hat', X)
    y_hat = pm.MutableData('y_hat', Y)
    n = pm.MutableData('N', N)
   
    mu = pm.Laplace('mu', 0, 0.1, shape=(2,))
    sigma = pm.Uniform('sigma', 0, 1000, shape=(2,))
    gamma = pm.Uniform('gamma', 0, 1)
    lmbda = pm.Uniform('lmbda', 0, 1)
    alpha = pm.Normal('alpha', mu[0], sigma[0], shape=(1,))
    beta = pm.Normal('beta', mu[1], sigma[1], shape=(1,))

    theta = pm.Deterministic('theta', gamma + (1 - gamma - lmbda) * pm.math.exp(pm.Normal.logcdf(alpha * x_hat + beta, 0, 1)))

    likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=y_hat)

with model:
    idata = pm.sample(30000, tune=5000, target_accept=.99)

#%%
PS = lambda x, a, b, g, l: g + (1 - g - l) * ss.norm().cdf(x*a + b)

popt, pcov = curve_fit(PS, X, Y/N)

t = np.linspace(X[0], X[-1], 100)
plt.plot(t, PS(t, *popt))
plt.scatter(X, Y/N)

alpha = idata.posterior.alpha.mean(['draw', 'chain']).values
beta = idata.posterior.beta.mean(['draw', 'chain']).values
gamma = idata.posterior.gamma.mean(['draw', 'chain']).values
lmbda = idata.posterior.lmbda.mean(['draw', 'chain']).values

plt.plot(t, PS(t, alpha, beta, gamma, lmbda))

#%%
theta = np.linspace(0,1,100)
for iX, iN, iY in zip(X, N, Y):
    plt.fill_betweenx(theta, iX, iX-ss.binom.pmf(k=iY, n=iN, p=theta)*0.1)
plt.plot(t, PS(t, alpha, beta))
plt.plot(t, PS(t, *popt))