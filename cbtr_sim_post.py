import numpy as np
import pandas as pd
import pickle, glob, os
from statsmodels import api as sm
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

rng = np.random.default_rng(101)

#%% Load the samples
save_path = 'output/github/*'
files = sorted(glob.iglob(save_path), key=os.path.getmtime, reverse=True)
f = 0  # the number of the descended files
file = files[f]

data = pickle.load(open(file, 'rb'))

#%% unpack the data
x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']

burnin = data['meta']['burnin']
n_iter = data['meta']['n_iter'] - burnin
n_rp = data['meta']['n_rp']

B_draws = data['B_draws'][:, burnin:, :, :]  # (n_rp, n_iter, p, q)
mu_draws = data['mu_draws'][:, burnin:]  # (n_rp, n_iter)
sigma_draws = data['sigma_draws'][:, burnin:]  # (n_rp, n_iter)

setting = data['meta']['setting']
rp_type = 'mw' if data['meta']['mode_wise'] else 'tw'


#%% compute in-sample fitting
y_in = np.array([np.einsum('opq, spq -> so', x_train[r], B_draws[r]) + mu_draws[r][..., None] for r in range(n_rp)])

#%% implement reverse logistic regression to estimate the normalising constant
bfs = np.ones(n_rp)
loglik = np.zeros((n_rp, n_iter))

for r in range(n_rp):
    for s in range(n_iter):
        mu_s = y_in[r][s]
        sigma_s = np.sqrt(sigma_draws[r][s])
        loglik[r, s] = norm.logpdf(y_train[0], loc=mu_s, scale=sigma_s).sum()

#%%
ref_model = 0

# estimate the log BF of ML (M_r) vs ML (M_ref), M_r = 0, M_ref = 1
y_model = np.repeat([0, 1], n_iter)
logpos_ref = loglik[ref_model]
for r in range(n_rp):
    if r == ref_model:
        continue
    logpos = loglik[r]
    logpos_dif = logpos_ref - logpos
    x_offset = np.concatenate([logpos_dif, -logpos_dif])

    df = pd.DataFrame({'x': x_offset, 'y': y_model})
    model = sm.GLM(df['y'], np.ones_like(df['x']), family=sm.families.Binomial(), offset=df['x'])
    result = model.fit()
    logbf = result.params.iloc[0]
    bfs[r] = np.exp(logbf)

# compute the normalising constant
ncs = np.array([bfs[r] / bfs.sum() for r in range(n_rp)])

#%% compute the prediction distribution
y_pred_noiseless = np.array([np.einsum('opq, spq -> so',x_test[r], B_draws[r]) + mu_draws[r][..., None] for r in range(n_rp)])

noises = rng.normal(loc=0, scale=sigma_draws[..., None]**.5, size=y_pred_noiseless.shape)

y_pred = y_pred_noiseless + noises

y_pred_mean = y_pred.mean(axis=1)  # (n_rp, n_test)

y_pred_bma = np.einsum('ro, r -> o', y_pred_mean, ncs)  # (n_test,)

#%% compute the rmse
rmse = np.array([mean_squared_error(y_test[0], y_pred_mean[r]) ** .5 for r in range(n_rp)])
rmse_agg = rmse.mean()
rmse_bma = mean_squared_error(y_test[0], y_pred_bma) ** .5

#%% plot the scatter plot
fig_path = 'plots/github/'
os.makedirs(fig_path, exist_ok=True)

plt.figure(constrained_layout=True)
for r in range(n_rp):
    plt.scatter(y_test[0], y_pred_mean[r], alpha=.4, label=f"RP {r}")

# Ensure the axes have the same scale
min_val = min(y_test[0].min(), y_pred_mean.min())
max_val = max(y_test[0].max(), y_pred_mean.max())
margin = .05 * (max_val - min_val)
plt.xlim(min_val - margin, max_val + margin)
plt.ylim(min_val - margin, max_val + margin)
plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio

# Add 45-degree line (y = x)
line_range = [min_val - margin, max_val + margin]
plt.plot(line_range, line_range, 'b--', linewidth=1, label="45° Line")  # Dashed black line

plt.xlabel("True y")
plt.ylabel("Predicted y")
plt.show()
save_fig_to = os.path.join(fig_path, f"scatter_{setting}_{rp_type}.jpeg")


