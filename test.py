# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:50:32 2020

@author: Amin
"""


from scipy.io import loadmat
import ssm


folder = 'C:\\Users\\Amin\\Dropbox\\Worm Tracking\\2019.12.09 - dNMF\\Python'
file = '1007_tail_06'

matfile = loadmat(folder + '\\' + file + '\\' + file + '-matlab.mat')
traces = matfile['C']

# Global parameters
T = traces.shape[1]
K = 4
D_obs = traces.shape[0]
D_latent = 2

rslds_svi = ssm.SLDS(D_obs, K, D_latent,
             transitions="recurrent_only",
             dynamics="diagonal_gaussian",
             emissions="gaussian_orthog",
             single_subspace=True)

rslds_svi.initialize(traces.T)

q_elbos_svi, q_svi = rslds_svi.fit(traces.T, method="bbvi",
                               variational_posterior="tridiag",
                               initialize=False, num_iters=1000)


xhat_svi = q_svi.mean[0]
zhat_svi = rslds_svi.most_likely_states(xhat_svi, y)

# Fit with Laplace EM
rslds_lem = ssm.SLDS(D_obs, K, D_latent,
             transitions="recurrent_only",
             dynamics="diagonal_gaussian",
             emissions="gaussian_orthog",
             single_subspace=True)
rslds_lem.initialize(traces.T)
q_elbos_lem, q_lem = rslds_lem.fit(traces.T, method="laplace_em",
                               variational_posterior="structured_meanfield",
                               initialize=False, num_iters=100, alpha=0.0)
xhat_lem = q_lem.mean_continuous_states[0]
zhat_lem = rslds_lem.most_likely_states(xhat_lem, traces.T)

# Plot some results
plt.figure()
plt.plot(q_elbos_svi, label="SVI")
plt.plot(q_elbos_lem[1:], label="Laplace-EM")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.tight_layout()

plt.figure(figsize=[10,4])
ax1 = plt.subplot(131)
plot_trajectory(z, x, ax=ax1)
plt.title("True")
ax2 = plt.subplot(132)
plot_trajectory(zhat_svi, xhat_svi, ax=ax2)
plt.title("Inferred, SVI")
ax3 = plt.subplot(133)
plot_trajectory(zhat_lem, xhat_lem, ax=ax3)
plt.title("Inferred, Laplace-EM")
plt.tight_layout()

plt.figure(figsize=(6,6))
ax = plt.subplot(111)
lim = abs(xhat_lem).max(axis=0) + 1
plot_most_likely_dynamics(rslds_lem, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
plt.title("Most Likely Dynamics, Laplace-EM")

plt.show()