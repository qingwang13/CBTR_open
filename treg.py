import numpy as np
from aux_functions import fullcon_w, fullcon_w_l2, fullcon_tau, fullcon_zeta, fullcon_lambda, fullcon_gamma, fullcon_sigma, fullcom_mu, w_gamma, psi_update, res_update, get_psi, get_coef, get_res
import time


def treg_parafac(X, y, hyper, factor, n_iter, b_true=None):
    # initializing
    nobs = len(y)
    p = X[0].shape
    D = factor
    M = len(p)
    # hyperparameters
    alpha, atau, btau, alam, blam, a_sig, b_sig, mu_sig = hyper
    ptau = atau + D * np.sum(p) / 2
    a_lamb = np.array([p[m] + alam for m in range(M)])
    aw = 3
    bw = 10

    # allocation
    gamma_draws = [[np.zeros((D, p[m])) + .1 for m in range(M)] for _ in range(n_iter)]
    tau_draws = np.zeros(n_iter) + 1
    zeta_draws = np.zeros((n_iter, D)) + 1
    w_draws = [[np.zeros((D, p[m])) + 1 for m in range(M)] for _ in range(n_iter)]
    lambda_draws = np.zeros((n_iter, M, D)) + 1
    mu_draws = np.zeros(n_iter) + 2
    sigma_draws = np.zeros(n_iter) + 5
    B_draws = np.zeros((n_iter, *p))

    # Initialization
    # Bd = np.zeros((D, *p))
    # rdt = np.zeros((nobs, D))
    # psi = [np.zeros((nobs, D, p[m])) + .1 for m in range(M)]
    # psi_out = [np.zeros((nobs, D, p[m], p[m])) + .1 for m in range(M)]
    B_draws[0], Bd = get_coef(gamma_draws[0], M, D)
    psi, psi_out = get_psi(X, gamma_draws[0])
    y_til, rdt = get_res(X, y, gamma_draws[0], mu_draws[0], M, D, nobs)
    gamma_w = w_gamma(gamma_draws[0], w_draws[0], M, D)

    rej = 0
    count = 0
    st = time.time()

    for i in range(1, n_iter):
        if i % 100 == 0:
            print(f'Iteration {i} of {n_iter}')

        # update zeta, tau, lambda, w, gamma, sigma
        tau_draws[i] = fullcon_tau(ptau, btau, zeta_draws[i-1], gamma_w)
        zeta_draws[i] = fullcon_zeta(gamma_w, tau_draws[i], alpha, D, p)
        for d in range(D):
            for m in range(M):
                count += 1
                gamma_cand = fullcon_gamma(psi[m][:, d], psi_out[m][:, d], y_til[:, d], tau_draws[i], zeta_draws[i][d], w_draws[i-1][m][d], sigma_draws[i-1], cho=False)
                if max(np.abs(gamma_cand)) < 5 or i <= 100:
                    gamma_draws[i][m][d] = gamma_cand
                else:
                    gamma_draws[i][m][d] = gamma_draws[i-1][m][d]
                    rej += 1
                w_draws[i][m][d] = np.array([fullcon_w(gamma_draws[i][m][d, pm], lambda_draws[i-1][m, d], zeta_draws[i][d], tau_draws[i]) for pm in range(p[m])])
                # w_draws[i][m][d] = np.array([fullcon_w_l2(aw, bw, gamma_draws[i][m][d, pm], tau_draws[i]) for pm in range(p[m])])
                lambda_draws[i][m][d] = fullcon_lambda(a_lamb[m], blam, gamma_draws[i][m][d], tau_draws[i], zeta_draws[i][d], w_draws[i][m][d])
                psi, psi_out = psi_update(psi, psi_out, gamma_draws[i], X, M, D, m, nobs)
                # print(f'gamma: {np.max(gamma_draws[i][m][d])}')
            y_til, rdt = res_update(X, y, y_til, gamma_draws[i], mu_draws[i-1], rdt, M, D, d, nobs)

        # updating process
        B_draws[i], Bd = get_coef(gamma_draws[i], M, D)
        gamma_w = w_gamma(gamma_draws[i], w_draws[i], M, D)

        sigma_draws[i] = fullcon_sigma(a_sig, b_sig, y, X, B_draws[i], mu_draws[i - 1])
        # sigma_draws[i] = fullcon_sigma(a_sig, b_sig, y, X, b_true, mu_draws[i - 1])
        mu_draws[i] = fullcom_mu(y, X, B_draws[i], sigma_draws[i], mu_sig)
        # mu_draws[i] = fullcom_mu(y, X, b_true, sigma_draws[i], mu_sig)

    end = time.time()
    ex_time = (end - st) / 60
    counts = np.array([count, rej])

    return B_draws, gamma_draws, tau_draws, zeta_draws, w_draws, lambda_draws, mu_draws, sigma_draws, counts, ex_time
