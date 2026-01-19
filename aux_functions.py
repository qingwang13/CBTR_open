import numpy as np
from gigrnd import gigrnd
from scipy.linalg import cho_factor, cho_solve, solve
from scipy.stats import multivariate_normal

rng_pos = np.random.default_rng()


def fullcon_gamma(psi, psi_out, ytil, tau, zeta, w, sigma, cho=False, epsilon=1e-3):
    psi_out_sum = psi_out.sum(axis=0)
    ytil = ytil.reshape(-1, 1)  # reshape ytil into a column vector
    psi_ytil_sum = (psi * ytil).sum(axis=0)
    W_inv = np.diag(1 / w)
    # W_inv = np.linalg.inv(W)
    cov_inv = psi_out_sum / sigma + W_inv / (tau * zeta) + epsilon * np.eye(psi.shape[1])
    # print(cov_inv)
    if cho:
        # compute mean using Cholesky decomposition
        L, lower = cho_factor(cov_inv)
        mean = cho_solve((L, lower), psi_ytil_sum / sigma)

        z = rng_pos.normal(size=psi.shape[1])
        return mean + cho_solve((L, lower), z)
    else:
        # cov = np.linalg.inv(cov_inv)
        try:
            cov = solve(cov_inv, np.eye(psi.shape[1]))
        except np.linalg.LinAlgError:
            print("Solve failed, falling back to pseudo-inverse.")
            cov = np.linalg.pinv(cov_inv)
        # cov = solve(cov_inv, np.eye(psi.shape[1]))
        mean = cov @ (psi_ytil_sum / sigma)
        gamma = rng_pos.multivariate_normal(mean, cov)
        return gamma


def fullcon_w(gamma, lamb, tau, zeta):
    a_w = lamb ** 2
    b_w = gamma ** 2 / (tau * zeta)
    if b_w == 0:
        return rng_pos.gamma(.5, (a_w * .5) ** -1)
    else:
        return gigrnd(.5, a_w, b_w ** -1)


def fullcon_w_l2(aw, bw, gamma, tau):
    a_w = aw + .5
    b_w = bw + gamma ** 2 / (2 * tau)
    return 1 / rng_pos.gamma(a_w, b_w ** -1)


def fullcon_zeta(gamma_w, tau, alpha, D, p):
    a_zeta = np.sum(p)/2 - alpha
    b_zeta = gamma_w.sum(axis=1) / (2 * tau)
    zeta = 1 / rng_pos.gamma(a_zeta, b_zeta ** -1)
    return zeta / np.sum(zeta)


def fullcon_tau(p_tau, b_tau, zeta, gamma_w):
    # a_tau_star = 2 * b_tau
    # b_tau_star = np.sum(gamma_w.sum(axis=1) / zeta)
    # return gigrnd(p_tau, a_tau_star, b_tau_star)
    atau_star = p_tau
    btau_star = np.sum(gamma_w.sum(axis=1) / (2 * zeta) + b_tau)
    return 1 / rng_pos.gamma(atau_star, btau_star ** -1)


def fullcon_lambda(a_lamb, b_lamb, gamma, tau, zeta, w):
    # b_lamb = np.sum(abs(gamma)) / (tau * zeta) ** .5 + b_lamb
    b_lamb = b_lamb + np.sum(w) / 2
    return rng_pos.gamma(a_lamb, b_lamb ** -1) ** .5


def fullcon_sigma(a_sigma, b_sigma, y, x, b, mu):
    a_sigma_star = a_sigma + len(y) / 2
    b_sigma_star = b_sigma + np.sum((y - x.reshape(x.shape[0], -1) @ b.flatten() - mu) ** 2) / 2
    return 1 / rng_pos.gamma(a_sigma_star, b_sigma_star ** -1)


def fullcom_mu(y, x, B, sig_y, sig_mu):
    var_mu = (len(y) / sig_y + 1 / sig_mu) ** -1
    mean_mu = var_mu * np.sum(y - x.reshape(x.shape[0], -1) @ B.flatten()) / sig_y
    return rng_pos.normal(mean_mu, var_mu ** .5)


def get_psi(X, margins):
    M = len(margins)  # number of modes
    D = margins[0].shape[0]  # number of factors
    nobs = len(X)  # number of observations

    tensor_subscripts = ''.join([chr(ord('a') + i) for i in range(M)])  # einsum subscripts for tensor

    psi = [np.zeros((nobs, D, margins[m].shape[1])) for m in range(M)]
    psi_out = [np.zeros((nobs, D, margins[m].shape[1], margins[m].shape[1])) for m in range(M)]

    for m in range(M):
        margin_subscripts = ','.join([chr(ord('a') + i) for i in range(M) if i != m])  # einsum subscripts for margins
        psi_subscripts = chr(ord('a') + m)  # einsum subscripts for psi
        einsum_str = f'{tensor_subscripts},{margin_subscripts}->{psi_subscripts}'  # construct the einsum string
        # print(einsum_str)
        for t in range(nobs):
            for d in range(D):
                args = [X[t]] + [margins[i][d] for i in range(M) if i != m]  # arguments for einsum
                psi[m][t, d] = np.einsum(einsum_str, *args)
                psi_out[m][t, d] = np.outer(psi[m][t, d], psi[m][t, d])

    return psi, psi_out


def psi_update(psi, psi_out, gamma, X, M, D, current_mode, nobs):
    tensor_subscripts = ''.join([chr(ord('a') + i) for i in range(M)])  # einsum subscripts for tensor
    for m in range(M):
        if m != current_mode:
            margin_subscripts = ','.join([chr(ord('a') + i) for i in range(M) if i != m])  # einsum subscripts for margins
            psi_subscripts = chr(ord('a') + m)  # einsum subscripts for psi
            einsum_str = f'{tensor_subscripts},{margin_subscripts}->{psi_subscripts}'  # construct the einsum string
            for t in range(nobs):
                for d in range(D):
                    args = [X[t]] + [gamma[i][d] for i in range(M) if i != m]  # arguments for einsum
                    psi[m][t, d] = np.einsum(einsum_str, *args)
                    psi_out[m][t, d] = np.outer(psi[m][t, d], psi[m][t, d])

    return psi, psi_out


# def psi_update(psi, psi_out, X, M, D, nobs, gamma):
#     for m in range(M):
#         mask = np.arange(M) != m
#         remaining_modes = np.arange(M)[mask]
#         for t in range(nobs):
#             for d in range(D):
#                 psi_int = X[t].copy()
#                 for i, rm in enumerate(remaining_modes):
#                     mode_to_contract = remaining_modes[i] - i
#                     psi_int = np.tensordot(psi_int, gamma[rm][d], axes=(mode_to_contract, 0))
#                 psi[m][t, d] = psi_int
#                 psi_out[m][t, d] = np.outer(psi[m][t, d], psi[m][t, d])
#     return psi, psi_out


def get_res(X, y, gamma, mu, M, D, nobs):
    rdt = np.zeros((nobs, D))

    # Construct the einsum subscript string dynamically
    tensor_subscripts = ''.join([chr(ord('a') + m) for m in range(M)])  # einsum subscripts for tensor ('abcd...')
    margin_subscripts = ','.join([f'd{chr(ord("a") + m)}' for m in range(M)])  # einsum subscripts for margins ('da,db,dc...')
    einsum_str = f'{tensor_subscripts},{margin_subscripts}->'  # construct the einsum string ('abcd..., da,db,dc...->')

    for t in range(nobs):
        for d in range(D):
            rdt[t, d] = np.einsum(einsum_str, X[t], *[np.array([gamma[m][_] for _ in range(D) if _ != d]) for m in range(M)])
    ytil = np.array([[y[t] - rdt[t, d] - mu for d in range(D)] for t in range(nobs)])

    return ytil, rdt


def res_update(X, y, ytil, gamma, mu, rdt, M, D, current_factor, nobs):
    # Construct the einsum subscript string dynamically
    tensor_subscripts = ''.join([chr(ord('a') + m) for m in range(M)])  # einsum subscripts for tensor ('abcd...')
    margin_subscripts = ','.join(
        [f'd{chr(ord("a") + m)}' for m in range(M)])  # einsum subscripts for margins ('da,db,dc...')
    einsum_str = f'{tensor_subscripts},{margin_subscripts}->'  # construct the einsum string ('abcd..., da,db,dc...->')

    for t in range(nobs):
        for d in range(D):
            if d != current_factor:
                rdt[t, d] = np.einsum(einsum_str, X[t], *[np.array([gamma[m][_] for _ in range(D) if _ != d]) for m in range(M)])
                ytil[t, d] = y[t] - rdt[t, d] - mu
    # ytil = np.array([[y[t] - rdt[t, d] - mu for d in range(D)] for t in range(nobs)])

    return ytil, rdt


# def res_update(X, y, Bd, gamma, mu, rdt, M, D, nobs):
#     for d in range(D):
#         Bd_int = gamma[0][d].copy()
#         for m in range(M - 1):
#             Bd_int = np.tensordot(Bd_int, gamma[m + 1][d], axes=0)
#         Bd[d] = Bd_int
#     for t in range(nobs):
#         for d in range(D):
#             mask = np.arange(D) != d
#             rdt[t, d] = np.vdot(np.sum(Bd[mask], axis=0), X[t])
#     ytil = np.array([[y[t] - rdt[t, d] - mu for d in range(D)] for t in range(nobs)])
#     return ytil, rdt, Bd


def w_gamma(gamma, w, M, D):
    return np.array(
        [[gamma[m][d] @ np.diag(1 / w[m][d]) @ gamma[m][d] for m in range(M)] for d in range(D)])


def outer_prod(vectors):
    # Create the einsum subscript string dynamically
    # For N vectors, the input subscripts are 'i,j,k,...' and the output is 'ijk...'
    input_subscripts = ','.join([chr(ord('i') + i) for i in range(len(vectors))])
    output_subscript = ''.join([chr(ord('i') + i) for i in range(len(vectors))])
    einsum_str = f'{input_subscripts}->{output_subscript}'

    # Compute the outer product using einsum
    return np.einsum(einsum_str, *vectors)


def get_coef(gamma, M, D):
    gamma_d = [[gamma[m][d] for m in range(M)] for d in range(D)]
    Bd = np.array([outer_prod(gamma_d[d]) for d in range(D)])
    B = Bd.sum(axis=0)
    return B, Bd

