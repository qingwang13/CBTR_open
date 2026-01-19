#%%
import multiprocessing
import numpy as np
import pickle
import os
import time
from joblib import Parallel, delayed


def rp_mode(tensor, matrices, M, qn, s):
    x_subscripts = ''.join([chr(ord('a') + i) for i in range(M)])
    h_subscripts = ','.join(f'{x_subscripts[i]}{chr(ord("a") + M + i)}' for i in range(M))
    output_subscripts = ''.join([chr(ord('a') + M + i) for i in range(M)])
    einsum_expr = f'{x_subscripts},' + ''.join(h_subscripts) + f'->{output_subscripts}'
    result = np.einsum(einsum_expr, tensor, *matrices)
    # return result * (qn ** (-.5)) * (s ** (M / 2))
    return result * (qn ** (-.5))


def rp_tensor(tensor, H, M, N, qn, s):
    x_subscripts = ''.join([chr(ord('a') + i) for i in range(M)])
    h_subscripts = x_subscripts + ''.join([chr(ord('a') + M + i) for i in range(N)])
    output_subscripts = ''.join([chr(ord('a') + M + i) for i in range(N)])
    einsum_expr = f'{x_subscripts},{h_subscripts}->{output_subscripts}'
    result = np.einsum(einsum_expr, tensor, H)
    return result * (qn ** (-.5)) * (s ** .5)
    # return result * (qn ** (-.5))


def adjust_q(p, qn, preserve_modes):
    """
    Adjusts the dimensions of q to ensure that qn remains constant
    while preserving certain modes.

    Parameters:
    - p: Original tensor shape (tuple)
    - qn: Target compressed size (int)
    - preserve_modes: List of indices to preserve

    Returns:
    - Adjusted q (tuple)
    """
    # Initialize q with the same shape as p
    q = np.array(p, dtype=int)

    # Compute the product of the preserved modes
    preserved_product = np.prod([p[i] for i in preserve_modes])

    # Compute the remaining compression product
    remaining_qn = qn // preserved_product  # Ensure integer division

    # Identify indices of modes that are not preserved
    non_preserved_modes = [i for i in range(len(p)) if i not in preserve_modes]

    # Compute new sizes for non-preserved modes
    scaling_factor = (remaining_qn / np.prod([p[i] for i in non_preserved_modes])) ** (1 / len(non_preserved_modes))

    for i in non_preserved_modes:
        q[i] = max(2, int(round(p[i] * scaling_factor)))  # Ensure at least 2 dimension

    return q


def mode_wise_random_projection(X_train, n_rp, com_rate, s, nobs, seed=821, X_test=None, mode_to_preserve=[],
                                std_entry=False):
    atoms = np.array([1, 0, -1])
    prob = np.array([1 / (2 * s), 1 - 1 / s, 1 / (2 * s)])
    p = np.array(X_train[0].shape)
    q = np.ceil(com_rate * p).astype(int)
    qn = np.prod(q)
    m_p = len(p)
    q_adj = adjust_q(p, qn, mode_to_preserve)
    if not std_entry:
        H_m = []
        for r in range(n_rp):
            local_rng = np.random.default_rng(seed + r)
            H_m.append(
                [np.eye(pi) if i in mode_to_preserve else local_rng.choice(atoms, size=(pi, qi), p=prob) * s ** .5
                 for i, (qi, pi) in enumerate(zip(q_adj, p))])
    else:
        H_m = []
        for r in range(n_rp):
            local_rng = np.random.default_rng(seed + r)
            H_m.append(
                [np.eye(pi) if i in mode_to_preserve else local_rng.standard_normal(size=(pi, qi)) * s ** .5
                 for i, (qi, pi) in enumerate(zip(q_adj, p))])
    start_m = time.time()
    func_m = delayed(rp_mode)
    X_c_m = [[] for _ in range(n_rp)]
    for r in range(n_rp):
        X_c_m[r] = np.array(
            Parallel(n_jobs=6, backend='threading')(func_m(X_train[_], H_m[r], m_p, qn, s) for _ in range(nobs[-1])))
        # if mode_to_preserve:
        #     m_q = len(mode_to_preserve)
        #     X_c_m[r] = xcm * (s ** ((m_p - m_q) / 2))
        # else:
        #     X_c_m[r] = xcm * (s ** (m_p / 2))

    timemode = (time.time() - start_m) / 60
    if X_test is None:
        return X_c_m, timemode
    else:
        X_c_m_test = [[] for _ in range(n_rp)]
        for r in range(n_rp):
            X_c_m_test[r] = np.array(Parallel(n_jobs=6, backend='threading')(
                func_m(X_test[_], H_m[r], m_p, qn, s) for _ in range(len(X_test))))
            # if mode_to_preserve:
            #     m_q = len(mode_to_preserve)
            #     X_c_m_test[r] = xcmt * (s ** ((m_p - m_q) / 2))
            # else:
            #     X_c_m_test[r] = xcmt * (s ** (m_p / 2))
        return X_c_m, X_c_m_test, timemode, H_m


def tensor_wise_random_projection(X_train, n_rp, com_rate, s, nobs, seed=509, X_test=None, mode_to_preserve=[],
                                  std_entry=False):
    atoms = np.array([1, 0, -1])
    prob = np.array([1 / (2 * s), 1 - 1 / s, 1 / (2 * s)])
    p = np.array(X_train[0].shape)
    q = np.ceil(com_rate * p).astype(int)
    qn = np.prod(q)
    q_adj = adjust_q(p, qn, mode_to_preserve)
    m_p = len(p)
    m_q = len(q)
    if not std_entry:
        H_t = []
        for r in range(n_rp):
            local_rng = np.random.default_rng(seed + r)
            H_t.append(local_rng.choice(atoms, size=(*p, *q_adj), p=prob))
    else:
        H_t = []
        for r in range(n_rp):
            local_rng = np.random.default_rng(seed + r)
            H_t.append(local_rng.standard_normal(size=(*p, *q_adj)))
    H_t = np.array(H_t)
    start_t = time.time()
    func_t = delayed(rp_tensor)
    X_c_t = [[] for _ in range(n_rp)]
    for r in range(n_rp):
        X_c_t[r] = np.array(Parallel(n_jobs=6, backend='threading')(
            func_t(X_train[_], H_t[r], m_p, m_q, qn, s) for _ in range(nobs[-1])))
    timetensor = (time.time() - start_t) / 60
    if X_test is None:
        return X_c_t, timetensor
    else:
        X_c_t_test = [[] for _ in range(n_rp)]
        for r in range(n_rp):
            X_c_t_test[r] = np.array(Parallel(n_jobs=6, backend='threading')(
                func_t(X_test[_], H_t[r], m_p, m_q, qn, s) for _ in range(len(X_test))))
        return X_c_t, X_c_t_test, timetensor, H_t


#%%
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    # data = pickle.load(open('sim_data/sim_raw_20_rand_stru_noisy.dat', 'rb'))
    data = pickle.load(open('app/data/macro_data_lag_m4.dat', 'rb'))
    nobs = [350]
    x_train, x_test = data['X'][:nobs[0]], data['X'][nobs[0]:]
    spar, n_rps = 3, 20
    rate = np.array([.6])
    m2p = [0, 2]
    lag = x_train.shape[1]
    entry_std = False
    x_c_m = [[] for _ in range(len(rate))]
    x_c_t = [[] for _ in range(len(rate))]
    x_c_m_test = [[] for _ in range(len(rate))]
    x_c_t_test = [[] for _ in range(len(rate))]
    h_m = [[] for _ in range(len(rate))]
    h_t = [[] for _ in range(len(rate))]
    time_mode = np.zeros(len(rate))
    time_tensor = np.zeros(len(rate))
    for i, com_rate in enumerate(rate):
        x_c_m[i], x_c_m_test[i], time_mode[i], h_m[i] = mode_wise_random_projection(x_train, n_rps, com_rate, spar,
                                                                                    nobs,
                                                                                    X_test=x_test, mode_to_preserve=m2p,
                                                                                    std_entry=entry_std)
        x_c_t[i], x_c_t_test[i], time_tensor[i], h_t[i] = tensor_wise_random_projection(x_train, n_rps, com_rate, spar,
                                                                                        nobs,
                                                                                        X_test=x_test,
                                                                                        mode_to_preserve=m2p,
                                                                                        std_entry=entry_std)

    n_sams = len(nobs)
    y_train = np.array([data['Y'][:nobs[n]] for n in range(n_sams)])
    y_test = np.array([data['Y'][nobs[n]:] for n in range(n_sams)])

    data_rp = {'X_c_m': x_c_m,
               'X_c_t': x_c_t,
               'X_c_m_test': x_c_m_test,
               'X_c_t_test': x_c_t_test,
               'Y_train': y_train,
               'Y_test': y_test,
               'nobs': nobs,
               'spar': spar,
               'atoms': np.array([1, 0, -1]),
               'prob': np.array([1 / (2 * spar), 1 - 1 / spar, 1 / (2 * spar)]),
               'n_rp': n_rps,
               'com_rate': rate,
               'm2p': m2p,
               'n_sim': n_sams,
               'time_mode': time_mode,
               'time_tensor': time_tensor,
               'p': x_train.shape[1],
               'lag': lag
               }
    os.makedirs('app/data/', exist_ok=True)

    pickle.dump(data_rp, open(f'app/data/macro_rp_{rate}_s{spar}_m2p{m2p}_lagm{lag}.dat', 'wb'))
