import numpy as np
from tensorly.datasets.synthetic import gen_image
from treg import treg_parafac
import pickle
from datetime import datetime
import os
from randproj_app import mode_wise_random_projection, tensor_wise_random_projection
from multiprocessing import Pool, freeze_support

# ----------------------------
# Worker-global storage (spawn-safe)
# ----------------------------
_x_train = None
_y_train0 = None
_hyper = None
_D = None
_n_iter = None
_nobs_train = None
_base_seed = None


def _init_worker(x_train, y_train0, hyper, D, n_iter, nobs_train):
    """Initializer runs once per worker process."""
    global _x_train, _y_train0, _hyper, _D, _n_iter, _nobs_train
    _x_train = x_train
    _y_train0 = y_train0
    _hyper = hyper
    _D = D
    _n_iter = n_iter
    _nobs_train = nobs_train


def _treg_parallel(r):
    result = treg_parafac(_x_train[r], _y_train0, _hyper, _D, _n_iter)
    print(f'num_rp: {r}')

    return result


# ----------------------------
# Synthetic coefficient patterns (your code)
# ----------------------------
def make_coefficients(rng, mode_1=20, mode_2=20, prob=0.25):

    b_1 = np.identity(mode_1)
    b_2 = gen_image(region='circle', image_height=mode_1, image_width=mode_2)
    b_3 = gen_image(region='swiss', image_height=mode_1, image_width=mode_2)
    b_4 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    b_5 = np.zeros((mode_1, mode_2))
    row_ind = np.linspace(1, 18, 3, dtype=int)
    b_5[row_ind, :] = 1
    b_6 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    b_7 = rng.choice([0, 1], size=(mode_1, mode_2), p=[prob, 1 - prob])

    return np.array([b_1, b_2, b_3, b_4, b_5, b_6, b_7])


# ----------------------------
# Main experiment
# ----------------------------
def main():
    # Reproducibility seed for *data generation*
    base_seed = 110
    rng = np.random.default_rng(base_seed)

    # ---- settings (edit here) ----
    mode_1, mode_2 = 20, 20
    p = np.array([mode_1, mode_2])
    prob = 0.25

    setting = [1]  # choose the true coefficients pattern index
    true_mu = 0
    true_sigma = 1

    nobs = 1500
    nobs_train = 1000
    nobs_test = nobs - nobs_train

    #%% random projection
    rp = True
    mode_wise = True
    entry_std = False
    n_rp = 10
    com_rate = .6
    s = 3
    m2p = [0]  # mode to preserve
    rp_type = 'mw' if mode_wise else 'tw'

    # Model / sampler settings
    D = 3
    n_iter = 2000
    burnin = int(n_iter * 0.2)

    # Hyperparameters (your code)
    alpha = D ** (-2) * np.ones(D)
    atau, btau = 5.5, 9
    alam = 20
    blam = alam ** (1 / (2 * D))
    a_sig, b_sig = 3, 5
    mu_sig = 1
    hyper = [alpha, atau, btau, alam, blam, a_sig, b_sig, mu_sig]

    # ---- 1) generate coefficients and data ----
    b_coef = make_coefficients(rng, mode_1=mode_1, mode_2=mode_2, prob=prob)
    b_true = b_coef[setting]  # shape: (len(setting), mode_1, mode_2)

    pattern = ['diagnol', 'circle', 'swiss', 'L-shape', 'stripes', 'blocks', 'random']

    X = np.array([rng.standard_normal(size=p) for _ in range(nobs)])
    X_train, X_test = X[:nobs_train], X[nobs_train:]

    # y: shape (len(b_true), nobs)
    y = np.array(
        [[X[i].flatten() @ b_true[j].flatten() for i in range(nobs)] for j in range(len(b_true))]
    ) + rng.normal(true_mu, np.sqrt(true_sigma), nobs)

    y_train = y[:, :nobs_train]
    y_test = y[:, nobs_train:]

    # ---- 2) random projection ----
    if rp:
        n_obs = [nobs_train]
        if mode_wise:
            x_train, x_test = mode_wise_random_projection(X_train, n_rp, com_rate, s, n_obs, X_test=X_test,
                                                          mode_to_preserve=m2p, std_entry=entry_std)[:2]
        else:
            x_train, x_test = tensor_wise_random_projection(X, n_rp, com_rate, s, n_obs, X_test=X_test,
                                                            mode_to_preserve=m2p, std_entry=entry_std)[:2]
    else:
        x_train = X[:nobs_train]
        x_test = X[nobs_train:]

    # ---- 3) fit in parallel over RP replicates ----
    with Pool(processes=min(n_rp, os.cpu_count() or 1),
              initializer=_init_worker,
              initargs=(x_train, y_train[0], hyper, D, n_iter, nobs_train), ) as pool:
        results = pool.map(
            _treg_parallel, range(n_rp)
        )

    B_draws = np.array([results[r][0] for r in range(n_rp)])
    sigma_draws = np.array([results[r][7] for r in range(n_rp)])
    mu_draws = np.array([results[r][6] for r in range(n_rp)])
    ex_time = np.array([results[r][9] for r in range(n_rp)])

    # ---- 4) unpack and save ----
    output = {'B_draws': B_draws,
              'mu_draws': mu_draws,
              'sigma_draws': sigma_draws,
              'comp_rate': com_rate,
              'psi': s,
              'm2p': m2p,
              'rank': D,
              'ex_time': ex_time,
              'x_train': x_train,
              'x_test': x_test,
              'y_train': y_train,
              'y_test': y_test,
              "meta": {
                  "timestamp": datetime.now().strftime("%m%d%H%M"),
                  "seed": base_seed,
                  "setting": pattern[setting[0]],
                  "true_mu": true_mu,
                  "true_sigma": true_sigma,
                  "nobs": nobs,
                  "nobs_train": nobs_train,
                  "nobs_test": nobs_test,
                  "rp": rp,
                  "mode_wise": mode_wise,
                  "entry_std": entry_std,
                  "n_rp": n_rp,
                  "com_rate": com_rate,
                  "psi": s,
                  "m2p": m2p,
                  "rank": D,
                  "n_iter": n_iter,
                  "burnin": burnin,
                  "hyper": hyper,
              },
              }

    out_dir = 'output/github/'
    os.makedirs(out_dir, exist_ok=True)

    fname = f'cbtr_{pattern[setting[0]]}_{rp_type}_m2p{m2p}.pkl'

    out_path = os.path.join(out_dir, fname)
    with open(out_path, 'wb') as f:
        pickle.dump(output, f)

    print(f"Saved output to {out_path}")
    return out_path


if __name__ == '__main__':
    freeze_support()
    main()
