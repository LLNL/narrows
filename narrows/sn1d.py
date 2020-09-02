import collections
import numpy as np

from .writer import write

LARGE_DOUBLE = 1e+50
SMALL_DOUBLE = 1e-50


def _converged(new_I0, old_I0, epsilon, i):
    done = False
    if i < 3:
        residual = None
    else:
        residual = np.max(np.abs(1 - new_I0/old_I0))
        done = residual < epsilon

    if residual:
        write('verbose', f'At end of i={i} residual={residual:.6f}')
    else:
        write('verbose', f'At end of i={i} residual=N/A')

    if done:
        write('verbose', 'Converged!')

    return done


def _dump(mu, w, Q, Qhat, I0, I1, psi, edges):
    write('verbose', f'mu {mu}')
    write('verbose', f'w {w}')
    write('verbose', f'Q {Q}')
    write('verbose', f'Qhat {Qhat}')
    write('verbose', f'I0 {I0}')
    write('verbose', f'I1 {I1}')
    write('verbose', f'psi {psi}')
    write('verbose', f'edges {edges}')


def _update_Qhat(sigma_s0, I0, mu, sigma_s1, I1, Q):
    return (sigma_s0 * I0) + (3 * mu * sigma_s1 * I1) + Q


def _update_I0(weight, psi, I0):
    new_I0 = []
    for j in range(len(I0)):
        weighted_sum = 0
        for m in range(len(psi)):
            weighted_sum += weight[m] * psi[m][j]
        new_I0.append(weighted_sum)
    return np.array(new_I0)


def _update_I1(weight, psi, I1, mu):
    new_I1 = []
    for j in range(len(I1)):
        weighted_sum = 0
        for m in range(len(psi)):
            weighted_sum += weight[m] * psi[m][j] * mu[m][j]
        new_I1.append(weighted_sum)
    return np.array(new_I1)


def _sweep(psi, mu, edges, Qhat, sigma_t, alpha=0):

    # right
    for m in range(int(len(mu) / 2), len(mu)):
        for j in range(1, len(edges)):
            width = edges[j] - edges[j-1]
            numer = ((2 * mu[m][j] - sigma_t[j] * width * (1 - alpha))
                     * psi[m][j-1] + width * Qhat[m][j])
            denom = 2 * mu[m][j] + sigma_t[j] * width * (1 + alpha)
            psi[m][j] = numer / denom

    # left
    for m in range(int(len(mu) / 2)):
        for j in range(len(edges) - 2, -1, -1):
            width = edges[j+1] - edges[j]
            numer = ((-2 * mu[m][j] - sigma_t[j+1] * width * (1 + alpha))
                     * psi[m][j+1] + width * Qhat[m][j])
            denom = -2 * mu[m][j] + sigma_t[j+1] * width * (1 - alpha)
            psi[m][j] = numer / denom

    return psi


def main(edges, sigma_t, sigma_s0, sigma_s1, source, ordinates, sn_epsilon,
         max_num_src_iter):

    # NxJ
    Q = np.repeat(source[np.newaxis, :], ordinates, axis=0)

    mu, weight = np.polynomial.legendre.leggauss(ordinates)
    # NxJ
    mu = np.array([mu] * len(edges))
    mu = mu.transpose()

    # J, cell-averaged scalar flux
    I0 = np.array([0.] * len(edges))
    # old_I0 = np.array([LARGE_DOUBLE] * len(edges))
    old_I0 = I0.copy()

    # J, cell-averaged current
    I1 = np.array([0.] * len(edges))

    # NxJ
    psi = [0.] * len(edges)
    psi = np.array([psi] * ordinates)

    _dump(mu, weight, Q, None, I0, I1, psi, edges)
    i = 1
    I0s = [I0]
    I1s = [I1]
    while not _converged(I0, old_I0, sn_epsilon, i) and i < max_num_src_iter:
        old_I0 = I0.copy()
        i += 1

        Qhat = _update_Qhat(sigma_s0, I0, mu, sigma_s1, I1, Q)
        psi = _sweep(psi, mu, edges, Qhat, sigma_t)
        I0 = _update_I0(weight, psi, I0)
        I1 = _update_I1(weight, psi, I1, mu)

        I0s.append(I0)
        I1s.append(I1)
        _dump(mu, weight, Q, Qhat, I0, I1, psi, edges)

    if i == max_num_src_iter:
        write('terse', f'Warning: Max num iterations: {max_num_src_iter} '
                       f'achieved before convergence.')

    Result = collections.namedtuple('Result',
                                    'edges psi I0 I1 weight mu i')
    return Result(edges, psi, I0, I1, weight, mu, i)
