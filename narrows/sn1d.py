#!/usr/bin/env python

import sys
import numpy as np
import collections

LARGE_DOUBLE = 1e+50
SMALL_DOUBLE = 1e-50

__version__ = '0.0.1'


def converged(new_I0, old_I0, epsilon, i, verbose):
    done = False
    if i < 3:
        residual = None
    else:
        residual = np.max(np.abs(1 - new_I0/old_I0))
        done = residual < epsilon

    if verbose:
        if residual:
            print('At end of i=%d residual=%.6f' % (i, residual))
        else:
            print('At end of i=%d residual=N/A' % i)

    if verbose:
        if done:
            print('Converged!')

    return done


def dump(mu, w, Q, Qhat, I0, I1, psi, zone_edges):
    print('mu ', mu)
    print('w ', w)
    print('Q ', Q)
    print('Qhat ', Qhat)
    print('I0 ', I0)
    print('I1 ', I1)
    print('psi ', psi)
    print('zone_edges ', zone_edges)


def make_src(source_description, zone_edges, num_ordinates, source_magnitude):
    # The spatial index changes the fastest, meaning Q[m][j]
    # is the source component in direction m at zone j
    Q = []

    for i in range(len(zone_edges)):
        if zone_edges[i] <= source_description[1]:
            Q.append(source_magnitude)
        else:
            Q.append(0.)

    '''
    print(np.array(uniform_source_description))
    print(zone_edges)
    print(np.array(Q))
    for edge, q in zip(zone_edges, np.array(Q)):
        print('%.2f %.2f' % (edge, q))
    '''
    return np.array([Q] * num_ordinates)


def main(args):

    # J
    zone_edges = np.linspace(0, args.zstop, args.num_sn_zones + 1)

    # NxJ
    Q = make_src(args.uniform_source_extent, zone_edges, args.num_ordinates,
                 args.source_magnitude)

    mu, weight = np.polynomial.legendre.leggauss(args.num_ordinates)
    # NxJ
    mu = np.array([mu] * len(zone_edges))
    mu = mu.transpose()

    # J, cell-averaged scalar flux
    I0 = np.array([0.] * len(zone_edges))
    # old_I0 = np.array([LARGE_DOUBLE] * len(zone_edges))
    old_I0 = I0.copy()

    # J, cell-averaged current
    I1 = np.array([0.] * len(zone_edges))

    # NxJ
    psi = [0.] * len(zone_edges)
    if args.point_source_location is not None:
        if args.point_source_location == 0:
            psi[0] = 1.
        else:
            print('%s cannot handle point_source_location %f != 0' %
                  (__name__, args.point_source_location), file=sys.stderr)
            sys.exit(-1)

    psi = np.array([psi] * args.num_ordinates)

    # dump(mu, weight, Q, None, I0, I1, psi, zone_edges)
    i = 1
    I0s = [I0]
    I1s = [I1]
    while not converged(I0, old_I0, args.epsilon_sn, i, args.verbose) \
            and i < args.max_num_iter:
        old_I0 = I0.copy()
        i += 1

        Qhat = update_Qhat(args.sigma_s0, I0, mu, args.sigma_s1, I1, Q)
        psi = sweep(psi, mu, zone_edges, Qhat, args.sigma_t)
        I0 = update_I0(weight, psi, I0)
        I1 = update_I1(weight, psi, I1, mu)

        I0s.append(I0)
        I1s.append(I1)
        # dump(mu, weight, Q, Qhat, I0, I1, psi, zone_edges)

    if i == args.max_num_iter:
        print('Warning: Max num iterations: %d achieved before convergence.'
              % args.max_num_iter)

    Result = collections.namedtuple('Result',
                                    'zone_edges psi I0 I1 weight mu i args')
    return Result(zone_edges, psi, I0, I1, weight, mu, i, args)


def update_Qhat(sigma_s0, I0, mu, sigma_s1, I1, Q):
    return (sigma_s0 * I0) + (3 * mu * sigma_s1 * I1) + Q


def update_I0(weight, psi, I0):
    new_I0 = []
    for j in range(len(I0)):
        weighted_sum = 0
        for m in range(len(psi)):
            weighted_sum += weight[m] * psi[m][j]
        new_I0.append(weighted_sum)
    return np.array(new_I0)


def update_I1(weight, psi, I1, mu):
    new_I1 = []
    for j in range(len(I1)):
        weighted_sum = 0
        for m in range(len(psi)):
            weighted_sum += weight[m] * psi[m][j] * mu[m][j]
        new_I1.append(weighted_sum)
    return np.array(new_I1)


def sweep(psi, mu, zone_edges, Qhat, sigma_t, alpha=0):

    # right
    for m in range(int(len(mu) / 2), len(mu)):
        for j in range(1, len(zone_edges)):
            width = zone_edges[j] - zone_edges[j-1]
            numer = ((2 * mu[m][j] - sigma_t * width * (1 - alpha))
                     * psi[m][j-1] + width * Qhat[m][j])
            denom = 2 * mu[m][j] + sigma_t * width * (1 + alpha)
            psi[m][j] = numer / denom

    # left
    for m in range(int(len(mu) / 2)):
        for j in range(len(zone_edges) - 2, -1, -1):
            width = zone_edges[j+1] - zone_edges[j]
            numer = ((-2 * mu[m][j] - sigma_t * width * (1 + alpha))
                     * psi[m][j+1] + width * Qhat[m][j])
            denom = -2 * mu[m][j] + sigma_t * width * (1 - alpha)
            psi[m][j] = numer / denom

    return psi
