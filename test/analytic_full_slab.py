#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.special as sp
import sys

MY_DIR = os.path.dirname(__file__)
sys.path.append(f'{MY_DIR}/..')
from narrows import parse_input, create_mesh  # noqa: E402

sys.path.append(f'{MY_DIR}')
from utility import show_or_save  # noqa: E402

plt.style.use(f'{MY_DIR}/style.mplstyle')


def solution(z, src_mag, sigma_a, zstop):
    EPS = 1e-25

    return ((src_mag / sigma_a) -
            (src_mag / (2 * sigma_a)) * (np.exp(-sigma_a * z) +
                                         np.exp(sigma_a * (z - zstop))) +
            (src_mag / 2) * (z * sp.exp1(sigma_a * z + EPS) +
                             (zstop - z) * sp.exp1(sigma_a * (zstop - z)
                                                   + EPS)))


def plot_analytic_flux(show, problem):
    z, src_mag, sigma_a, zstop = get_parameters_for(problem)
    flux = solution(z, src_mag, sigma_a, zstop)
    plt.plot(z, flux, label='analytic')
    plt.legend()
    plt.xlabel('z coordinate')
    plt.ylabel(r'$\phi(z)$')
    plt.title(f'{problem} flux')
    show_or_save(show, problem, 'analytic_flux')


def get_parameters_for(problem):
    deck = parse_input([f'{problem}.yaml'])
    mesh = create_mesh(deck)

    src_mag = deck.src['src1'].magnitude
    sigma_a = deck.mat['mat1'].sigma_a
    zstop = deck.reg['reg1'].end

    return mesh.edge, src_mag, sigma_a, zstop


def parse_args(argv):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
                description='Plot analytic flux for full_slab.yaml',
                formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s', '--show',
                        action='store_true',
                        help='show instead of save plot')
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    plot_analytic_flux(args.show, f'{MY_DIR}/full_slab')


if __name__ == '__main__':
    main()
