#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.special as sp
import sys

import analyze

MY_DIR = os.path.dirname(__file__)
sys.path.append(f'{MY_DIR}/..')
import narrows  # noqa: E402


plt.style.use(f'{os.path.dirname(__file__)}/style.mplstyle')


def solution(z, src_mag, sigma_t, zstop):
    EPS = 1e-25

    return ((src_mag / sigma_t) -
            (src_mag / (2 * sigma_t)) * (np.exp(-sigma_t * z) +
                                         np.exp(sigma_t * (z - zstop))) +
            (src_mag / 2) * (z * sp.exp1(sigma_t * z + EPS) +
                             (zstop - z) * sp.exp1(sigma_t * (zstop - z)
                                                   + EPS)))


def plot_analytic_flux(show, problem):
    z, src_mag, sigma_t, zstop = get_parameters_for(problem)
    flux = solution(z, src_mag, sigma_t, zstop)
    plt.plot(z, flux, label='analytic')
    plt.legend()
    plt.xlabel('z coordinate')
    plt.ylabel(r'$\phi(z)$')
    plt.title(f'{problem} flux')
    analyze.show_or_save(show, problem, '_analytic_flux')


def get_parameters_for(problem):
    deck = narrows.parse_input([f'{problem}.yaml'])
    mesh = narrows.create_mesh(deck)

    src_mag = deck.src['src1'].magnitude
    sigma_t = deck.mat['mat1'].sigma_a + deck.mat['mat1'].sigma_s0
    zstop = deck.reg['reg1'].end

    return mesh.edge, src_mag, sigma_t, zstop


if __name__ == '__main__':
    plot_analytic_flux(False, 'full_slab')
