#!/usr/bin/env python
'''
Analyze the output from yamldriver.py
'''

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

from analytic_full_slab import (
    get_parameters_for,
    solution
)

MY_DIR = os.path.dirname(__file__)
sys.path.append(f'{MY_DIR}/..')
from narrows import (  # noqa: E402
    get_runtimes
)

sys.path.append(f'{MY_DIR}')
from utility import (  # noqa: E402
    show_or_save
)

plt.style.use(f'{MY_DIR}/style.mplstyle')

QUANTS = ['flux', 're', 'time', 'loss']


def relative_error(estimate, true):
    return (true - estimate) / true


def get_max_relative_error(re, z):
    max_re = np.max(np.abs(re))
    where = np.argmax(np.abs(re))
    max_z = z[where]
    return max_re, max_z


def edge2center(edge):
    shifted = np.roll(edge, -1)
    halfway = (shifted - edge) / 2
    center = edge + halfway
    return center[:-1]


def plot_loss(args, npzfile):
    plotname = 'loss'
    loss = npzfile['loss']
    plt.semilogy(np.arange(loss.size), loss)
    plt.xlabel('epoch')
    plt.ylabel(r'$\mathcal{L}$')
    plt.title(f'{args.problem} {plotname}')
    show_or_save(args.show, args.problem, plotname)


def get_algorithm2z_flux_pair(npzfile):
    edge = npzfile['edge']
    algorithm2z_flux_pair = {}
    if 'flux' in npzfile:
        algorithm = npzfile['algorithm'].item()
        z = edge2center(edge) if algorithm == 'mc' else edge
        algorithm2z_flux_pair[algorithm] = (z, npzfile['flux'])
    else:
        if 'nn_flux' in npzfile:
            algorithm2z_flux_pair['nn'] = (edge, npzfile['nn_flux'])
        if 'mc_flux' in npzfile:
            center = edge2center(edge)
            algorithm2z_flux_pair['mc'] = (center, npzfile['mc_flux'])
        if 'sn_flux' in npzfile:
            algorithm2z_flux_pair['sn'] = (edge, npzfile['sn_flux'])
    return algorithm2z_flux_pair


def print_algorithm2pair(args, algorithm2pair, suffix):
    if args.verbose:
        for algorithm, (z, y) in algorithm2pair.items():
            if len(algorithm2pair) > 1:
                print(f'z: {z}')
                print(f'{algorithm}_{suffix}: ', end='')
            print(y)
            print()


def plot_flux(args, npzfile):
    plotname = 'flux'
    algorithm2z_flux_pair = get_algorithm2z_flux_pair(npzfile)
    print_algorithm2pair(args, algorithm2z_flux_pair, plotname)

    for algorithm, (z, flux) in algorithm2z_flux_pair.items():
        if len(algorithm2z_flux_pair) > 1:
            plt.plot(z, flux, label=algorithm)
        else:
            plt.plot(z, flux)

    if any(algorithm2z_flux_pair.keys()):
        plt.legend()
    plt.xlabel('z coordinate')
    plt.ylabel(r'$\phi(z)$')
    plt.title(f'{args.problem} {plotname}')
    show_or_save(args.show, args.problem, plotname)


def plot_relative_error(args, npzfile):
    plotname = 're'

    algorithm2z_flux_pair = get_algorithm2z_flux_pair(npzfile)
    _, src_mag, sigma_t, zstop = get_parameters_for(args.problem)

    algorithm2z_re_pair = {}
    algorithm2maxre_z_pair = {}
    for algorithm, (z, flux) in algorithm2z_flux_pair.items():
        analytic_soln = solution(z, src_mag, sigma_t, zstop)
        re = relative_error(flux, analytic_soln)
        algorithm2z_re_pair[algorithm] = (z, re)
        algorithm2maxre_z_pair[algorithm] = get_max_relative_error(re, z)

    print_algorithm2pair(args, algorithm2z_re_pair, plotname)

    for algorithm, (z, re) in algorithm2z_re_pair.items():
        if len(algorithm2z_re_pair) > 1:
            plt.plot(z, re, label=algorithm)
        else:
            plt.plot(z, flux)

    if any(algorithm2z_re_pair.keys()):
        plt.legend()
    plt.title(f'{args.problem} relative error')
    plt.xlabel('z coordinate')
    plt.ylabel('relative error')
    show_or_save(args.show, args.problem, plotname)

    print_maxre(algorithm2maxre_z_pair)


def print_maxre(algorithm2maxre_z_pair):
    df = pd.DataFrame(algorithm2maxre_z_pair)
    labels = 'max(abs(re)) z_location'.split()
    column_mapper = {index: label for index, label in enumerate(labels)}
    new_df = df.transpose().rename(columns=column_mapper)
    print(new_df)


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
                description=__doc__,
                formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('problem',
                        help='the name of the problem eg reedp1')
    parser.add_argument('-q', '--quants_to_analyze',
                        type=str,
                        choices=QUANTS,
                        default='flux',
                        nargs='+',
                        help='the quantities to analyze')
    parser.add_argument('-s', '--show',
                        action='store_true',
                        help='show instead of save plot')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='verbose output')
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    npzfile = np.load(f'{args.problem}.npz')
    if 'flux' in args.quants_to_analyze:
        plot_flux(args, npzfile)
    if 're' in args.quants_to_analyze:
        plot_relative_error(args, npzfile)
    if 'time' in args.quants_to_analyze:
        print(get_runtimes(npzfile))
    if 'loss' in args.quants_to_analyze:
        plot_loss(args, npzfile)


if __name__ == '__main__':
    main()
