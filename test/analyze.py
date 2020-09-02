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

QUANTS = ['flux', 're', 'time', 'loss', 'fluxloss']


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


def plot_movie(args, npzfile, loss_limits=None, flux_limits=None, title=None,
               legend_loc=None):
    plotname = 'flux'
    algorithm2z_flux_pair = get_algorithm2z_flux_pair(npzfile)
    (nn_z, _) = algorithm2z_flux_pair['nn']
    del algorithm2z_flux_pair['nn']

    if 'flux_history' in npzfile:
        nn_flux_history = npzfile['flux_history']
    else:
        nn_flux_history = npzfile['nn_flux_history']

    if flux_limits:
        min_flux, max_flux = flux_limits
    else:
        min_flux = min([nn_flux_history.min()] +
                       [x[1].max() for x in algorithm2z_flux_pair.values()])
        max_flux = max([nn_flux_history.max()] +
                       [x[1].max() for x in algorithm2z_flux_pair.values()])

    spatial_loss_history = npzfile['spatial_loss_history']
    if loss_limits:
        min_loss, max_loss = loss_limits
    else:
        min_loss = spatial_loss_history.min()
        max_loss = spatial_loss_history.max()

    max_it = len(npzfile['loss'])
    max_num_digits = len(str(max_it))

    num_recordings = len(nn_flux_history)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for index, (nn_flux, spatial_loss) in enumerate(
            zip(nn_flux_history, spatial_loss_history)):

        if index == (num_recordings - 1):
            it = max_it
        else:
            it = index * npzfile['hinterval']
        zero_padded_it = f'{it:0{max_num_digits}}'

        ax1.set_ylim(ymin=min_flux, ymax=max_flux)
        for algorithm, (z, flux) in algorithm2z_flux_pair.items():
            ax1.plot(z, flux, label=algorithm)
        ax1.plot(nn_z, nn_flux, label=f'nn {zero_padded_it}')

        ax1.set_xlabel('z coordinate')
        ax1.set_ylabel(r'$\phi(z)$')

        ax2.set_ylabel('loss', color='r')
        ax2.set_ylim(ymin=min_loss, ymax=max_loss)
        ax2.semilogy(nn_z, spatial_loss, color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        if legend_loc:
            ax1.legend(loc=legend_loc)
        else:
            ax1.legend(loc='upper right')

        if title:
            plt.title(title)
        else:
            plt.title(f'{args.problem} {plotname}')
        show_or_save(args.show, args.problem, f'{plotname}_{zero_padded_it}')
        ax1.clear()
        ax2.clear()


def plot_flux(args, npzfile, loss=False):
    plotname = 'flux'
    if loss:
        plotname += '_loss'
    algorithm2z_flux_pair = get_algorithm2z_flux_pair(npzfile)
    print_algorithm2pair(args, algorithm2z_flux_pair, plotname)
    num_algorithms = len(algorithm2z_flux_pair)

    fig, ax1 = plt.subplots()

    for algorithm, (z, flux) in algorithm2z_flux_pair.items():
        if num_algorithms > 1:
            ax1.plot(z, flux, label=algorithm)
        else:
            ax1.plot(z, flux)

    ax1.set_xlabel('z coordinate')
    ax1.set_ylabel(r'$\phi(z)$')

    if loss:
        ax2 = ax1.twinx()
        ax2.set_ylabel('loss', color='r')
        (z, _) = algorithm2z_flux_pair['nn']
        ax2.plot(z, npzfile['spatial_loss'], color='r')
        ax2.tick_params(axis='y', labelcolor='r')

    if num_algorithms > 1:
        ax1.legend()

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
    num_algorithms = len(algorithm2z_re_pair)

    for algorithm, (z, re) in algorithm2z_re_pair.items():
        if num_algorithms > 1:
            plt.plot(z, re, label=algorithm)
        else:
            plt.plot(z, flux)

    if num_algorithms > 1:
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
    parser.add_argument('-m', '--movie',
                        action='store_true',
                        help='make a convergence movie')
    parser.add_argument('-f', '--style',
                        help='matplotlib style file')
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
    if args.style:
        plt.style.use(args.style)
    else:
        plt.style.use(f'{MY_DIR}/style.mplstyle')

    npzfile = np.load(f'{args.problem}.npz')
    if args.movie:
        plot_movie(args, npzfile)
    else:
        if 'flux' in args.quants_to_analyze:
            plot_flux(args, npzfile)
        if 're' in args.quants_to_analyze:
            plot_relative_error(args, npzfile)
        if 'time' in args.quants_to_analyze:
            print(get_runtimes(npzfile))
        if 'loss' in args.quants_to_analyze:
            plot_loss(args, npzfile)
        if 'fluxloss' in args.quants_to_analyze:
            plot_flux(args, npzfile, loss=True)


if __name__ == '__main__':
    main()
