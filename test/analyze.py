#!/usr/bin/env python
'''
Analyze the output from yamldriver.py
'''

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

import analytic_full_slab

plt.style.use(f'{os.path.dirname(__file__)}/style.mplstyle')

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


def plot_loss(args):
    plotname = 'loss'
    loss = load_pickle(f'{args.problem}.loss.pkl')
    plt.semilogy(np.arange(loss.size), loss)
    plt.xlabel('epoch')
    plt.ylabel(r'$\mathcal{L}$')
    plt.title(f'{args.problem} {plotname}')
    show_or_save(args.show, args.problem, plotname)


def get_algorithm2z_flux_pair(npzfile):
    edge = npzfile['edge']
    algorithm2z_flux_pair = {}
    if 'flux' in npzfile:
        algorithm2z_flux_pair[None] = (edge, npzfile['flux'])
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
            if algorithm:
                print(f'z: {z}')
                print(f'{algorithm}_{suffix}: ', end='')
            print(y)
            print()


def plot_flux(args, npzfile):
    plotname = 'flux'
    algorithm2z_flux_pair = get_algorithm2z_flux_pair(npzfile)
    print_algorithm2pair(args, algorithm2z_flux_pair, plotname)

    for algorithm, (z, flux) in algorithm2z_flux_pair.items():
        if algorithm:
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
    _, src_mag, sigma_t, zstop = \
        analytic_full_slab.get_parameters_for(args.problem)

    algorithm2z_re_pair = {}
    algorithm2maxre_z_pair = {}
    for algorithm, (z, flux) in algorithm2z_flux_pair.items():
        analytic_soln = analytic_full_slab.solution(z, src_mag, sigma_t, zstop)
        re = relative_error(flux, analytic_soln)
        algorithm2z_re_pair[algorithm] = (z, re)
        algorithm2maxre_z_pair[algorithm] = get_max_relative_error(re, z)

    print_algorithm2pair(args, algorithm2z_re_pair, plotname)

    for algorithm, (z, re) in algorithm2z_re_pair.items():
        if algorithm:
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


def load_dict(fname):
    with open(fname) as f:
        lines = f.readlines()
    return eval(lines[0][:-1])


def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def print_runtimes(runtimes):
    for algo, time in runtimes.items():
        print('%7s' % algo, '%.2e' % time)


def show_or_save(show, probname, plotname):
    if show:
        plt.show()
    else:
        if not os.path.exists('fig'):
            os.mkdir('fig')
        plt.savefig('fig/%s%s.png' % (probname, plotname))
    plt.clf()


def parse_args():
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
    args = parser.parse_args()
    return args


def main(args):
    npzfile = np.load(f'{args.problem}.npz')
    runtimes = load_dict(f'{args.problem}.time')
    if 'flux' in args.quants_to_analyze:
        plot_flux(args, npzfile)
    if 're' in args.quants_to_analyze:
        plot_relative_error(args, npzfile)
    if 'time' in args.quants_to_analyze:
        print_runtimes(runtimes)
    if 'loss' in args.quants_to_analyze:
        plot_loss(args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
