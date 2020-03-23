#!/usr/bin/env python

import sys
import os
import pickle
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.special as sp

mpl.use('TkAgg')
plt.style.use('style.mplstyle')

ALGORITHMS = ['sn', 'mc', 'nn', 'an']
QUANTS = ['flux', 're', 'z', 'time', 'loss']
PROBLEMS = ['p1', 'p2']
problem2fname = {'p1': 'p1-full_slab',
                 'p2': 'p2-half_slab'}


def print_arrays(myargs, problem_args, npzfile):
    print('nn_zone_centers', npzfile['nn_zone_centers'])
    print('nn_flux', npzfile['nn_flux'])
    print('sn_zone_edges', npzfile['sn_zone_edges'])
    print('sn_flux', npzfile['sn_flux'])
    print('mc_zone_edges', npzfile['mc_zone_edges'])
    print('mc_flux', npzfile['mc_flux'])
    print('analytic_zone_edges', npzfile['analytic_zone_edges'])
    print('analytic_flux', npzfile['analytic_flux'])


def plot_z(myargs, problem_args, npzfile):
    nn_x = np.arange(npzfile['nn_zone_centers'].size)
    sn_x = np.arange(npzfile['sn_zone_edges'].size)
    mc_x = np.arange(npzfile['mc_zone_edges'].size)

    plt.plot(nn_x, npzfile['nn_zone_centers'], label='nn')
    plt.plot(sn_x, npzfile['sn_zone_edges'], label='sn')
    plt.plot(mc_x, npzfile['mc_zone_edges'], label='mc')

    plt.legend()
    show_or_save(myargs.show, myargs.problem, 'z')


def plot_relative_error(myargs, problem_args, npzfile):
    mc_flux = npzfile['mc_flux']
    mc_zone_centers = edges2centers(npzfile['mc_zone_edges'],
                                    problem_args.num_mc_zones)

    nn_an_flux = analytic_soln(npzfile['nn_zone_centers'])
    sn_an_flux = analytic_soln(npzfile['sn_zone_edges'])
    mc_an_flux = analytic_soln(mc_zone_centers)

    if myargs.verbose:
        print('mc_zone_centers')
        print(mc_zone_centers)
        print('mc_flux')
        print(mc_flux)
        print('mc_an_flux')
        print(mc_an_flux)

    nn_re = relative_error(npzfile['nn_flux'], nn_an_flux)
    sn_re = relative_error(npzfile['sn_flux'], sn_an_flux)
    mc_re = relative_error(mc_flux, mc_an_flux)

    if 'nn' in myargs.algorithms:
        plt.plot(npzfile['nn_zone_centers'], nn_re, label='nn')
    if 'sn' in myargs.algorithms:
        plt.plot(npzfile['sn_zone_edges'], sn_re, label='sn')
    if 'mc' in myargs.algorithms:
        plt.plot(mc_zone_centers, mc_re, label='mc')
    plt.legend()
    plt.xlabel('z coordinate')
    plt.ylabel('relative error')
    show_or_save(myargs.show, myargs.problem, 're')

    nn_mre = get_max_relative_error(nn_re, npzfile['nn_zone_centers'])
    sn_mre = get_max_relative_error(sn_re, npzfile['sn_zone_edges'])
    mc_mre = get_max_relative_error(mc_re, mc_zone_centers)

    hdr_fmt = '%2s %10s %10s'
    tbl_fmt = '%2s %10f %10.2f'
    print(hdr_fmt % ('', 'max(abs(re))', 'z_location'))
    print(tbl_fmt % ('nn', nn_mre[0], nn_mre[1]))
    print(tbl_fmt % ('sn', sn_mre[0], sn_mre[1]))
    print(tbl_fmt % ('mc', mc_mre[0], mc_mre[1]))


def get_max_relative_error(re, z):
    max_re = np.max(np.abs(re))
    where = np.argmax(np.abs(re))
    max_z = z[where]
    return max_re, max_z


def analytic_soln(z):
    EPS = 1e-9
    src_mag = 8
    sigma_t = 8
    zstop = 1
    if src_mag != sigma_t:
        print('Bad src_mag != sigma_t')
        sys.exit(-1)
    return 1 - 0.5 * (np.exp(-z * sigma_t) + np.exp(sigma_t * (z - zstop)) -
                      src_mag * z * sp.exp1(sigma_t * z + EPS) -
                      src_mag * (zstop - z) * sp.exp1(sigma_t * (zstop - z)
                      + EPS))


def plot_loss(myargs, problem_args, loss):
    plt.semilogy(np.arange(loss.size), loss)
    plt.xlabel('epoch')
    plt.ylabel(r'$\mathcal{L}$')
    show_or_save(myargs.show, myargs.problem, 'loss')


def plot_flux(myargs, problem_args, npzfile):
    nn_flux = npzfile['nn_flux']
    nn_zone_centers = npzfile['nn_zone_centers']
    nn_label = 'nn'

    sn_flux = npzfile['sn_flux']
    sn_zone_edges = npzfile['sn_zone_edges']
    sn_label = 'sn'

    mc_flux = npzfile['mc_flux']
    mc_zone_centers = edges2centers(npzfile['mc_zone_edges'],
                                    problem_args.num_mc_zones)
    mc_label = 'mc'

    if myargs.problem == 'p1':
        analytic_zone_edges = sn_zone_edges
        analytic_flux = analytic_soln(analytic_zone_edges)

    if myargs.verbose:
        print('analytic_zone_edges')
        print(analytic_zone_edges)
        print('analytic_flux')
        print(analytic_flux)
        print('mc_flux')
        print(mc_flux)
        print('mc_zone_centers')
        print(mc_zone_centers)
        print('nn_zone_centers')
        print(nn_zone_centers)
        print('nn_flux')
        print(nn_flux)

    plotname = 'flux_'
    if 'nn' in myargs.algorithms:
        plt.plot(nn_zone_centers, nn_flux, label=nn_label)
        plotname += 'nn'
    if 'sn' in myargs.algorithms:
        plt.plot(sn_zone_edges, sn_flux, label=sn_label)
        plotname += 'sn'
    if 'mc' in myargs.algorithms:
        plt.plot(mc_zone_centers, mc_flux, label=mc_label)
        plotname += 'mc'
    if myargs.problem == 'p1' and 'an' in myargs.algorithms:
        plt.plot(analytic_zone_edges, analytic_flux, label='analytic')
        plotname += 'analytic'
    plt.legend()
    plt.xlabel('z coordinate')
    plt.ylabel(r'$\phi(z)$')
    show_or_save(myargs.show, myargs.problem, plotname)


def relative_error(estimate, true):
    return (true - estimate) / true


def edges2centers(edges, num_zones):
    shift_amount = (1. / num_zones) / 2
    centers = (edges + shift_amount)[:-1]
    return centers


def load_dict(fname):
    with open(fname) as f:
        lines = f.readlines()
    return eval(lines[0][:-1])


def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def main(myargs):
    fname = problem2fname[myargs.problem]
    npzfile = np.load('%s.npz' % fname)
    problem_args = load_pickle('%s.args' % fname)
    runtimes = load_dict('%s.time' % fname)
    loss = load_pickle('%s.loss' % fname)
    if 'z' in myargs.quants_to_analyze:
        plot_z(myargs, problem_args, npzfile)
    if 're' in myargs.quants_to_analyze:
        plot_relative_error(myargs, problem_args, npzfile)
    if 'flux' in myargs.quants_to_analyze:
        plot_flux(myargs, problem_args, npzfile)
    if 'time' in myargs.quants_to_analyze:
        print_runtimes(runtimes)
    if 'loss' in myargs.quants_to_analyze:
        plot_loss(myargs, problem_args, loss)


def print_runtimes(runtimes):
    for algo, time in runtimes.items():
        print('%7s' % algo, '%.2e' % time)


def show_or_save(show, probname, plotname):
    if show:
        plt.show()
    else:
        if not os.path.exists('fig'):
            os.mkdir('fig')
        plt.savefig('fig/%s%s.eps' % (probname, plotname))
    plt.clf()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithms',
                        type=str,
                        choices=ALGORITHMS,
                        default=ALGORITHMS,
                        nargs='+',
                        help='the algorithms to plot')
    parser.add_argument('-q', '--quants_to_analyze',
                        type=str,
                        choices=QUANTS,
                        default='re',
                        nargs='+',
                        help='the quantities to analyze')
    parser.add_argument('-p', '--problem',
                        type=str,
                        choices=PROBLEMS,
                        default='p1',
                        help='the problem to analyze')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='verbose output')
    parser.add_argument('-s', '--show',
                        action='store_true',
                        help='show instead of save plot')

    if input_args:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
