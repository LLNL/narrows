#!/usr/bin/env python
'''
Perform a scaling study from 10^START to 10^6 zones
in powers of ten, comparing the runtime of
nn prediction vs sn, as well as the accuracy.

The point is to answer the question:
Is it faster to use a nn trained on a low-res
solution to predict a high-res solution than
it is to run Sn?
'''

import argparse
from collections import OrderedDict
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle

import analyze
import driver

mpl.use('TkAgg')
plt.style.use('style.mplstyle')

PROB = 'p1-full_slab'
START = 1
STOP = 6
SCALES = range(START, STOP + 1)
DOTTED = '--'
POINTS = '.'


def run_scaling_study():

    problem_args = analyze.load_pickle('%s.args' % PROB)
    nn_solver = analyze.load_pickle('%s.nn' % PROB)

    results = OrderedDict()

    for scale in SCALES:
        print('Running 10^%d nodes' % scale)
        num_zones = int(math.pow(10, scale))
        nn_z = np.linspace(0, 1, num_zones, dtype=np.float32)
        nn_flux, nn_pred_time = driver.predict_nn(nn_solver, nn_z)

        problem_args.num_sn_zones = num_zones
        sn_flux, sn_zone_edges, sn_runtime = driver.run_sn(problem_args)

        results[scale] = {'nn_flux': nn_flux,
                          'nn_z': nn_z,
                          'nn_pred_time': nn_pred_time,
                          'sn_flux': sn_flux,
                          'sn_zone_edges': sn_zone_edges,
                          'sn_runtime': sn_runtime}

    out_fname = '%s.%dto%d.scaling' % (PROB, START, STOP)
    with open(out_fname, 'wb') as f:
        pickle.dump(results, f)

    return results


def plot_re(nn_re, nn_z, sn_re, sn_z):
    plt.plot(nn_z, nn_re, label='nn')
    plt.plot(sn_z, sn_re, label='sn')
    plt.legend()
    plt.xlabel('z')
    plt.ylabel('relative error')

    nn_mre = analyze.get_max_relative_error(nn_re, nn_z)
    sn_mre = analyze.get_max_relative_error(sn_re, nn_z)

    hdr_fmt = '%2s %10s %10s'
    tbl_fmt = '%2s %10f %10.2f'
    print(hdr_fmt % ('', 'max(abs(re))', 'z_location'))
    print(tbl_fmt % ('nn', nn_mre[0], nn_mre[1]))
    print(tbl_fmt % ('sn', sn_mre[0], sn_mre[1]))

    analyze.show_or_save(True, 'foo', 'bar')


def load_results():
    fname = '%s.%dto%d.scaling' % (PROB, START, STOP)
    with open(fname, 'rb') as f:
        results = pickle.load(f)

    return results


def plot_errors(args, nn_errors, sn_errors):
    fig, ax = plt.subplots()
    x = [10**s for s in SCALES]
    plt.semilogx(x, nn_errors, label='nn', linestyle=DOTTED, marker=POINTS)
    plt.semilogx(x, sn_errors, label='sn', linestyle=DOTTED, marker=POINTS)
    plt.legend()
    plt.xlabel('number of zones')
    plt.ylabel('max relative error')
    analyze.show_or_save(args.show, args.problem_name, '_re')


def plot_times(args, nn_times, sn_times):
    fig, ax = plt.subplots()
    x = [10**s for s in SCALES]
    plt.loglog(x, nn_times, label='nn pred', linestyle=DOTTED, marker=POINTS)
    plt.loglog(x, sn_times, label='sn', linestyle=DOTTED, marker=POINTS)
    plt.legend()
    plt.xlabel('number of zones')
    plt.ylabel('runtime (s)')

    hdr = 'alg ' + (' '.join(('%8s' % ('10^%d' % x) for x in SCALES)))
    nn = 'nn  ' + (' '.join('%.2e' % t for t in nn_times))
    sn = 'sn  ' + (' '.join('%.2e' % t for t in sn_times))

    print(hdr)
    print(nn)
    print(sn)

    analyze.show_or_save(args.show, args.problem_name, '_time')


def analyze_scaling_study(args, results):
    nn_errors = []
    nn_times = []
    sn_errors = []
    sn_times = []

    for scale in SCALES:
        nn_flux = results[scale]['nn_flux']
        nn_z = results[scale]['nn_z']
        nn_pred_time = results[scale]['nn_pred_time']
        nn_an_flux = analyze.analytic_soln(nn_z)
        nn_re = analyze.relative_error(nn_flux, nn_an_flux)

        sn_flux = results[scale]['sn_flux']
        sn_z = results[scale]['sn_zone_edges']
        sn_runtime = results[scale]['sn_runtime']
        sn_an_flux = analyze.analytic_soln(sn_z)
        sn_re = analyze.relative_error(sn_flux, sn_an_flux)

        # print('nn', '%.2e' % nn_pred_time)
        # print('sn', '%.2e' % sn_runtime)
        # plot_re(nn_re, nn_z, sn_re, sn_z)

        nn_mre, nn_mre_location = analyze.get_max_relative_error(nn_re, nn_z)
        sn_mre, sn_mre_location = analyze.get_max_relative_error(sn_re, nn_z)

        nn_errors.append(nn_mre)
        sn_errors.append(sn_mre)

        nn_times.append(nn_pred_time)
        sn_times.append(sn_runtime)

    plot_errors(args, nn_errors, sn_errors)
    plot_times(args, nn_times, sn_times)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('problem_name',
                        help='a name to identify the problem we are running')
    parser.add_argument('-r', '--run',
                        action='store_true',
                        help='run the scaling study')
    parser.add_argument('-s', '--show',
                        action='store_true',
                        help='show plots')

    if input_args:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


if __name__ == '__main__':
    # args = parse_args('p1scalstud -s'.split())
    # args = parse_args('p1scalstud'.split())
    args = parse_args()
    if args.run:
        results = run_scaling_study()
    else:
        results = load_results()
        analyze_scaling_study(args, results)
