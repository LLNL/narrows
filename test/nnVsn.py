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
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd
import sys
import yaml

import analyze
import analytic_full_slab

MY_DIR = os.path.dirname(__file__)
sys.path.append(f'{MY_DIR}/..')
import narrows  # noqa: E402

plt.style.use(f'{MY_DIR}/style.mplstyle')

PROB = f'{MY_DIR}/full_slab'
START = 1
STOP = 4
SCALES = range(START, STOP + 1)
DOTTED = '--'
POINTS = '.'


def run_scaling_study(epsilon):
    with open(f'{PROB}.yaml') as f:
        yamlinput = yaml.full_load(f)
    yamlinput['ctrl']['mc'] = False

    results = OrderedDict()
    for scale in SCALES:
        print('Running 10^%d nodes' % scale)
        num_cells = int(math.pow(10, scale))
        yamlinput['ctrl']['cells_per_region'] = num_cells
        problem = f'{PROB}_{scale}'
        with open(f'{problem}.yaml', 'w') as f:
            yaml.dump(yamlinput, f)
        argv = f'''
{problem}.yaml
-d epsilon={epsilon}
-d verb=terse'''.split()
        narrows.main(argv)

        npzfile = np.load(f'{problem}.npz')

        results[scale] = {'edge': npzfile['edge'],
                          'nn_flux': npzfile['nn_flux'],
                          'nn_pred_time': npzfile['pred_time'],
                          'sn_flux': npzfile['sn_flux'],
                          'sn_runtime': npzfile['sn_time']}

    out_fname = '%s.%dto%d.scaling' % (PROB, START, STOP)
    with open(out_fname, 'wb') as f:
        pickle.dump(results, f)

    return results


def load_results():
    fname = '%s.%dto%d.scaling' % (PROB, START, STOP)
    with open(fname, 'rb') as f:
        results = pickle.load(f)

    return results


def plot_errors(show, nn_errors, sn_errors):
    fig, ax = plt.subplots()
    x = [10**s for s in SCALES]
    plt.semilogx(x, nn_errors, label='nn', linestyle=DOTTED, marker=POINTS)
    plt.semilogx(x, sn_errors, label='sn', linestyle=DOTTED, marker=POINTS)
    plt.legend()
    plt.title(f'{PROB} scaling study max relative error')
    plt.xlabel('number of zones')
    plt.ylabel('max relative error')
    analyze.show_or_save(show, PROB, 'scalstud_re')


def plot_times(show, nn_times, sn_times):
    fig, ax = plt.subplots()
    x = [10**s for s in SCALES]
    plt.loglog(x, nn_times, label='nn pred', linestyle=DOTTED, marker=POINTS)
    plt.loglog(x, sn_times, label='sn', linestyle=DOTTED, marker=POINTS)
    plt.legend()
    plt.title(f'{PROB} scaling study runtime')
    plt.xlabel('number of zones')
    plt.ylabel('runtime (s)')

    print_times(nn_times, sn_times)
    analyze.show_or_save(show, PROB, 'scalstud_time')


def print_times(nn_times, sn_times):
    df = pd.DataFrame({'nn': nn_times, 'sn': sn_times})
    scales = [f'10^{x}' for x in SCALES]
    column_mapper = {index: scale for index, scale in enumerate(scales)}
    new_df = df.transpose().rename(columns=column_mapper)
    pd.set_option('display.float_format', lambda x: '%.2e' % x)
    print(new_df)


def analyze_scaling_study(show, results):

    with open(f'{PROB}.yaml') as f:
        yamlinput = yaml.full_load(f)

    src_mag = yamlinput['src']['src1']['magnitude']
    sigma_t = (yamlinput['mat']['mat1']['sigma_a'] +
               yamlinput['mat']['mat1']['sigma_s0'])
    zstop = yamlinput['reg']['reg1']['end']

    nn_errors = []
    nn_times = []
    sn_errors = []
    sn_times = []
    for scale in SCALES:
        edge = results[scale]['edge']
        an_flux = analytic_full_slab.solution(edge, src_mag, sigma_t, zstop)

        nn_flux = results[scale]['nn_flux']
        nn_pred_time = results[scale]['nn_pred_time']
        nn_re = analyze.relative_error(nn_flux, an_flux)

        sn_flux = results[scale]['sn_flux']
        sn_runtime = results[scale]['sn_runtime']
        sn_re = analyze.relative_error(sn_flux, an_flux)

        nn_mre, nn_mre_location = analyze.get_max_relative_error(nn_re, edge)
        sn_mre, sn_mre_location = analyze.get_max_relative_error(sn_re, edge)

        nn_errors.append(nn_mre)
        sn_errors.append(sn_mre)

        nn_times.append(nn_pred_time)
        sn_times.append(sn_runtime)

    plot_errors(show, nn_errors, sn_errors)
    plot_times(show, nn_times, sn_times)


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run',
                        action='store_true',
                        help='run the scaling study')
    parser.add_argument('-e', '--epsilon',
                        default=0.1,
                        help='convergence criterion epsilon')
    parser.add_argument('-s', '--show',
                        action='store_true',
                        help='show plots')

    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.run:
        results = run_scaling_study(args.epsilon)
    else:
        results = load_results()
        analyze_scaling_study(args.show, results)


if __name__ == '__main__':
    main()
