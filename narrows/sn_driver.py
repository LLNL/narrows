#!/usr/bin/env python

import argparse
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import shutil

import sn1d

mpl.use('TkAgg')

MAX_NUM_ITER = 300
NUM_ORDINATES = 2
NUM_ZONES = 10
SOURCE = [1.]
ZSTOP = 1.
EPSILON = 1e-6
SIGMA_S0 = 9.5
SIGMA_S1 = 0.
SIGMA_T = 10.


def runp1orp2(args, probname):
    delete_and_makedirs(args.output_directory)
    p1orp2common(args, probname)
    args.num_ordinates = 4
    args.num_zones = 20
    p1orp2common(args, probname)
    args.num_zones = 1000
    p1orp2common(args, probname)


def p1orp2common(args, probname):
    print_args(args)
    result = sn1d.main(args)
    ylabel = 'Scalar flux (1 / (cm2 * s) )'
    xlabel = 'Position in slab (cm)'
    title = 'Flux %s N=%d J=%d' % (probname, args.num_ordinates,
                                   args.num_zones)
    plot_flux(args, result, xlabel, ylabel, title)
    write_result(args, result, title)

    return result


def write_result(args, result, title):

    timestamp = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    argdump = ['%s, %s\n' % (key, str(value)) for key, value in
               vars(args).items()]
    flux = ['%f, %f\n' % (zone_edge, flux) for zone_edge, flux in
            zip(result.zone_edges, result.I0)]
    code_info = '%s, %s, %s' % (sn1d.MODULE_NAME, sn1d.__version__, timestamp)

    output = ['%s\n' % code_info] + argdump + flux

    fname = '%s.csv' % title2fname(args, title)
    with open(fname, 'w') as f:
        f.writelines(output)


def delete_and_makedirs(dirname):
    if os.path.exists(dirname):
        print('Deleting %s' % dirname)
        shutil.rmtree(dirname)
    print('Making %s' % dirname)
    os.makedirs(dirname)


def title2fname(args, title):
    return '%s/%s' % (args.output_directory, title.replace(' ', '_'))


def print_args(args):
    for arg in vars(args):
        print('%20s' % arg, getattr(args, arg))


def plot_flux(args, result, xlabel, ylabel, title):
    plt.plot(result.zone_edges, result.I0, label='After iter %d' % result.i)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if args.show:
        plt.show()
    else:
        plt.savefig('%s.png' % title2fname(args, title))
    plt.clf()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--num_ordinates',
                        default=NUM_ORDINATES, type=int,
                        help='the number of discrete ordinates')
    parser.add_argument('-J', '--num_zones',
                        default=NUM_ZONES, type=int,
                        help='the number of zones')
    parser.add_argument('-H', '--zstop',
                        default=ZSTOP, type=int,
                        help='the z extent')
    parser.add_argument('-e', '--epsilon',
                        default=EPSILON, type=float,
                        help='used for source iteration convergence criterion')
    parser.add_argument('--sigma_t',
                        default=SIGMA_T, type=float,
                        help='the total macroscopic cross section')
    parser.add_argument('--sigma_s0',
                        default=SIGMA_S0, type=float,
                        help='the isotropic scattering '
                             'macroscopic cross section')
    parser.add_argument('--sigma_s1',
                        default=SIGMA_S1, type=float,
                        help='the linear anisotropic scattering '
                             'macroscopic cross section')
    parser.add_argument('-I', '--max_num_iter',
                        default=MAX_NUM_ITER, type=int,
                        help='the maximum number of iterations')
    parser.add_argument('-Q', '--source',
                        default=SOURCE,
                        nargs='+',
                        type=float,
                        help='description of source')
    parser.add_argument('--point-source-at-zero',
                        default=0, type=float,
                        help='strength of isotropic point source at z=0')
    parser.add_argument('-q', '--quiet',
                        action='store_true',
                        help='description of source')
    parser.add_argument('-s', '--show',
                        action='store_true',
                        help='show plots')
    parser.add_argument('-o', '--output_directory',
                        type=str,
                        help='directory in which to write output')
    parser.add_argument('-v', '--version',
                        action='version',
                        version='%s %s' % (sn1d.MODULE_NAME, sn1d.__version__))

    if input_args:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if len(args.source) != args.zstop:
        raise RuntimeError("len(source):%d and z-extent:%d must be equal"
                           % (len(args.source), args.zstop))
    return args
