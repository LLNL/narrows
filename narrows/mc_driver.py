#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np

import mc1d

NUM_PARTICLES = int(1e6)
NUM_ZONES = 10
ZSTOP = 1
SOURCE = [0, 1]
SIGMA_S0 = 9.5
SIGMA_S1 = 0.
SIGMA_T = 10.
SEED = 1 << 20
MAX_NUM_SEGMENTS = 1e2 * NUM_ZONES
NUM_PHYSICAL_PARTICLES = 1.


def runp1orp2(args, probname):
    print('%s Inputs:' % probname)
    print_args(args)
    result = mc1d.main(args)
    print('%s Results:' % probname)
    print(result.tally.dump())
    plot_flux(args, result, probname)


def plot_flux(args, result, title, ymin=0):
    flux = get_weighted_flux(args, result)
    plt.plot(result.zone_edges[:-1], flux)
    # TODO: plot error bars!!!
    # plt.plot(result.zone_edges[:-1], get_flux_std_dev(args, result))
    setymin(ymin, flux)
    show_or_save(args, title)


def show_or_save(args, title):
    if args.show:
        plt.show()
    else:
        plt.savefig('%s.png' % title)
    plt.clf()


def setymin(ymin, flux):
    if ymin is not None:
        ymin = min(0, np.min(flux))
        ymax = np.max(flux) * 1.1
        plt.ylim(ymin, ymax)


def get_weight(args, result):
    return args.num_physical_particles / result.tally._num_particles


def get_weighted_current(args, result, neg):
    weight = get_weight(args, result)
    if neg:
        return result.tally._negcur * weight
    else:
        return result.tally._poscur * weight


def get_weighted_flux(args, result):
    weight = get_weight(args, result)
    return result.tally.get_flux() * weight


def get_weighted_flux_squared(args, result):
    weight = get_weight(args, result)
    return result.tally._flux_squared * weight


def get_flux_std_dev(args, result):
    flux = get_weighted_flux(args, result)
    flux2 = get_weighted_flux_squared(args, result)
    weight = get_weight(args, result)
    return np.sqrt(weight * (flux2 - (flux * flux)))


def print_args(args):
    for arg in vars(args):
        print('%20s' % arg, getattr(args, arg))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--num_particles',
                        default=NUM_PARTICLES, type=int,
                        help='the number of particle histories to run')
    parser.add_argument('-J', '--num_zones',
                        default=NUM_ZONES, type=int,
                        help='the number of zones')
    parser.add_argument('-H', '--zstop',
                        default=ZSTOP, type=int,
                        help='the z extent')
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
    parser.add_argument('-Q', '--uniform_source_extent',
                        default=SOURCE,
                        nargs=2,
                        type=float,
                        help='extent (low, high) of uniform source')
    parser.add_argument('-P', '--point_source_location',
                        type=float,
                        help='location of optional point source')
    parser.add_argument('-Y', '--num_physical_particles',
                        default=NUM_PHYSICAL_PARTICLES,
                        type=float,
                        help='number of physical particles used in tally '
                             'normalization')
    parser.add_argument('-S', '--seed',
                        default=SEED, type=int,
                        help='the seed for the particle source')
    parser.add_argument('-M', '--max_num_segments',
                        default=MAX_NUM_SEGMENTS, type=int,
                        help='the maximum number of segments a particle can '
                             'travel')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='verbose output')
    parser.add_argument('-s', '--show',
                        action='store_true',
                        help='show plots')
    parser.add_argument('-o', '--output_directory',
                        type=str,
                        help='directory in which to write output')

    if input_args:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.uniform_source_extent[0] < 0 or \
       args.uniform_source_extent[1] > args.zstop:
        raise RuntimeError('uniform_source_extent on (%.2f, %.2f) must be in '
                           '[0, %d]' % (args.uniform_source_extent[0],
                                        args.uniform_source_extent[1],
                                        args.zstop))
    if args.point_source_location:
        if args.point_source_location < 0 or \
           args.point_source_location > args.zstop:
            raise RuntimeError('point_source_location (%.2f) must be in '
                               '[0, %d]' % (args.point_source_location,
                                            args.zstop))
    return args


_ = '''\
if __name__ == '__main__':
    args = parse_args()
    mc1d.main(args)
'''
