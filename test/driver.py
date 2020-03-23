#!/usr/bin/env python
import pickle
import time
import argparse
import numpy as np
import torch
import sys
sys.path.append('..')

#from ann.ANNSlabSolver import ANNSlabSolver
#from sn import sn1d
#from mc import mc1d
from narrows import ANNSlabSolver, sn1d, mc1d

np.random.seed(0)
torch.manual_seed(0)

# Shared All
SOURCE = [0, 1]
ZSTOP = 1.
SIGMA_S0 = 9.5
SIGMA_S1 = 0.
SIGMA_T = 10.
SOURCE_MAGNITUDE = 1.

# Shared NN and SN
NUM_ORDINATES = 4

# NN exclusive
NUM_NN_ZONES = 50
LEARNING_RATE = 1e-3
NN_EPSILON = 1e-13
GAMMA_R = 100
NUM_HIDDEN_LAYER_NODES = 5

# SN exclusive
NUM_SN_ZONES = 50
MAX_NUM_ITER = 300
SN_EPSILON = 1e-6

# MC exclusive
NUM_MC_ZONES = 50
NUM_PARTICLES = int(1e4)
SEED = 1 << 20
MAX_NUM_SEGMENTS = 1e2 * NUM_MC_ZONES
NUM_PHYSICAL_PARTICLES = 1.


def run(args):
    write_pickle_to_file(args, args, 'args')

    if args.skip_nn:
        sn_flux, sn_zone_edges, sn_runtime = run_sn(args)
        mc_flux, mc_zone_edges, mc_runtime = run_mc(args)
        runtimes = {'sn': sn_runtime,
                    'mc': mc_runtime}
        np.savez('%s.npz' % args.problem_name,
                 sn_zone_edges=sn_zone_edges,
                 sn_flux=sn_flux,
                 mc_zone_edges=mc_zone_edges,
                 mc_flux=mc_flux)
    else:
        nn_flux, nn_zone_centers, nn_train_runtime, nn_pred_runtime = \
                run_nn(args)
        sn_flux, sn_zone_edges, sn_runtime = run_sn(args)
        mc_flux, mc_zone_edges, mc_runtime = run_mc(args)
        runtimes = {'nntrain': nn_train_runtime,
                    'nnpred': nn_pred_runtime,
                    'sn': sn_runtime,
                    'mc': mc_runtime}
        np.savez('%s.npz' % args.problem_name,
                 nn_zone_centers=nn_zone_centers,
                 nn_flux=nn_flux,
                 sn_zone_edges=sn_zone_edges,
                 sn_flux=sn_flux,
                 mc_zone_edges=mc_zone_edges,
                 mc_flux=mc_flux)

    write_runtimes_to_file(args, runtimes)


def write_runtimes_to_file(args, runtimes):
    with open('%s.time' % args.problem_name, 'w') as f:
        f.write('%s\n' % str(runtimes))


def run_mc(args):
    start = time.time()
    result = mc1d.main(args)
    runtime = time.time() - start
    flux = get_weighted_flux(args, result)
    zone_edges = result.zone_edges

    return flux, zone_edges, runtime


def run_sn(args):
    start = time.time()
    result = sn1d.main(args)
    runtime = time.time() - start
    flux = result.I0
    zone_edges = result.zone_edges
    return flux, zone_edges, runtime


def create_nn_object(args):
    source_fraction = args.uniform_source_extent[1]
    nn_solver = ANNSlabSolver(args.num_ordinates,
                              args.num_hidden_layer_nodes,
                              args.num_nn_zones,
                              z_max=args.zstop,
                              sigma_t=args.sigma_t,
                              sigma_s0=args.sigma_s0,
                              sigma_s1=args.sigma_s1,
                              source=args.source_magnitude,
                              source_fraction=source_fraction,
                              learning_rate=args.learning_rate,
                              gamma_r=args.gamma_r,
                              eps=args.epsilon_nn)
    return nn_solver


def train_nn(args, nn_solver):
    print("Training neural network")
    start = time.time()
    loss_history = nn_solver.train()
    training_runtime = time.time() - start

    write_pickle_to_file(args, loss_history, 'loss')
    return training_runtime


def predict_nn(nn_solver, z=None):
    start = time.time()
    flux = nn_solver.predict(z)
    prediction_runtime = time.time() - start
    return flux, prediction_runtime


def run_nn(args):
    nn_solver = create_nn_object(args)
    training_time = train_nn(args, nn_solver)
    flux, prediction_time = predict_nn(nn_solver)

    zone_edges = nn_solver.z

    write_pickle_to_file(args, nn_solver, 'nn')

    return flux, zone_edges, training_time, prediction_time


def write_pickle_to_file(args, data, ext):
    fname = '%s.%s' % (args.problem_name, ext)
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def get_weighted_flux(args, result):
    weight = get_weight(args, result)
    return result.tally.get_flux() * weight


def get_weight(args, result):
    return args.num_physical_particles / result.tally._num_particles


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('problem_name',
                        help='a name to identify the problem we are running')
    parser.add_argument('-NP', '--num_particles',
                        default=NUM_PARTICLES, type=int,
                        help='the number of particle histories to run')
    parser.add_argument('-NR', '--num_ordinates',
                        default=NUM_ORDINATES, type=int,
                        help='the number of discrete ordinates')
    parser.add_argument('-JMC', '--num_mc_zones',
                        default=NUM_MC_ZONES, type=int,
                        help='the number of zones for Monte Carlo solve')
    parser.add_argument('-JSN', '--num_sn_zones',
                        default=NUM_SN_ZONES, type=int,
                        help='the number of zones for SN solve')
    parser.add_argument('-JNN', '--num_nn_zones',
                        default=NUM_NN_ZONES, type=int,
                        help='the number of zones for SN solve')
    parser.add_argument('-H', '--zstop',
                        default=ZSTOP, type=int,
                        help='the length of the slab')
    parser.add_argument('-K', '--num_hidden_layer_nodes',
                        default=NUM_HIDDEN_LAYER_NODES, type=int,
                        help='number of nodes in neural net hidden layer')
    parser.add_argument('-eSN', '--epsilon_sn',
                        default=SN_EPSILON, type=float,
                        help='used for source iteration convergence criterion')
    parser.add_argument('-eNN', '--epsilon_nn',
                        default=NN_EPSILON, type=float,
                        help='used for neural net training convergence '
                             'criterion')
    parser.add_argument('-g', '--gamma_r',
                        default=GAMMA_R, type=float,
                        help='weight for neural net right boundary '
                             'correctness')
    parser.add_argument('-LR', '--learning_rate',
                        default=LEARNING_RATE, type=float,
                        help='the neural network learning rate')
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
    parser.add_argument('-Q', '--uniform_source_extent',
                        default=SOURCE,
                        nargs=2,
                        type=float,
                        help='extent (low, high) of uniform source')
    parser.add_argument('-P', '--point_source_location',
                        type=float,
                        help='location of optional point source')
    parser.add_argument('-QS', '--source_magnitude',
                        default=SOURCE_MAGNITUDE,
                        type=float,
                        help='magnitude of the uniform and/or point sources')
    parser.add_argument('-S', '--seed',
                        default=SEED, type=int,
                        help='the seed for the particle source')
    parser.add_argument('-M', '--max_num_segments',
                        default=MAX_NUM_SEGMENTS, type=int,
                        help='the maximum number of segments a particle can '
                             'travel')
    parser.add_argument('-Y', '--num_physical_particles',
                        default=NUM_PHYSICAL_PARTICLES,
                        type=float,
                        help='number of physical particles used in tally '
                             'normalization')
    parser.add_argument('-q', '--quiet',
                        action='store_true',
                        help='description of source')
    parser.add_argument('-s', '--show',
                        action='store_true',
                        help='show plots')
    parser.add_argument('-o', '--output_directory',
                        type=str,
                        help='directory in which to write output')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='verbose output')
    parser.add_argument('--skip_nn',
                        action='store_true',
                        help='do not run neural network')

    if input_args:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args
