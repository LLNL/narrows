import importlib
import numpy as np
import os
import pytest
import shutil
import sys
MY_DIR = os.path.dirname(__file__)
sys.path.append(f'{MY_DIR}/..')

import driver  # noqa: E402
import analyze  # noqa: E402


def get_args():
    zstop = 1
    argstr = (
     f'full_slab '
     f'--sigma_t 8 '
     f'--sigma_s0 0 '
     f'--sigma_s1 0 '
     f'--zstop {zstop} '
     f'--num_ordinates 4 '
     f'--num_hidden_layer_nodes 5 '
     f'--learning_rate 1e-3 '
     f'--epsilon_sn 1e-6 '
     f'--epsilon_nn 1e-13 '
     f'--num_sn_zones 50 '
     f'--num_mc_zones 50 '
     f'--num_nn_zones 50 '
     f'--num_particles 1000000 '
     f'--num_physical_particles 8 '
     f'--uniform_source_extent 0 {zstop} '
     f'--source_magnitude 8')
    args = driver.parse_args(argstr.split())
    return args


def assert_flux_equal(baseline_npzfile, npzfile,
                      algorithms=analyze.ALGORITHMS):
    for algorithm in algorithms:
        baseline_flux = baseline_npzfile[f'{algorithm}_flux']
        new_flux = npzfile[f'{algorithm}_flux']
        np.testing.assert_array_equal(baseline_flux, new_flux)


def load_npzfiles(problem_name):
    baseline = np.load(f'{MY_DIR}/{problem_name}.baseline.npz')
    new = np.load(f'{MY_DIR}/{problem_name}.npz')
    return baseline, new


def run_test(function, algorithms=analyze.ALGORITHMS):
    os.chdir(MY_DIR)
    args = function()
    npzfiles = load_npzfiles(args.problem_name)
    assert_flux_equal(*npzfiles, algorithms=algorithms)


def baseline(function):
    args = function()
    src = f'{MY_DIR}/{args.problem_name}.npz'
    dest = f'{MY_DIR}/{args.problem_name}.baseline.npz'
    shutil.copyfile(src, dest)


def skip_nn():
    args = get_args()
    args.problem_name = f'{args.problem_name}_skip_nn'
    args.skip_nn = True
    driver.run(args)
    return args


def test_skip_nn():
    run_test(skip_nn, algorithms=['sn', 'mc'])


def baseline_skip_nn():
    baseline(skip_nn)


def analytic():
    args = get_args()
    args.problem_name = f'{args.problem_name}_analytic'
    zone_edges = np.linspace(0, args.zstop, args.num_sn_zones + 1)
    flux = analyze.analytic_soln(zone_edges, args)

    np.savez(f'{MY_DIR}/{args.problem_name}.npz',
             an_zone_edges=zone_edges,
             an_flux=flux)

    return args


def test_analytic():
    run_test(analytic, algorithms=['an'])


def baseline_analytic():
    baseline(analytic)


def nn():
    args = get_args()
    args.problem_name = f'{args.problem_name}_nn'
    args.epsilon_nn = 0.1  # Converge really quickly to keep runtime down
    driver.run(args)
    return args


def test_nn():
    run_test(nn, algorithms=['nn'])


def baseline_nn():
    baseline(nn)


def tensorboard():
    args = get_args()
    args.problem_name = f'{args.problem_name}_tensorboard'
    args.epsilon_nn = 0.1  # Converge really quickly to keep runtime down
    args.tensorboard = True
    driver.run(args)
    return args


@pytest.mark.skipif(importlib.util.find_spec('torch.utils.tensorboard') is None,
                    reason='tensorboard is not installed')
def test_tensorboard():
    run_test(tensorboard, algorithms=['nn'])
    assert os.path.exists('runs')


def baseline_tensorboard():
    baseline(tensorboard)
