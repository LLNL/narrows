'''
This test checks that the flux calculated for the full slab input has not
changed with respect to a baseline.
'''

import importlib
import numpy as np
import os
import pytest
import shutil
import sys
MY_DIR = os.path.dirname(__file__)
sys.path.append(f'{MY_DIR}/../..')

import narrows  # noqa: E402

sys.path.append(f'{MY_DIR}/..')
import analytic_full_slab  # noqa: E402

ALGORITHMS = ['sn', 'mc', 'nn', 'an']

FULL_SLAB = f'{MY_DIR}/../full_slab'

TESTS = '''
skip_nn
analytic
nn
tensorboard
'''.split()

RTOL = 1.9e-07


def assert_flux_equal(baseline_npzfile, npzfile,
                      algorithms=ALGORITHMS):
    for algorithm in algorithms:
        baseline_flux = baseline_npzfile[f'{algorithm}_flux']
        new_flux = npzfile[f'{algorithm}_flux']
        np.testing.assert_allclose(baseline_flux, new_flux, rtol=RTOL)


def load_npzfiles(problem_name):
    baseline = np.load(f'{MY_DIR}/{problem_name}.baseline.npz')
    new = np.load(f'{MY_DIR}/{problem_name}.npz')
    return baseline, new


def run_test(function, algorithms=ALGORITHMS):
    os.chdir(MY_DIR)
    problem_name = function()
    npzfiles = load_npzfiles(problem_name)
    assert_flux_equal(*npzfiles, algorithms=algorithms)


def baseline(test, skip_out_file=False):
    function = globals()[test]
    problem = function()

    # Baseline .npz file
    src = f'{MY_DIR}/{problem}.npz'
    dest = f'{MY_DIR}/{problem}.baseline.npz'
    shutil.copyfile(src, dest)

    if not skip_out_file:
        # Baseline .out file
        src = f'{MY_DIR}/{problem}.out'
        dest = f'{MY_DIR}/{problem}.baseline.out'
        shutil.copyfile(src, dest)


def baseline_all():
    for test in TESTS:
        if test == 'analytic':
            baseline(test, skip_out_file=True)
        else:
            baseline(test)


def skip_nn():
    problem = 'fs_skip_nn'
    argv = f'''
{FULL_SLAB}.yaml
-d nn=False
-d out={problem}
'''.split()
    narrows.main(argv)
    return problem


def test_skip_nn():
    run_test(skip_nn, algorithms=['sn', 'mc'])


def analytic():
    problem = 'fs_analytic'
    edge, src_mag, sigma_t, zstop = \
        analytic_full_slab.get_parameters_for(FULL_SLAB)
    flux = analytic_full_slab.solution(edge, src_mag, sigma_t, zstop)

    np.savez(f'{MY_DIR}/{problem}.npz',
             edge=edge,
             an_flux=flux)

    return problem


def test_analytic():
    run_test(analytic, algorithms=['an'])


def get_deck_and_mesh(argv):
    deck = narrows.parse_input(argv)
    mesh = narrows.create_mesh(deck)
    return deck, mesh


def nn():
    problem = 'fs_nn'
    argv = f'''
{FULL_SLAB}.yaml
-d out={problem}
-d epsilon=0.1
'''.split()
    narrows.main(argv)
    return problem


def test_nn():
    run_test(nn, algorithms=['nn'])


def tensorboard():
    problem = 'fs_tensorboard'
    argv = f'''
{FULL_SLAB}.yaml
-d out={problem}
-d epsilon=0.1
-d tensorboard=True
'''.split()
    narrows.main(argv)
    return problem


@pytest.mark.skipif(importlib.util.find_spec('torch.utils.tensorboard')
                    is None,
                    reason='tensorboard is not installed')
def test_tensorboard():
    run_test(tensorboard, algorithms=['nn'])
    assert os.path.exists('runs')
