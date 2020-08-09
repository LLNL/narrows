'''
This test checks that genplots runs without error.
'''

import os
import sys
MY_DIR = os.path.dirname(__file__)
sys.path.append(f'{MY_DIR}/..')

from genplots import (  # noqa: E402
    run_full_slab,
    run_scaling_study,
    run_half_slab
)


# Use large epsilon to converge very quickly so tests run fast.
SLAB_EPSILON = 1e2
SCALING_EPSILON = 1e6


def test_run_full_slab():
    run_full_slab(SLAB_EPSILON)


def test_run_scaling_study():
    run_scaling_study(SCALING_EPSILON)


def test_run_half_slab():
    run_half_slab(SLAB_EPSILON)
