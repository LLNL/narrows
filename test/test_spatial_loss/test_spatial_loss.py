import os
import sys
MY_DIR = os.path.dirname(__file__)

sys.path.append(f'{MY_DIR}/..')
import analyze  # noqa: E402

sys.path.append(f'{MY_DIR}/../..')
import narrows  # noqa: E402

SHARP_SLAB_INPUT = f'{MY_DIR}/../sharp_slab.yaml'
SHARP_SLAB_OUTPUT = f'{MY_DIR}/sharp_slab'


def test_spatial_loss():
    '''Test that we can generate a flux plot which also shows the spatial
    distribution of the flux on the other vertical axis.'''

    argv = f'''
{SHARP_SLAB_INPUT}
-d out={SHARP_SLAB_OUTPUT}
-d epsilon=0.1
'''.split()
    narrows.main(argv)

    argv = f'''
{SHARP_SLAB_OUTPUT}
-q fluxloss
'''.split()
    analyze.main(argv)


def test_spatial_loss_movie():
    '''Test that we can generate N flux-and-loss plots.'''

    argv = f'''
{SHARP_SLAB_INPUT}
-d out={SHARP_SLAB_OUTPUT}_history
-d epsilon=0.1
-d ahistory=True
-d hinterval=100
'''.split()
    narrows.main(argv)

    argv = f'''
{SHARP_SLAB_OUTPUT}_history
-m
'''.split()
    analyze.main(argv)
