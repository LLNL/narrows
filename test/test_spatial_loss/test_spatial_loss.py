'''
This test checks that we can generate a flux plot which also shows the spatial
distribution of the flux on the other vertical axis.
'''

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
