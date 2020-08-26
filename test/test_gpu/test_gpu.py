import os
import pytest
import sys
import torch
MY_DIR = os.path.dirname(__file__)

sys.path.append(f'{MY_DIR}/../..')
import narrows  # noqa: E402

FULL_SLAB_INPUT = f'{MY_DIR}/../full_slab.yaml'
FULL_SLAB_OUTPUT = f'{MY_DIR}/full_slab'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires gpu')
def test_gpu():
    '''Test that we can train the neural network on a gpu.'''

    argv = f'''
{FULL_SLAB_INPUT}
-d gpu=True
-d out={FULL_SLAB_OUTPUT}_gpu
-d epsilon=0.1
'''.split()
    narrows.main(argv)

    with open(f'{FULL_SLAB_OUTPUT}_gpu.out') as f:
        output = f.read()
    assert 'Skipping gpu' not in output
