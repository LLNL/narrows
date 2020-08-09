#!/usr/bin/env python

import argparse
import os
import sys
MY_DIR = os.path.dirname(__file__)
sys.path.append(f'{MY_DIR}/..')

import narrows  # noqa: E402

sys.path.append(f'{MY_DIR}')
import analytic_full_slab  # noqa: E402
import analyze  # noqa: E402
import nnVsn  # noqa: E402

FULL_SLAB = f'{MY_DIR}/full_slab'
HALF_SLAB = f'{MY_DIR}/half_slab'


def run_full_slab(epsilon):
    '''Run full_slab and generate plots'''

    print('Running full_slab')
    argv = f'''
{FULL_SLAB}.yaml
-d epsilon={epsilon}
-d verb=terse
'''.split()
    narrows.main(argv)

    print('Generating full_slab analytic flux plot')
    analytic_full_slab.main([])

    print('Generating full_slab relative error plot')
    argv = f'''
{FULL_SLAB}
-q re
'''.split()
    analyze.main(argv)

    print('Generating full_slab loss plot')
    argv = f'''
{FULL_SLAB}
-q loss
'''.split()
    analyze.main(argv)

    print('Printing full_slab runtime')
    argv = f'''
{FULL_SLAB}
-q time
'''.split()
    analyze.main(argv)


def run_scaling_study(epsilon):
    '''Run scaling study and generate plots'''

    print('Running scaling study')
    argv = f'''
--run
--epsilon={epsilon}
'''.split()
    nnVsn.main(argv)

    print('Analyzing scaling study')
    nnVsn.main([])


def run_half_slab(epsilon):
    '''Run half_slab and generate flux plot'''

    print('Running half_slab')
    argv = f'''
{HALF_SLAB}.yaml
-d epsilon={epsilon}
-d verb=terse
'''.split()
    narrows.main(argv)

    print('Generating half_slab flux plot')
    argv = f'''
{HALF_SLAB}
-q flux
'''.split()
    analyze.main(argv)

    print('Printing half_slab runtime')
    argv = f'''
{HALF_SLAB}
-q time
'''.split()
    analyze.main(argv)


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
                description='Run calculations and generate plots for M&C19',
                formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-e', '--epsilon',
                        default=0.1,
                        help='convergence criterion epsilon')
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    run_full_slab(args.epsilon)
    run_scaling_study(args.epsilon)
    run_half_slab(args.epsilon)


if __name__ == '__main__':
    main()
