'''
This test generates the flux plots for mc21.
'''

from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import yaml
MY_DIR = os.path.dirname(__file__)

sys.path.append(f'{MY_DIR}/..')
import analyze  # noqa: E402

sys.path.append(f'{MY_DIR}/../..')
import narrows  # noqa: E402

# Switch to True to generate the actual mc21 plots, which takes 5 minutes
# instead of 5 seconds.
REAL_MC21_PLOTS = False

if REAL_MC21_PLOTS:
    name2iter = {
        'full_slab': 30907,
        'half_slab': 24256,
        'reedp1': 16581,
        'ss0': 12062,
        'ss1': 15836,
        'ss5': 21067,
    }
else:
    # Converge quickly.
    name2iter = {
        'full_slab': 10,
        'half_slab': 10,
        'reedp1': 10,
        'ss0': 10,
        'ss1': 10,
        'ss5': 10,
    }

name2fluxlimits = {
    'full_slab': (0.45, 1.05),
    'half_slab': (-0.1, 2.1),
    'reedp1': (-0.5, 2.2),
    'ss0': (0.45, 1.05),
    'ss1': (-0.1, 1.1),
    'ss5': (-0.1, 1.1),
}

name2legendloc = {
    'full_slab': 'center',
    'half_slab': 'upper right',
    'reedp1': 'upper left',
    'ss0': 'upper right',
    'ss1': 'upper right',
    'ss5': 'upper right',
}

LOSS_LIMITS = (1e-4, 8e3)


def run(name, deck, extra_options=None):
    max_num_iter = name2iter[name]
    argv = f'''
{deck}
-d out={MY_DIR}/{name}
-d max_num_iter={max_num_iter}
-d ahistory=True
-d hinterval=999999
-d interval=10000
-d sn=False
'''.split()

    if not REAL_MC21_PLOTS:
        # Converge quickly.
        argv.extend('-d num_particles=1000'.split())

    if extra_options:
        argv.extend(extra_options)

    narrows.main(argv)

    problem = f'{MY_DIR}/{name}'
    args = analyze.parse_args([problem])
    npzfile = np.load(f'{problem}.npz')

    plt.style.use(f'{MY_DIR}/mc21.mplstyle')

    analyze.plot_movie(args,
                       npzfile,
                       loss_limits=LOSS_LIMITS,
                       flux_limits=name2fluxlimits[name],
                       title='~',
                       legend_loc=name2legendloc[name])


def run_existing_input(name, extra_options=None):
    deck = f'{MY_DIR}/../{name}.yaml'
    run(name, deck, extra_options)


def test_mc21_p1():
    name = 'full_slab'
    run_existing_input(name)


def test_mc21_p2():
    name = 'half_slab'
    extra_options = '''
-d max_num_segments=1000
'''.split()
    run_existing_input(name, extra_options)


def test_mc21_p3():
    name = 'reedp1'
    extra_options = '''
-d cells_per_region=100
-d mc=True
-d max_num_segments=1000
'''.split()
    run_existing_input(name, extra_options)


def run_sharp_slab(sigma_a):
    with open(f'{MY_DIR}/../sharp_slab.yaml') as f:
        yamlinput = yaml.full_load(f)
    yamlinput['mat']['mat2']['sigma_a'] = sigma_a

    name = f'ss{sigma_a}'
    deck = f'{MY_DIR}/{name}.yaml'
    with open(deck, 'w') as f:
        yaml.dump(yamlinput, f)
    run(name, deck)


def test_mc21_p4():
    run_sharp_slab(0)


def test_mc21_p5():
    run_sharp_slab(1)


def test_mc21_p6():
    run_sharp_slab(5)
