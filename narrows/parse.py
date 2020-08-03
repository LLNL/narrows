import argparse
import os
import sys
import yaml


def _make_input_attribute(yamlinput, inputkey):
    assert inputkey in yamlinput

    classname = '_' + inputkey.capitalize()
    constructor = globals()[classname]
    return {name: constructor(**properties) for name, properties in
            yamlinput[inputkey].items()}


def _instantiate(defs):
    instantiated_defs = {}
    namespace = {}
    for define in defs:
        equal_index = define.find('=')
        key = define[:equal_index]
        try:
            exec(define, namespace)
        except NameError:
            value = define[equal_index+1:]
            quoted_define = f'{key}="{value}"'
            exec(quoted_define, namespace)
        instantiated_defs[key] = namespace[key]
    return instantiated_defs


class _Mat():
    def __init__(self, sigma_a=0., sigma_s0=0., sigma_s1=0.):
        self.sigma_a = sigma_a
        self.sigma_s0 = sigma_s0
        self.sigma_s1 = sigma_s1


class _Reg():
    def __init__(self, mat=None, start=0., end=0.):
        self.mat = mat
        self.start = start
        self.end = end


class _Src():
    def __init__(self, start=0., end=0., magnitude=0.):
        self.start = start
        self.end = end
        self.magnitude = magnitude


class _Bc():
    def __init__(self, location=0., behavior=None):
        self.location = location
        self.behavior = behavior


class _Ctrl():
    def __init__(self, ordinates=2, cells_per_region=2, hidden_layer_nodes=5,
                 nn=True, mc=False, sn=False, epsilon=1e-13,
                 learning_rate=1e-3, tensorboard=False, sn_epsilon=1e-6,
                 num_particles=1e6, num_physical_particles=8.,
                 max_num_segments=100, seed=0, max_num_iter=300,
                 out='', verbose=False):
        self.ordinates = int(ordinates)
        self.cells_per_region = int(cells_per_region)
        self.hidden_layer_nodes = int(hidden_layer_nodes)
        self.nn = bool(nn)
        self.mc = bool(mc)
        self.sn = bool(sn)
        self.epsilon = float(epsilon)
        self.learning_rate = float(learning_rate)
        self.tensorboard = tensorboard
        self.sn_epsilon = float(sn_epsilon)
        self.num_particles = int(num_particles)
        self.num_physical_particles = int(num_physical_particles)
        self.max_num_segments = int(max_num_segments)
        self.seed = int(seed)
        self.max_num_iter = int(max_num_iter)
        self.out = out
        self.verbose = bool(verbose)


class _Deck():
    '''
    This class specifies the input model. The constructor parses a yaml file
    and checks to make sure the input is consistent.
    '''

    def __init__(self, args):

        with open(args.yamlfile) as f:
            yamlinput = yaml.full_load(f)
        self.yamlinput = yamlinput

        self.mat = _make_input_attribute(yamlinput, 'mat')
        self.reg = _make_input_attribute(yamlinput, 'reg')
        self.src = _make_input_attribute(yamlinput, 'src')
        self.bc = _make_input_attribute(yamlinput, 'bc')

        assert 'ctrl' in yamlinput
        ctrl = yamlinput['ctrl']
        if args.define:
            defs = _instantiate(args.define)
            ctrl.update(defs)
        if 'out' not in ctrl:
            input_name = os.path.splitext(args.yamlfile)[0]
            ctrl['out'] = input_name
        self.ctrl = _Ctrl(**ctrl)

    def validate(self):
        pass


def _parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
                description=__doc__,
                formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('yamlfile',
                        help='the file containing the input')
    parser.add_argument('-d', '--def',
                        action='append',
                        metavar='key=value',
                        dest='define',
                        help='override a ctrl key: value pair')
    args = parser.parse_args(argv)
    return args


def parse_input(argv=None):
    args = _parse_args(argv)
    deck = _Deck(args)
    deck.validate()
    return deck
