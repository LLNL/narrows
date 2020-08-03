import sys

from .parse import parse_input
from .mesh import create_mesh
from .driver import run


def _real_main(argv=None):
    deck = parse_input(argv)
    mesh = create_mesh(deck)
    return run(deck, mesh)


def main(argv=None):
    try:
        return _real_main(argv)
    except KeyboardInterrupt:
        sys.exit('\nERROR: Interrupted by user')
