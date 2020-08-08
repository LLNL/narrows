import sys


VERBOSITY = None
OUTPUT_FILE = None
INITIALIZED = False

verb2int = {'none': 0,
            'terse': 1,
            'moderate': 2,
            'verbose': 3}


def write(verb, *args, error=False, file_only=False):
    assert INITIALIZED, 'Cannot print before calling initialize'

    if VERBOSITY >= verb2int[verb]:
        stream = sys.stderr if error else sys.stdout
        if not file_only:
            print(*args, file=stream)
        print(*args, file=OUTPUT_FILE)


def initialize(verb, output_file):
    global VERBOSITY, OUTPUT_FILE, INITIALIZED
    VERBOSITY = verb2int[verb]
    OUTPUT_FILE = open(f'{output_file}.out', 'w')
    INITIALIZED = True


def close():
    OUTPUT_FILE.close()
