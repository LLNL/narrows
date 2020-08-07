import os
from pathlib import Path
import re
import subprocess
import sys

import narrows


def get_dependency_versions():
    top_level = _get_top_level_of_repo()
    req_path = top_level / 'requirements.txt'

    if os.path.exists(req_path):
        with open(req_path) as f:
            lines = f.read()
        matches = re.finditer(r'([^>=<]+)[>=<].*\n', lines)

        dependencies = [m.group(1) for m in matches]
        names = map(lambda x: 'yaml' if x == 'pyyaml' else x, dependencies)
        versions = [__import__(n).__version__ for n in names]
        return zip(dependencies, versions)

    return None


def get_python_version():
    return '.'.join(str(x) for x in sys.version_info[:3])


def get_narrows_version():
    '''Get a descriptive version of this instance of Narrows.
    Function taken from Spack.

    If this is a git repository, and if it is not on a release tag,
    return a string like:

        release_version-commits_since_release-commit

    If we *are* at a release tag, or if this is not a git repo, return
    the real narrows release number (e.g., 0.0.1).
    '''

    top_level = _get_top_level_of_repo()
    git_path = top_level / '.git'
    if os.path.exists(git_path):
        desc, _, returncode = _git('describe', '--tags', fail_on_error=False)
        if returncode == 0:
            match = re.match(r'v([^-]+)-([^-]+)-g([a-f\d]+)', desc)
            if match:
                v, n, commit = match.groups()
                return f'{v}-{n}-{commit}'

    return narrows.__version__


def _get_top_level_of_repo():
    return Path(__file__).parent.parent


def _shell(cmd, cwd=None, fail_on_error=True):
    err_pipe = subprocess.PIPE

    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=err_pipe,
                         stdin=subprocess.PIPE,
                         universal_newlines=True)
    stdout, stderr = p.communicate()

    if stdout:
        stdout = stdout.rstrip('\r\n')
    if stderr:
        stderr = stderr.rstrip('\r\n')

    return stdout, stderr, p.returncode


def _git(*cmd, **kwargs):
    return _shell(['git'] + list(cmd), **kwargs)
