Narrows
-------
Narrows solves the discrete ordinates transport equation using a
neural network.

To install and run narrows:

    $ git clone https://github.com/llnl/narrows.git
    $ virtualenv -p python3.8 venv
    $ ./venv/bin/pip install -r narrows/requirements.txt
    $ cd narrows/test
    $ ../../venv/bin/python ../bin/narrows full_slab.yaml
    $ ../../venv/bin/python ./analyze.py full_slab -s

Contributing
------------
Narrows is a small, open source, research project. Questions, discussion,
and contributions are welcome. Contributions may include bugfixes,
documentation, or even new features. Please submit a
[pull request](https://help.github.com/articles/using-pull-requests/)
with ``develop`` as the destination branch in the
[Narrows repository](https://github.com/llnl/narrows).


Authors
-------
Narrows was created by Mike Pozulp (pozulp1@llnl.gov), Kyle Bilton, and Patrick Brantley.


### Citing Narrows

If you are referencing Narrows in a publication, please cite the following
paper:

 * Michael M. Pozulp.
   [**1D Transport Using Neural Nets, SN, and MC**](http://mike.pozulp.com/2019nnPaper.pdf).
   In *Proceedings of M&C 2019*, 876-885. Portland, Oregon, August 25-29, 2019. LLNL-CONF-772639.

License
-------
Narrows is distributed under the terms of the MIT license.
All new contributions must be made under the MIT license.
See [LICENSE](https://github.com/llnl/narrows/blob/master/LICENSE)
for details.

SPDX-License-Identifier: MIT

LLNL-CODE-806068
