#!/usr/bin/env python
"""\
YARD - Yet Another ROC Drawer
=============================

This is yet another Python package for drawing ROC curves. It also
lets you draw precision-recall, accumulation and concentrated ROC
(CROC) curves and calculate the AUC (area under curve) statistics.
The significance of differences between AUC scores can also be
tested using paired permutation tests.

You may also be interested in CROC_, a similar package on the
Python Package Index that implements ROC curves. ``yard`` was developed
independently from CROC_, but several features of CROC have inspired
similar ones in ``yard``.

.. _CROC: http://pypi.python.org/pypi/CROC
"""

from yard.data import *
from yard.curve import *
from yard.version import __version__

__author__ = "Tamas Nepusz"
__email__ = "tamas@cs.rhul.ac.uk"
__copyright__ = "Copyright (c) 2010-2016, Tamas Nepusz"
__license__ = "MIT"
