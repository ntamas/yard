#!/usr/bin/env python
"""\
YARD -- Yet Another ROC Drawer
==============================

This is yet another Python package for drawing ROC curves. It also
lets you draw precision-recall and accumulation curves with both
untransformed and transformed X axes and calculate AUC (area under
curve) statistics. The significance of differences between AUC
scores can also be tested.
"""

from yard.data import *
from yard.curve import *

__author__  = "Tamas Nepusz"
__email__   = "tamas@cs.rhul.ac.uk"
__copyright__ = "Copyright (c) 2010, Tamas Nepusz"
__license__ = "MIT"
