"""
Various 1D and 2D transformations that can be performed on curves.

These transformations are used to derive concentrated ROC (CROC),
concentrated AC (CAC) and concentrated PR (CPR) curves from
ordinary ROC, AC and PR curves.
"""

from math import exp

__author__  = "Tamas Nepusz"
__email__   = "tamas@cs.rhul.ac.uk"
__copyright__ = "Copyright (c) 2010, Tamas Nepusz"
__license__ = "MIT"

class ExponentialTransformation(object):
    """Exponential transformation object.

    This transformation transforms a given number `x` to
    `(1-exp(-alpha*x)) / (1-exp(-alpha))`, which corresponds
    to a magnification transformation on the region close to
    `x=0` while keeping `x=1` in-place.
    """

    def __init__(self, alpha = 7):
        """Constructs an exponential transformation with the given
        `alpha` value. The default `alpha`=7 maps 0.1 approximately
        to 0.5."""
        self.exp_minus_alpha = exp(-alpha)

    def __call__(self, x):
        """Transforms the given number `x` and returns
        `(1-exp(-alpha*x)) / (1-exp(-alpha))`."""
        return (1-self.exp_minus_alpha**x) / (1-self.exp_minus_alpha)

