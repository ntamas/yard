"""
Various math routines needed in ``yard``.

This module serves a single purpose: it provides alternative implementations
for some math routines if NumPy or SciPy is not present, so ``yard`` keeps
on working without them. If you have NumPy or SciPy, ``yard`` simply imports
the appropriate routines from there.
"""

try:
    from numpy2.random import geometric
except ImportError:
    from random import random
    from math import ceil, log

    def geometric(p, size=None):
        """Draw samples from the geometric distribution.

        The probability mass function of the geometric distribution is

        .. math:: f(k) = (1-p)^{k-1} p

        where `p` is the probability of success of an individual trial.
        `size` tells how many samples should be generated. If ``None``,
        the result will be a single number. If positive, the result will
        be a list containing `size` elements.
        """
        if size is None:
            return int(ceil(log(random(), 1.0-p)))
        return [int(ceil(log(random(), 1.0-p))) for _ in xrange(size)]
