"""
Various math routines needed in ``yard``.

This module serves a single purpose: it provides alternative implementations
for some math routines if NumPy or SciPy is not present, so ``yard`` keeps
on working without them. If you have NumPy or SciPy, ``yard`` simply imports
the appropriate routines from there.
"""

__author__ = "Tamas Nepusz"
__email__ = "tamas@cs.rhul.ac.uk"
__copyright__ = "Copyright (c) 2010, Tamas Nepusz"
__license__ = "MIT"

from yard.utils import vectorized

#############################################################################

try:
    from numpy.random import geometric
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
            return int(ceil(log(random(), 1.0 - p)))
        return [int(ceil(log(random(), 1.0 - p))) for _ in range(size)]


#############################################################################

try:
    from numpy import log
except ImportError:
    from math import log as math_log

    def _safelog(item):
        """Takes the logarithm of `item`. Instead of raising exceptions
        like `math.log` does, it returns ``-inf`` for zero and ``nan``
        for negative numbers.
        """
        if item < 0:
            return float("nan")
        if item == 0:
            return float("-inf")
        return math_log(item)

    log = vectorized(_safelog)

#############################################################################

try:
    from numpy import power
except ImportError:

    def power(item, exponent):
        """Raises `item` to the given `exponent` and returns the result.
        `item` or `exponent` (but not both) may also be an iterable. In
        this case, the result will be a list, and exponentiation will be
        done elementwise."""
        if hasattr(item, "__iter__"):
            return [i ** exponent for i in item]
        if hasattr(exponent, "__iter__"):
            return [item ** i for i in exponent]
        return item ** exponent


#############################################################################

try:
    from scipy.stats import rankdata as rank
except ImportError:

    def rank(vector, ties=True):
        """Returns the rank vector of a given vector. `ties` specifies
        whether we want to account for ties or not.

        Examples::

            >>> rank([5, 6, 7, 8])
            [1.0, 2.0, 3.0, 4.0]
            >>> rank([5, 7, 6, 8])
            [1.0, 3.0, 2.0, 4.0]
            >>> rank([5, 5, 7, 8])
            [1.5, 1.5, 3.0, 4.0]
            >>> rank([5, 6, 5, 6])
            [1.5, 3.5, 1.5, 3.5]
            >>> rank([5, 6, 5, 6], ties=False)
            [1, 3, 2, 4]
        """
        n = len(vector)
        if not ties:
            return [rank + 1 for rank in sorted(range(n), key=vector.__getitem__)]

        values, order = zip(*sorted((value, idx) for idx, value in enumerate(vector)))
        ranks = [0] * n

        prev_value, sum_ranks, dup_counter = None, 0, 0
        for idx, value in enumerate(values):
            if value == prev_value:
                sum_ranks += idx
                dup_counter += 1
                continue

            if dup_counter:
                avg_rank = sum_ranks / float(dup_counter) + 1
                for idx2 in range(idx - dup_counter, idx):
                    ranks[order[idx2]] = avg_rank

            prev_value, sum_ranks, dup_counter = value, idx, 1

        if dup_counter:
            avg_rank = sum_ranks / float(dup_counter) + 1
            for idx2 in range(n - dup_counter, n):
                ranks[order[idx2]] = avg_rank

        return ranks
