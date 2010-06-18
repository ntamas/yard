"""Utility classes that do not fit elsewhere"""

__author__  = "Tamas Nepusz"
__email__   = "tamas@cs.rhul.ac.uk"
__copyright__ = "Copyright (c) 2010, Tamas Nepusz"
__license__ = "MIT"

def axis_label(label):
    """Creates a decorator that attaches an attribute named ``__axis_label__``
    to a function. This is used later in the plotting functions to derive an
    appropriate axis label if the function is plotted on an axis.

    Usage::

        @axis_label("x squared")
        def f(x):
            return x ** 2
    """
    def result(func):
        func.__axis_label__ = label
        return func
    return result


def rank(vector, ties = True):
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
        return [rank+1 for rank in sorted(range(n), key=vector.__getitem__)]

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
            for idx2 in xrange(idx-dup_counter, idx):
                ranks[order[idx2]] = avg_rank

        prev_value, sum_ranks, dup_counter = value, idx, 1

    if dup_counter:
        avg_rank = sum_ranks / float(dup_counter) + 1
        for idx2 in xrange(n-dup_counter, n):
            ranks[order[idx2]] = avg_rank

    return ranks
