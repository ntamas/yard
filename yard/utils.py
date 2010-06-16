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

