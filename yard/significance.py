"""
Tests for assessing the statistical significance of differences between
the AUC for ROC curves.
"""

from yard.curve import ROCCurve
from yard.mathematics import geometric

__author__ = "Tamas Nepusz"
__email__ = "tamas@cs.rhul.ac.uk"
__copyright__ = "Copyright (c) 2010, Tamas Nepusz"
__license__ = "MIT"


class SignificanceTest(object):
    """Abstract class that defines the interface of significance
    tests.
    """

    def __init__(self, curve_factory=ROCCurve):
        """Creates a significance test using the given curve type.
        `curve_factory` must be a class name or a factory method that can
        accept a `BinaryClassifierData` instance and produce an instance
        of `Curve`. The produced `Curve` instance must also have an
        ``auc_from_pos_ranks`` method.
        """
        if not hasattr(curve_factory, "__call__"):
            raise TypeError("curve_factory must be callable")
        if not hasattr(curve_factory, "auc_from_pos_ranks"):
            raise TypeError("curve_factory must have an auc_from_pos_ranks " "method")

        self.curve_factory = curve_factory

    def test(self, data1, data2):
        """Tests whether the AUC scores of two ROC curves are significantly
        different or not. `data1` and `data2` must be instances of
        `yard.data.BinaryClassifierData`. Returns the observed difference
        in the AUC scores and the p-value."""
        raise NotImplementedError


class PairedPermutationTest(SignificanceTest):
    """Class implementing a paired permutation significance test for
    binary classifier curves.

    Testing is done by first calculating the rank scores of positive instances
    in each dataset. Each pair is then flipped with probability 0.5 to obtain a
    new rank configuration, and the AUC scores are calculated for this new
    configuration. This is repeated a given number of times (see
    `self.num_repetitions`), and the differences between the AUC scores for the
    two datasets are calculated for each permutation.  It is then counted how
    many times did the difference exceed the actual observed difference
    calculated from the original curves. This ratio serves as an estimate for
    the p-value.
    """

    def __init__(self, *args, **kwds):
        if "num_repetitions" in kwds:
            self.num_repetitions = int(kwds["num_repetitions"])
            del kwds["num_repetitions"]
        else:
            self.num_repetitions = 1000
        super(PairedPermutationTest, self).__init__(*args, **kwds)

    def test(self, data1, data2):
        """Tests whether the AUC scores of two ROC curves are significantly
        different or not. `data1` and `data2` must be instances of
        `yard.data.BinaryClassifierData`. Returns the observed difference
        in the AUC scores and the p-value.

        It is assumed that `data1` and `data2` contain the same examples with
        different scures, and it is not checked whether this is true or not.
        """
        n = len(data1)
        if len(data2) != n:
            raise ValueError("the two datasets must be equal in length")

        ranks1 = data1.get_positive_ranks()
        ranks2 = data2.get_positive_ranks()
        m = len(ranks1)
        if m != len(ranks2):
            raise ValueError("the two datasets must have the same " "positive examples")

        dummy_curve = self.curve_factory([])
        auc_from_ranks = dummy_curve.auc_from_pos_ranks
        observed_diff = auc_from_ranks(ranks1, n) - auc_from_ranks(ranks2, n)
        abs_observed_diff = abs(observed_diff)
        num_success = 0

        for trial in range(self.num_repetitions):
            idx = 0
            while True:
                idx += geometric(p=0.01)
                if idx >= m:
                    break
                ranks1[idx], ranks2[idx] = ranks2[idx], ranks1[idx]
            diff = abs(auc_from_ranks(ranks1, n) - auc_from_ranks(ranks2, n))
            if diff >= abs_observed_diff:
                num_success += 1

        return observed_diff, num_success / float(self.num_repetitions)
