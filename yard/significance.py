"""
Tests for assessing the statistical significance of differences between
the AUC for ROC curves.
"""

from yard.curve import ROCCurve
from yard.mathematics import geometric

class SignificanceTest(object):
    """Abstract class that defines the interface of significance
    tests.
    """

    def test(self, data1, data2):
        """Tests whether the AUC scores of two ROC curves are significantly
        different or not. `data1` and `data2` must be instances of
        `yard.data.BinaryClassifierData`. Returns the observed difference
        in the AUC scores and the p-value."""
        raise NotImplementedError


class PairedPermutationTest(object):
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

    def __init__(self):
        self.num_repetitions = 1000

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
            raise ValueError("the two datasets must have the same "
                             "positive examples")


        auc_from_ranks = ROCCurve.auc_from_pos_ranks
        observed_diff = auc_from_ranks(ranks1, n) - auc_from_ranks(ranks2, n)
        abs_observed_diff = abs(observed_diff)
        num_success = 0

        for trial in xrange(self.num_repetitions):
            idx = 0
            while True:
                idx += geometric(p=0.5)
                if idx >= m:
                    break
                ranks1[idx], ranks2[idx] = ranks2[idx], ranks1[idx]
            diff = abs(auc_from_ranks(ranks1, n) - auc_from_ranks(ranks2, n))
            if diff >= abs_observed_diff:
                num_success += 1

        return observed_diff, num_success / float(self.num_repetitions)

