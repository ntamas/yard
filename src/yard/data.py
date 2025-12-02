"""\
Routines and classes for drawing ROC curves, calculating
sensitivity, specificity, precision, recall, TPR, FPR and such.
"""

from __future__ import division

from bisect import bisect_left

from yard.mathematics import rank
from yard.utils import axis_label

__author__ = "Tamas Nepusz"
__email__ = "tamas@cs.rhul.ac.uk"
__copyright__ = "Copyright (c) 2010, Tamas Nepusz"
__license__ = "MIT"


# pylint: disable-msg=C0103, E0202, E0102, E1101, R0913
class BinaryConfusionMatrix(object):
    """Class representing a binary confusion matrix.

    This class acts like a 2 x 2 matrix, it can also be indexed like
    that, but it also has some attributes to make the code using
    binary confusion matrices easier to read. These attributes are:

      - ``tp``: number of true positives
      - ``tn``: number of true negatives
      - ``fp``: number of false positives
      - ``fn``: number of false negatives
    """

    __slots__ = ("tp", "fp", "tn", "fn")

    def __init__(self, data=None, tp=0, fp=0, fn=0, tn=0):
        self.tp, self.fp, self.fn, self.tn = tp, fp, fn, tn
        if data:
            self.data = data

    @property
    def data(self):
        """Returns the data stored by this confusion matrix"""
        return [[self.tn, self.fn], [self.fp, self.tp]]

    @data.setter
    def data(self, data):
        """Sets the data stored by this confusion matrix"""
        if isinstance(data, BinaryConfusionMatrix):
            self.data = data.data
            return

        if len(data) != 2:
            raise ValueError("confusion matrix must have 2 rows")
        if any(len(row) != 2 for row in data):
            raise ValueError("confusion matrix must have 2 columns")
        (self.tn, self.fn), (self.fp, self.tp) = data

    @axis_label("Accuracy")
    def accuracy(self):
        """Returns the accuracy, i.e. (TP+TN) / (P+N).

        Example::

            >>> matrix = BinaryConfusionMatrix(tp=77, fp=77, fn=23, tn=23)
            >>> matrix.accuracy()
            0.5
        """
        num = self.tp + self.tn
        if num == 0:
            return 0
        den = num + self.fp + self.fn
        return num / den

    @axis_label("Error rate")
    def error_rate(self):
        """Returns the error rate, i.e. (FP+FN) / (P+N).

        Example::

            >>> matrix = BinaryConfusionMatrix(tp=77, fp=77, fn=23, tn=23)
            >>> matrix.error_rate()
            0.5
        """
        num = self.fp + self.fn
        if num == 0:
            return 0
        return num / (self.tp + self.fp + self.tn + self.fn)

    @axis_label("Fraction of data classified negative")
    def fdn(self):
        """Returns the fraction of data classified as negative (FDN)

        Example::

            >>> matrix = BinaryConfusionMatrix(tp=63, fp=28, fn=37, tn=72)
            >>> print round(matrix.fdn(), 6)
            0.545
        """
        num = self.fn + self.tn
        den = num + self.fp + self.tp
        return num / den

    @axis_label("Fraction of data classified positive")
    def fdp(self):
        """Returns the fraction of data classified as positive (FDP)

        Example::

            >>> matrix = BinaryConfusionMatrix(tp=63, fp=28, fn=37, tn=72)
            >>> print round(matrix.fdp(), 6)
            0.455
        """
        num = self.fp + self.tp
        den = num + self.fn + self.tn
        return num / den

    @axis_label("False discovery rate")
    def fdr(self):
        """Returns the false discovery date (FDR), also known as prediction
        conditioned fallout. It is defined as FP / (TP+FP).

        Example::

            >>> matrix = BinaryConfusionMatrix(tp=63, fp=28, fn=37, tn=72)
            >>> print round(matrix.fdr(), 6)
            0.307692
        """
        return self.fp / (self.fp + self.tp)

    @axis_label("False negative rate")
    def fnr(self):
        """Returns the false negative rate (FNR), i.e. FN / (FN + TP).

        Example::

            >>> matrix = BinaryConfusionMatrix(tp=63, fp=28, fn=37, tn=72)
            >>> print round(matrix.fnr(), 6)
            0.37
        """
        return self.fn / (self.fn + self.tp)

    @axis_label("False positive rate")
    def fpr(self):
        """Returns the false positive rate (FPR), i.e. FP / (FP + TN).

        Example::

            >>> matrix = BinaryConfusionMatrix(tp=63, fp=28, fn=37, tn=72)
            >>> print round(matrix.fpr(), 6)
            0.28
        """
        return self.fp / (self.fp + self.tn)

    @axis_label("F-score")
    def f_score(self, f=1.0):
        """Returns the F-score.

        The value of `f` controls the weighting between precision and recall
        in the F-score formula. `f` = 1 means that equal importance is attached
        to precision and recall. In general, recall is considered `f` times more
        important than precision.
        """
        sq = float(f * f)
        num = (1 + sq) * self.tp
        return num / (num + sq * self.fn + self.fp)

    @axis_label("Matthews correlation coefficient")
    def mcc(self):
        """Returns the Matthews correlation coefficient (also known as
        phi correlation coefficient)"""
        num = self.tp * self.tn - self.fp * self.fn
        den = self.tp + self.fp
        den *= self.tp + self.fn
        den *= self.tn + self.fp
        den *= self.tn + self.fn
        return num / (den ** 0.5)

    @axis_label("Negative predictive value")
    def npv(self):
        """Returns the negative predictive value (NPV), i.e. TN / (TN+FN).

        Example::

            >>> matrix = BinaryConfusionMatrix(tp=63, fp=28, fn=37, tn=72)
            >>> print round(matrix.npv(), 4)
            0.6606
        """
        return self.tn / (self.tn + self.fn)

    @axis_label("Odds ratio")
    def odds_ratio(self):
        """Returns the odds ratio.

        Example::

            >>> matrix = BinaryConfusionMatrix(tp=63, fp=28, fn=37, tn=72)
            >>> print round(matrix.odds_ratio(), 3)
            4.378
        """
        num = self.tp * self.tn
        den = self.fp * self.fn
        if den == 0:
            return float("nan") if num == 0 else float("inf")
        return num / den

    @axis_label("Precision")
    def precision(self):
        """Returns the precision, a.k.a. the positive predictive value (PPV), i.e.
        TP / (TP+FP)."""
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return 1.0

    @axis_label("Recall")
    def recall(self):
        """Returns the recall, a.k.a. the true positive rate (TPR) or sensitivity,
        i.e. TP / (TP+FN)."""
        return self.tp / (self.tp + self.fn)

    @axis_label("Rate of negative predictions")
    def rnp(self):
        """Returns the rate of negative predictions, i.e. (TN+FN) / (TN+FN+TP+FP)."""
        num = self.tn + self.fn
        if num == 0:
            return 0
        return num / (self.tp + self.fp + self.tn + self.fn)

    @axis_label("Rate of positive predictions")
    def rpp(self):
        """Returns the rate of positive predictions, i.e. (TP+FP) / (TN+FN+TP+FP)."""
        num = self.tp + self.fp
        if num == 0:
            return 0
        return num / (self.tp + self.fp + self.tn + self.fn)

    @axis_label("True negative rate")
    def tnr(self):
        """Returns the true negative rate (TNR), a.k.a. specificity"""
        return self.tn / (self.fp + self.tn)

    def __eq__(self, other):
        return (
            self.tp == other.tp
            and self.tn == other.tn
            and self.fp == other.fp
            and self.fn == other.fn
        )

    def __getitem__(self, coords):
        obs, exp = coords
        return self._data[obs][exp]

    def __hash__(self):
        return hash((self.tp, self.tn, self.fp, self.fn))

    def __repr__(self):
        return "%s(tp=%d, fp=%d, fn=%d, tn=%d)" % (
            self.__class__.__name__,
            self.tp,
            self.fp,
            self.fn,
            self.tn,
        )

    def __setitem__(self, coords, value):
        obs, exp = coords
        self._data[obs][exp] = value

    # Some aliases
    ppv = precision
    sensitivity = recall
    tpr = recall
    specificity = tnr
    fallout = fpr
    miss = fnr
    phi = mcc


class BinaryClassifierData(object):
    """Class representing the output of a binary classifier.

    The dataset must contain ``(x, y)`` pairs where `x` is a predicted
    value and `y` defines whether the example is positive or negative.
    When `y` is less than or equal to zero, it is considered a negative
    example, otherwise it is positive. ``False`` also means a negative
    and ``True`` also means a positive example.

    The class has an instance attribute called `title`, representing
    the title of the dataset. This title will be used in ROC curve
    plots in the legend. If the `title` is ``None``, the dataset will
    not appear in legends.
    """

    def __init__(self, data, title=None):
        self._title = None

        if isinstance(data, BinaryClassifierData):
            self.data = data.data
        else:
            self.data = sorted(self._normalize_point(point) for point in data)
        self.title = title
        self.total_positives = sum(point[1] > 0 for point in self.data)
        self.total_negatives = len(self.data) - self.total_positives

    def __getitem__(self, index):
        return tuple(self.data[index])

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _normalize_point(point):
        """Normalizes a data point by setting the second element
        (which tells whether the example is positive or negative)
        to either ``True`` or ``False``.

        Returns the new data point as a tuple."""
        return point[0], point[1] > 0

    def get_confusion_matrix(self, threshold):
        """Returns the confusion matrix at a given threshold.

        The outcome corresponding to values larger than or equal to the
        threshold is assumed to be 1 and the outcome correspondong to
        values smaller than the threshold is assumed to be zero.

        Example::

            >>> outcomes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            >>> expected = [0, 0, 0, 1, 0, 1, 1, 1, 1]
            >>> data = BinaryClassifierData(zip(outcomes, expected))
            >>> data.get_confusion_matrix(0.2)
            BinaryConfusionMatrix(tp=5, fp=3, fn=0, tn=1)
            >>> data.get_confusion_matrix(0.75)
            BinaryConfusionMatrix(tp=2, fp=0, fn=3, tn=4)
        """
        result = [[0, 0], [0, 0]]
        # Find the index in the data where the predictions start to
        # exceed the threshold
        idx = bisect_left(self.data, (threshold, False))
        if idx <= len(self.data) // 2:
            for _, is_pos in self.data[:idx]:
                result[0][is_pos] += 1
            result[1][0] = self.total_negatives - result[0][0]
            result[1][1] = self.total_positives - result[0][1]
        else:
            for _, is_pos in self.data[idx:]:
                result[1][is_pos] += 1
            result[0][0] = self.total_negatives - result[1][0]
            result[0][1] = self.total_positives - result[1][1]
        return BinaryConfusionMatrix(data=result)

    def get_negative_ranks(self):
        """Returns the ranks of the negative instances."""
        observations, exps = zip(*self.data)
        ranks = rank(observations)
        del observations
        return [ranks[idx] for idx, truth in enumerate(exps) if not truth]

    def get_positive_ranks(self):
        """Returns the ranks of the positive instances."""
        observations, exps = zip(*self.data)
        ranks = rank(observations)
        del observations
        return [ranks[idx] for idx, truth in enumerate(exps) if truth]

    def iter_confusion_matrices(self, thresholds=None):
        """Iterates over the possible prediction thresholds in the
        dataset and yields tuples containing the threshold and the
        corresponding confusion matrix. This method can be used to
        generate ROC curves and it is more efficient than getting
        the confusion matrices one by one.

        `thresholds` contains the thresholds for which we evaluate the
        confusion matrix. If it is ``None``, all possible thresholds
        from the dataset will be evaluated. If it is an integer `n`,
        we will choose `n+1` threshold levels equidistantly from
        the range `0-1` (so the thresholds divide the interval `0-1`
        to `n` equal intervals). If it is an iterable, then each member
        yielded by the iterable must be a threshold.

        Example::

            >>> outcomes = [10, 20, 20, 30, 40]
            >>> expected = [0, 1, 0, 0, 1]
            >>> data = BinaryClassifierData(zip(outcomes, expected))
            >>> thresholds = [20, 30, 40, 50]
            >>> l1 = list(data.iter_confusion_matrices([20, 30, 40, 50]))
            >>> l2 = [(x, data.get_confusion_matrix(x)) for x in thresholds]
            >>> l1 == l2
            True
        """
        if not len(self):
            return

        if thresholds is None:
            thresholds = [pred for pred, _ in self.data]
            thresholds.append(float("inf"))
        elif not hasattr(thresholds, "__iter__"):
            n = float(thresholds)
            thresholds = [i / n for i in range(thresholds + 1)]
        thresholds = sorted(set(thresholds))

        if not thresholds:
            return

        threshold = thresholds.pop(0)
        result = self.get_confusion_matrix(threshold)
        yield threshold, BinaryConfusionMatrix(result)

        n = len(self)
        row_idx = bisect_left(self.data, (threshold, False))
        for threshold in thresholds:
            while row_idx < n:
                row = self.data[row_idx]
                if row[0] >= threshold:
                    break
                if row[1]:
                    # This data point is a positive example. Since
                    # we are below the threshold now (and we weren't
                    # in the previous iteration), we have one less
                    # TP and one more FN
                    result.tp -= 1
                    result.fn += 1
                else:
                    # This data point is a negative example. Since
                    # we are below the threshold now (and we weren't
                    # in the previous iteration), we have one more
                    # TN and one less FP
                    result.tn += 1
                    result.fp -= 1
                row_idx += 1
            yield threshold, BinaryConfusionMatrix(result)

    @property
    def title(self):
        """The title of the plot"""
        return self._title

    @title.setter
    def title(self, value):
        """Sets the title of the plot"""
        try:
            types = (str, unicode)
        except:
            types = str

        if value is None or isinstance(value, types):
            self._title = value
        else:
            self._title = str(value)
