"""\
Routines and classes for drawing ROC curves, calculating
sensitivity, specificity, precision, recall, TPR, FPR and such.
"""

from bisect import bisect_left
from itertools import izip

__author__  = "Tamas Nepusz"
__email__   = "tamas@cs.rhul.ac.uk"
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

    def accuracy(self):
        """Returns the accuracy"""
        num = self.tp + self.tn
        den = num + self.fp + self.fn
        return num / float(den)

    def fdn(self):
        """Returns the fraction of data classified as negative (FDP)"""
        num = self.fn + self.tn
        den = num + self.fp + self.tp
        return num / float(den)

    def fdp(self):
        """Returns the fraction of data classified as positive (FDP)"""
        num = self.fp + self.tp
        den = num + self.fn + self.tn
        return num / float(den)

    def fdr(self):
        """Returns the false discovery date (FDR)"""
        return self.fp / float(self.fp + self.tp)

    def fpr(self):
        """Returns the false positive rate (FPR)"""
        return self.fp / float(self.fp + self.tn)

    def f_score(self, f=1.0):
        """Returns the F-score"""
        sq = float(f*f)
        sq1 = 1+sq
        num = sq1 * self.tp
        return num / (num + sq * self.fn + self.fp)

    def mcc(self):
        """Returns the Matthews correlation coefficient"""
        num = self.tp * self.tn - self.fp * self.fn
        den = (self.tp + self.fp)
        den *= (self.tp + self.fn)
        den *= (self.tn + self.fp)
        den *= (self.tn + self.fn)
        return num / (den ** 0.5)

    def npv(self):
        """Returns the negative predictive value (NPV)"""
        return self.tn / float(self.tn + self.fn)

    def odds_ratio(self):
        """Returns the odds ratio"""
        num = self.tp * self.tn
        den = self.fp * self.fn
        if den == 0:
            return float('nan') if num == 0 else float('inf')
        return num / float(den)

    def precision(self):
        """Returns the precision, a.k.a. the positive predictive value (PPV)"""
        try:
            return self.tp / float(self.tp + self.fp)
        except ZeroDivisionError:
            return 1.0
    ppv = precision

    def sensitivity(self):
        """Returns the sensitivity, a.k.a. the true negative rate (TPR)"""
        return self.tp / float(self.tp + self.fn)
    tpr = sensitivity
    recall = sensitivity

    def specificity(self):
        """Returns the specificity, a.k.a. the true negative rate (TNR)"""
        return self.tn / float(self.fp + self.tn)
    tnr = specificity

    def __eq__(self, other):
        return self.tp == other.tp and self.tn == other.tn and \
               self.fp == other.fp and self.fn == other.fn

    def __getitem__(self, coords):
        obs, exp = coords
        return self._data[obs][exp]

    def __repr__(self):
        return "%s(tp=%d, fp=%d, fn=%d, tn=%d)" % \
                (self.__class__.__name__, self.tp, self.fp, self.fn, self.tn)

    def __setitem__(self, coords, value):
        obs, exp = coords
        self._data[obs][exp] = value


    
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
        self.total_positives = sum(point[1] > 0 for point in data)
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
        """Returns the confusion matrix at a given threshold"""
        result = [[0, 0], [0, 0]]
        # Find the index in the data where the predictions start to
        # exceed the threshold
        idx = bisect_left(self.data, (threshold, False))
        if idx <= len(self.data) / 2:
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

    def iter_confusion_matrices(self, thresholds=None):
        """Iterates over the possible prediction thresholds in the
        dataset and yields tuples containing the threshold and the
        corresponding confusion matrix. This method can be used to
        generate ROC curves and it is more efficient than getting
        the confusion matrices one by one.
        
        @param thresholds: the thresholds for which we evaluate the
          confusion matrix. If it is ``None``, all possible thresholds
          from the dataset will be evaluated. If it is an integer `n`,
          we will choose `n` threshold levels equidistantly from
          the range `0-1`. If it is an iterable, then each member
          yielded by the iterable must be a threshold."""
        if not len(self):
            return

        if thresholds is None:
            thresholds = [pred for pred, _ in self.data]
        elif not hasattr(thresholds, "__iter__"):
            n = float(thresholds)
            thresholds = [i/n for i in xrange(thresholds)]
            thresholds.append(1)
        thresholds = sorted(set(thresholds))

        if not thresholds:
            return

        thresholds.append(float('inf'))

        threshold = thresholds.pop(0)
        result = self.get_confusion_matrix(threshold)
        yield threshold, result

        row_idx, n = 0, len(self)
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
        if value is None or isinstance(value, (str, unicode)):
            self._title = value
        else:
            self._title = str(value)


class BinaryClassifierPerformanceCurve(object):
    """Class representing a broad class of binary classifier performance
    curves.

    By using this class diretly, you are free to specify what's on the X
    and Y axes of the plot. If you are interested in ROC curves, see
    `ROCCurve`, which is a subclass of this class. If you are interested
    in precision-recall curves, see `PrecisionRecallCurve`, which is also
    a subclass.
    """

    def __init__(self, data, x_axis, y_axis):
        """Constructs a binary classifier performance curve from the given
        dataset using the two given measures on the X and Y axes.

        The dataset must contain ``(x, y)`` pairs where `x` is a predicted
        value and `y` defines whether the example is positive or negative.
        When `y` is less than or equal to zero, it is considered a negative
        example, otherwise it is positive. ``False`` also means a negative
        and ``True`` also means a positive example. The dataset can also
        be an instance of :class:`BinaryClassifierData`.
        """
        self._data = None
        self._points = None
        self.x_method_name = x_axis
        self.y_method_name = y_axis
        self.data = data

    def auc(self):
        """Returns the area under the curve.
        
        The area is calculated using a trapezoidal approximation to make the
        AUC of the `ROCCurve` class relate to the Gini coefficient (where
        G1 + 1 = 2 * AUC).
        """
        points = self.get_points()
        auc = sum((y0+y1) / 2. * (x0-x1) \
                  for (x0, y0), (x1, y1) in izip(points, points[1:]))
        return auc

    def coarsen(self, **kwds):
        """Coarsens the curve.

        This method is useful before plotting a curve that consists
        of many data points that are potentially close to each other.
        The method of coarsening is defined by the keyword arguments
        passed to this function.

        There are two different coarsening methods. The first
        method is invoked as ``coarsen(every=k)`` (where `k` is an
        integer) and it will keep every `k`th point from the curve.
        You can also call ``coarsen(until=k)`` which will keep on
        removing points from the curve (approximately evenly) until
        only `k` points remain. If there are less than `k` points
        initially, the curve will not be changed.
        """

        # Note: we will always keep the first and the last element

        if "every" in kwds and "until" in kwds:
            raise TypeError("use either every=... or until=...")
        if "every" not in kwds and "until" not in kwds:
            raise TypeError("use either every=... or until=...")

        points = self.get_points()
        if not points:
            return

        if "every" in kwds:
            k = int(kwds["every"])
            self._points = points[::k]
            if len(points) % k != 0:
                self._points.append(points[-1])
            return

        k = int(kwds["until"])
        n = len(points)
        step = (n-1) / (k-1.)
        result = [points[int(idx*step)] for idx in xrange(1, k-1)]
        result.append(points[-1])
        self._points = result

    @property
    def data(self):
        """Returns the data points from which we generate the curve"""
        return self._data

    @data.setter
    def data(self, data):
        """Sets the data points from which we generate the curve."""
        if isinstance(data, BinaryClassifierData):
            self._data = data
        else:
            self._data = BinaryClassifierData(data)
        self._points = None

    def get_points(self):
        """Returns the points of the curve as a list of tuples.

        This method caches its results, so make sure you don't modify
        ``self.data`` after you have called this method or any other
        method that makes use of the points themselves (such as
        :func:`self.auc`)."""
        if self._points is None:
            meth_x = getattr(BinaryConfusionMatrix, self.x_method_name)
            meth_y = getattr(BinaryConfusionMatrix, self.y_method_name)
            self._points = [(meth_x(mat), meth_y(mat)) for _, mat in \
                    self._data.iter_confusion_matrices()]
        return self._points

    def get_empty_figure(self, *args, **kwds):
        """Returns an empty `matplotlib.Figure` that can be used
        to show the classifier curve. The arguments of this function are
        passed on intact to the constructor of `matplotlib.Figure`,
        except these (which are interpreted here):

            - `title`: the title of the figure.

            - `xlabel`: the label of the X axis. If omitted, we will
              try to infer it from `self.x_method_name`.

            - `ylabel`: the label of the Y axis. If omitted, we will
              try to infer it from `self.y_method_name`.

        These must be given as keyword arguments.
        """
        import matplotlib.pyplot as plt

        # Set up the dict mapping method names to labels
        known_labels = dict(\
                accuracy="Accuracy", \
                fpr="False positive rate", \
                f_score="F-score", \
                mcc="Matthews correlation coefficient", \
                npv="Negative predictive value", \
                ppv="Positive predictive value", \
                precision="Precision", \
                recall="Recall", \
                sensitivity="Sensitivity", \
                specificity="Specificity", \
                tpr="True positive rate"
        )

        # Process the input arguments
        if "xlabel" in kwds:
            xlabel = kwds["xlabel"]
            del kwds["xlabel"]
        else:
            xlabel = known_labels.get(self.x_method_name, self.x_method_name)

        if "ylabel" in kwds:
            ylabel = kwds["ylabel"]
            del kwds["ylabel"]
        else:
            ylabel = known_labels.get(self.y_method_name, self.y_method_name)

        if "title" in kwds:
            title = kwds["title"]
            del kwds["title"]
        else:
            title = None

        # Construct the figure
        fig = plt.figure(*args, **kwds)

        # Create the axes, set the axis labels
        axes = fig.add_subplot(111)
        if title:
            axes.set_title(title)
        if xlabel:
            axes.set_xlabel(xlabel)
        if ylabel:
            axes.set_ylabel(ylabel)
        axes.set_xbound(0.0, 1.0)
        axes.set_ybound(0.0, 1.0)

        return fig

    def get_figure(self, *args, **kwds):
        """Returns a `matplotlib.Figure` that shows the curve.
        The arguments of this function are passed on intact to
        `get_empty_figure()`, except the following which are
        interpreted here:
            
            - `legend`: whether we want a legend on the figure or not.
              If ``False``, no legend will be shown. If ``True``,
              `matplotlib` will try to place the legend in an
              optimal position. If an integer or string, it will be
              interpreted as a location code by `matplotlib`.
        """
        if "legend" in kwds:
            legend = kwds["legend"]
            del kwds["legend"]

        # Get an empty figure and its axes, and plot the curve on the axes
        fig = self.get_empty_figure(*args, **kwds)
        self.plot_on_axes(fig.get_axes()[0], legend=legend)
        return fig

    def plot_on_axes(self, axes, style='r-', legend=True):
        """Plots the curve on the given `matplotlib.Axes` object.
        `style` specifies the style of the curve using ordinary
        ``matplotlib`` conventions. `legend` specifies the position
        where the legend should be added. ``False`` or ``None``
        means no legend.
        """
        # Plot the points
        xs, ys = zip(*self.get_points())
        curve = axes.plot(xs, ys, style)

        # Create the legend
        if legend is True:
            legend = 0
        if legend is not None and legend is not False:
            label = self._data.title
            if label is not None:
                axes.legend(curve, (label, ), legend)

        return curve

    def show(self, *args, **kwds):
        """Constructs and shows a `matplotlib.Figure` that plots the
        curve. If you need the figure itself for further manipulations,
        call `get_figure()` instead of this method.

        The arguments of this function are passed on intact to
        `get_figure()`.
        """
        self.get_figure(*args, **kwds).show()


class ROCCurve(BinaryClassifierPerformanceCurve):
    """Class representing a ROC curve.
    
    A ROC curve plots the true positive rate on the Y axis versus
    the false positive rate on the X axis.
    """

    def __init__(self, data):
        """Constructs a ROC curve from the given dataset.

        The dataset must contain ``(x, y)`` pairs where `x` is a predicted
        value and `y` defines whether the example is positive or negative.
        When `y` is less than or equal to zero, it is considered a negative
        example, otherwise it is positive. ``False`` also means a negative
        and ``True`` also means a positive example. The dataset can also
        be an instance of `BinaryClassifierData`.
        """
        super(ROCCurve, self).__init__(data, "fpr", "tpr")

    def get_empty_figure(self, *args, **kwds):
        """Returns an empty `matplotlib.Figure` that can be used
        to show the ROC curve. The arguments of this function are
        passed on intact to the constructor of `matplotlib.Figure`,
        except these (which are interpreted here):
            
            - `title`: the title of the figure.

            - `xlabel`: the label of the X axis. If omitted, we will
              try to infer it from `self.x_method_name`.

            - `ylabel`: the label of the Y axis. If omitted, we will
              try to infer it from `self.y_method_name`.

            - `no_discrimination_line`: if ``True``, the no discrimination
              line will be drawn. If ``False``, it won't be drawn. If
              a string, it is interpreted as a line style by
              ``matplotlib`` and this line style will be used to draw
              the no discrimination line. If it is a tuple, the first
              element of the tuple will be interpreted as the color
              and the second will be interpreted as the line style
              by ``matplotlib``.

        These must be given as keyword arguments.
        """
        if "no_discrimination_line" in kwds:
            no_discrimination_line = kwds["no_discrimination_line"]
            del kwds["no_discrimination_line"]
        else:
            no_discrimination_line = ("#444444", ":")

        # Create the figure by calling the superclass
        fig = super(ROCCurve, self).get_empty_figure(*args, **kwds)
        axes = fig.get_axes()[0]

        # Plot the no-discrimination line
        if no_discrimination_line:
            if isinstance(no_discrimination_line, (tuple, list)):
                color, linestyle = no_discrimination_line
                axes.plot([0, 1], color=color, linestyle=linestyle)
            else:
                axes.plot([0, 1], no_discrimination_line)

        return fig


class PrecisionRecallCurve(BinaryClassifierPerformanceCurve):
    """Class representing a precision-recall curve.
    
    A precision-recall curve plots precision on the Y axis versus
    recall on the X axis.
    """

    def __init__(self, data):
        """Constructs a precision-recall curve from the given dataset.

        The dataset must contain ``(x, y)`` pairs where `x` is a predicted
        value and `y` defines whether the example is positive or negative.
        When `y` is less than or equal to zero, it is considered a negative
        example, otherwise it is positive. ``False`` also means a negative
        and ``True`` also means a positive example. The dataset can also
        be an instance of `BinaryClassifierData`.
        """
        super(PrecisionRecallCurve, self).__init__(data, "recall", "precision")


class AccumulationCurve(BinaryClassifierPerformanceCurve):
    """Class representing an accumulation curve.
    
    An accumulation curve plots the true positive rate on the Y axis
    versus the fraction of data classified as positive on the X axis.
    """

    def __init__(self, data):
        """Constructs an accumulation curve from the given dataset.

        The dataset must contain ``(x, y)`` pairs where `x` is a predicted
        value and `y` defines whether the example is positive or negative.
        When `y` is less than or equal to zero, it is considered a negative
        example, otherwise it is positive. ``False`` also means a negative
        and ``True`` also means a positive example. The dataset can also
        be an instance of `BinaryClassifierData`.
        """
        super(AccumulationCurve, self).__init__(data, "fdp", "tpr")
