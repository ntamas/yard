"""
Curve classes used in YARD.

This package contains implementations for all the curves YARD can plot.
At the time of writing, this includes:

    - ROC curves (`ROCCurve`)
    - CROC curves (`CROCCurve`)
    - Precision-recall curves (`PrecisionRecallCurve`)
    - Sensitivity-specificity plots (`SensitivitySpecificityCurve`)
    - Accumulation curves (`AccumulationCurve`)
    - F-score curves (`FScoreCurve`)
"""

__author__ = "Tamas Nepusz"
__email__ = "tamas@cs.rhul.ac.uk"
__copyright__ = "Copyright (c) 2010, Tamas Nepusz"
__license__ = "MIT"

from bisect import bisect
from yard.data import BinaryConfusionMatrix, BinaryClassifierData
from yard.transform import ExponentialTransformation
from yard.utils import axis_label, itersubclasses


class Curve(object):
    """Class representing an arbitrary curve on a 2D space.

    At this stage, a curve is nothing else but a series of points.
    """

    def __init__(self, points):
        """Constructs a curve with the given points. `points` must be
        an iterable of 2-tuples containing the coordinates of the points.
        """
        self._points = None
        self.points = points

    def auc(self):
        """Returns the area under the curve.

        The area is calculated using a trapezoidal approximation to make the
        AUC of the `ROCCurve` class relate to the Gini coefficient (where
        G1 + 1 = 2 * AUC).
        """
        points = self.points
        auc = sum(
            (y0 + y1) / 2.0 * (x1 - x0)
            for (x0, y0), (x1, y1) in zip(points, points[1:])
        )
        return auc

    def coarsen(self, **kwds):
        """Coarsens the curve in-place.

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

        points = self.points
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
        step = (n - 1) / (k - 1.0)
        result = [points[int(idx * step)] for idx in range(1, k - 1)]
        result.append(points[-1])
        self._points = result

    def get_empty_figure(self, *args, **kwds):
        """Returns an empty `matplotlib.Figure` that can be used to show the
        curve. The arguments of this function are passed on intact to the
        constructor of `matplotlib.Figure`, except these (which are interpreted
        here):

            - `title`: the title of the figure.
            - `xlabel`: the label of the X axis.
            - `ylabel`: the label of the Y axis.

        These must be given as keyword arguments.
        """
        import matplotlib.pyplot as plt

        # Extract the keyword arguments handled here
        kwds_extra = dict(xlabel=None, ylabel=None, title=None)
        for name in kwds_extra.keys():
            if name in kwds:
                kwds_extra[name] = kwds[name]
                del kwds[name]

        # Construct the figure
        fig = plt.figure(*args, **kwds)

        # Create the axes, set the axis labels and the plot title
        axes = fig.add_subplot(111)
        for name, value in kwds_extra.items():
            if value is not None:
                getattr(axes, "set_%s" % name)(value)

        # axes.set_xbound(0.0, 1.0)
        # axes.set_ybound(0.0, 1.0)
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
        else:
            legend = False

        # Get an empty figure and its axes, and plot the curve on the axes
        fig = self.get_empty_figure(*args, **kwds)
        self.plot_on_axes(fig.get_axes()[0], legend=legend)
        return fig

    def get_interpolated_point(self, x):
        """Returns an interpolated point on this curve at the given
        X position.

        The default implementation uses linear interpolation from the
        nearest two points.

        It is assumed that `self._points` is sorted in ascending order.
        If not, this function will produce wrong results.
        """
        points = self.points
        pos = bisect(points, (x, 0))

        # Do we have an exact match?
        try:
            if points[pos][0] == x:
                return points[pos]
        except IndexError:
            pass

        # Nope, so we have to interpolate
        if pos == 0:
            # Extrapolating instead
            (x1, y1), (x2, y2) = points[:2]
        elif pos == len(points):
            # Extrapolating instead
            (x1, y1), (x2, y2) = points[-2:]
        else:
            # Truly interpolating
            (x1, y1), (x2, y2) = points[pos - 1 : pos + 1]
        r = (x2 - x) / float(x2 - x1)
        return (x, y1 * r + y2 * (1 - r))

    def plot_on_axes(self, axes, style="r-", legend=True):
        """Plots the curve on the given `matplotlib.Axes` object.
        `style` specifies the style of the curve using ordinary
        ``matplotlib`` conventions. `legend` specifies the position
        where the legend should be added. ``False`` or ``None``
        means no legend.
        """
        # Plot the points
        xs, ys = zip(*self.points)
        (curve,) = axes.plot(xs, ys, style)

        # Create the legend
        if legend is True:
            legend = 0
        if legend is not None and legend is not False:
            label = self._data.title
            if label is not None:
                axes.legend(curve, (label,), legend)

        return curve

    @property
    def points(self):
        """Returns the points of this curve as a list of 2-tuples.

        The returned list is the same as the list used internally in
        the instance. Don't modify it unless you know what you're doing.
        """
        return self._points

    @points.setter
    def points(self, points):
        """Sets the points of this curve. The method makes a copy of the
        given iterable."""
        self._points = sorted(tuple(point) for point in points)

    def resample(self, new_xs):
        """Resamples the curve in-place at the given X positions.
        `xs` must be a list of positions on the X axis; interpolation
        will be used to calculate the corresponding Y values based on
        the nearest known values.
        """
        self._points = [self.get_interpolated_point(x) for x in new_xs]

    def show(self, *args, **kwds):
        """Constructs and shows a `matplotlib.Figure` that plots the
        curve. If you need the figure itself for further manipulations,
        call `get_figure()` instead of this method.

        The arguments of this function are passed on intact to
        `get_figure()`.
        """
        self.get_figure(*args, **kwds).show()

    def transform(self, transformation):
        """Transforms the curve in-place by sending all the points to a given
        callable one by one. The given callable must expect two real numbers
        and return the transformed point as a tuple."""
        self.points = [transformation(*point) for point in self._points]

    def transform_x(self, transformation):
        """Transforms the X axis of the curve in-place by sending all the
        points to a given callable one by one. The given callable must expect
        a single real number and return the transformed value."""
        self.points = [(transformation(x), y) for x, y in self._points]

    def transform_y(self, transformation):
        """Transforms the Y axis of the curve in-place by sending all the
        points to a given callable one by one. The given callable must expect
        a single real number and return the transformed value."""
        self._points = [(x, transformation(y)) for x, y in self._points]


class CurveFactory(object):
    """Factory class to construct `Curve` instances from short identifiers.

    Short identifiers for curve types are typically used in the command-line
    interface of `yard` to let the user specify which curve he/she wants to
    plot. This factory class interprets the short identifiers and constructs
    appropriate `Curve` instances.
    """

    @classmethod
    def construct_from_name(cls, name, *args, **kwds):
        """Constructs a curve from a short name used in command line arguments
        across the whole ``yard`` package.

        `name` is matched against the ``identifier`` class-level properties of
        all the subclasses of `Curve` to find the subclass to be constructed.
        All the remaining arguments are passed on intact to the constructor of
        the subclass. Returns a new instance of the found subclass, or raises
        ``ValueError`` if an invalid name was given.
        """
        return cls.find_class_by_name(name)(*args, **kwds)

    @staticmethod
    def find_class_by_name(name):
        """Finds the class corresponding to a given short name used in command
        line arguments across the whole ``yard`` package.

        `name` is matched against the ``identifier`` class-level properties of
        all the subclasses of `Curve` to find the subclass to be constructed.
        Returns the found subclass (not an instance of it), or raises
        ``ValueError`` if an invalid name was given.
        """
        name = name.lower()
        for cls in itersubclasses(Curve):
            if hasattr(cls, "identifier") and cls.identifier == name:
                return cls
        raise ValueError("no such curve type: %s" % name)

    @staticmethod
    def get_curve_names():
        return sorted(
            [
                cls.identifier
                for cls in itersubclasses(Curve)
                if hasattr(cls, "identifier")
            ]
        )


class BinaryClassifierPerformanceCurve(Curve):
    """Class representing a broad class of binary classifier performance
    curves.

    By using this class diretly, you are free to specify what's on the X
    and Y axes of the plot. If you are interested in ROC curves, see
    `ROCCurve`, which is a subclass of this class. If you are interested
    in precision-recall curves, see `PrecisionRecallCurve`, which is also
    a subclass. Accumulation curves are implemented in `AccumulationCurve`,
    sensitivity-specificity plots are in `SensitivitySpecificityCurve`
    etc.
    """

    def __init__(self, data, x_func, y_func):
        """Constructs a binary classifier performance curve from the given
        dataset using the two given measures on the X and Y axes.

        The dataset must contain ``(x, y)`` pairs where `x` is a predicted
        value and `y` defines whether the example is positive or negative.
        When `y` is less than or equal to zero, it is considered a negative
        example, otherwise it is positive. ``False`` also means a negative
        and ``True`` also means a positive example. The dataset can also
        be an instance of :class:`BinaryClassifierData`.

        `x_func` and `y_func` must either be unbound method instances of
        the `BinaryConfusionMatrix` class, or functions that accept
        `BinaryConfusionMatrix` instances as their only arguments and
        return a number.
        """
        self._data = None
        self._points = None
        self.x_func = x_func
        self.y_func = y_func

        if not hasattr(self.x_func, "__call__"):
            raise TypeError("x_func must be callable")
        if not hasattr(self.y_func, "__call__"):
            raise TypeError("y_func must be callable")

        self.data = data

    def _calculate_points(self):
        """Returns the actual points of the curve as a list of tuples."""
        x_func, y_func = self.x_func, self.y_func
        self.points = [
            (x_func(mat), y_func(mat))
            for _, mat in self._data.iter_confusion_matrices()
        ]

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
        self._calculate_points()

    def get_empty_figure(self, *args, **kwds):
        """Returns an empty `matplotlib.Figure` that can be used
        to show the classifier curve. The arguments of this function are
        passed on intact to the constructor of `matplotlib.Figure`,
        except these (which are interpreted here):

            - `title`: the title of the figure.
            - `xlabel`: the label of the X axis. If omitted, we will
              try to infer it from `self.x_func`.
            - `ylabel`: the label of the Y axis. If omitted, we will
              try to infer it from `self.y_func`.

        These must be given as keyword arguments.

        Axis labels are inferred from the function objects that were
        used to obtain the points of the curve; in particular, this method
        is looking for an attribute named ``__axis_label__``, attached to
        the function objects. You can attach such an attribute easily
        by using `yard.utils.axis_label` as a decorator.
        """

        # Infer the labels of the X and Y axes
        def infer_label(func):
            try:
                return getattr(func, "__axis_label__")
            except AttributeError:
                return func.__name__

        if "xlabel" not in kwds:
            kwds["xlabel"] = infer_label(self.x_func)
        if "ylabel" not in kwds:
            kwds["ylabel"] = infer_label(self.y_func)

        return super(BinaryClassifierPerformanceCurve, self).get_empty_figure(
            *args, **kwds
        )

    @classmethod
    def get_friendly_name(cls):
        """Returns a human-readable name of the curve that can be
        used in messages."""
        return cls.__name__


class ROCCurve(BinaryClassifierPerformanceCurve):
    """Class representing a ROC curve.

    A ROC curve plots the true positive rate on the Y axis versus
    the false positive rate on the X axis.
    """

    identifier = "roc"

    def __init__(self, data):
        """Constructs a ROC curve from the given dataset.

        The dataset must contain ``(x, y)`` pairs where `x` is a predicted
        value and `y` defines whether the example is positive or negative.
        When `y` is less than or equal to zero, it is considered a negative
        example, otherwise it is positive. ``False`` also means a negative
        and ``True`` also means a positive example. The dataset can also
        be an instance of `BinaryClassifierData`.
        """
        super(ROCCurve, self).__init__(
            data, BinaryConfusionMatrix.fpr, BinaryConfusionMatrix.tpr
        )

    def auc(self):
        """Constructs the area under the ROC curve by a linear transformation
        of the rank sum of positive instances."""
        pos_ranks = self.data.get_positive_ranks()
        return self.auc_from_pos_ranks(pos_ranks, len(self.data))

    @staticmethod
    def auc_from_pos_ranks(ranks, total):
        """Returns the AUC under a ROC curve, given the ranks of the positive
        examples and the total number of examples.

        This method can be used to calculate an AUC value quickly without
        constructing the curve itself if you have the positive ranks.
        """
        num_pos = len(ranks)
        num_neg = float(total - num_pos)
        sum_pos_ranks = (total + 1) * num_pos - sum(ranks)
        return 1.0 - sum_pos_ranks / (num_pos * num_neg) + (num_pos + 1) / (2 * num_neg)

    def get_empty_figure(self, *args, **kwds):
        """Returns an empty `matplotlib.Figure` that can be used
        to show the ROC curve. The arguments of this function are
        passed on intact to the constructor of `matplotlib.Figure`,
        except these (which are interpreted here):

            - `title`: the title of the figure.
            - `xlabel`: the label of the X axis.
            - `ylabel`: the label of the Y axis.
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

    @classmethod
    def get_friendly_name(cls):
        """Returns a human-readable name of the curve that can be
        used in messages."""
        return "ROC curve"


class PrecisionRecallCurve(BinaryClassifierPerformanceCurve):
    """Class representing a precision-recall curve.

    A precision-recall curve plots precision on the Y axis versus
    recall on the X axis.
    """

    identifier = "pr"

    def __init__(self, data):
        """Constructs a precision-recall curve from the given dataset.

        The dataset must contain ``(x, y)`` pairs where `x` is a predicted
        value and `y` defines whether the example is positive or negative.
        When `y` is less than or equal to zero, it is considered a negative
        example, otherwise it is positive. ``False`` also means a negative
        and ``True`` also means a positive example. The dataset can also
        be an instance of `BinaryClassifierData`.
        """
        super(PrecisionRecallCurve, self).__init__(
            data, BinaryConfusionMatrix.recall, BinaryConfusionMatrix.precision
        )

    @classmethod
    def get_friendly_name(cls):
        """Returns a human-readable name of the curve that can be
        used in messages."""
        return "precision-recall curve"

    def get_interpolated_point(self, x):
        """Returns an interpolated point on this curve at the given
        X position.

        This method performs the proper non-linear interpolation that
        is required for precision-recall curves. Basically, for each
        point, we find the two nearest known points, infer the original
        TP, FP and FN values at those points, and then we interpolate
        linearly in the space of TP-FP-FN values, while recalculating
        the precision and the recall at x.

        It is assumed that `self._points` is sorted in ascending order.
        If not, this function will produce wrong results.
        """
        points = self.points
        pos = bisect(points, (x, 0))

        # Do we have an exact match?
        try:
            if points[pos][0] == x:
                return points[pos]
        except IndexError:
            pass

        # Nope, so we have to interpolate
        if pos == 0:
            # Extrapolation is not possible, just return the
            # first element from points
            return points[0]
        elif pos == len(points):
            # Extrapolation is not possible, just return the
            # last element from points
            return points[-1]

        # Truly interpolating
        (x1, y1), (x2, y2) = points[pos - 1 : pos + 1]
        # The calculations (spelled out nicely) would be as follows:
        #
        # total_pos = self.data.total_positives
        # tp_left, tp_right = total_pos * x1, total_pos * x2
        # fp_left  = tp_left * (1. - y1) / y1
        # fp_right = tp_right * (1. - y2) / y2
        # r = (tp_right-tp_mid)/float(tp_right-tp_left)
        # fp_mid = fp_left*r + fp_right*(1-r)
        # tp_mid = total_pos * x
        # recall_mid = tp_mid / (tp_mid + fp_mid)
        # return (x, recall_mid)
        #
        # Now, we recognise that we can divide almost everything with
        # total_pos, leading us to the following implementation:
        fp_left_over_total_pos = x1 * (1.0 - y1) / y1
        fp_right_over_total_pos = x2 * (1.0 - y2) / y2
        r = (x2 - x) / float(x2 - x1)
        fp_mid_over_total_pos = fp_left_over_total_pos * r + fp_right_over_total_pos * (
            1 - r
        )
        return (x, x / (x + fp_mid_over_total_pos))


class SensitivitySpecificityCurve(BinaryClassifierPerformanceCurve):
    """Class representing a sensitivity-specificity plot.

    A sensitivity-specificity curve plots the sensitivity on the Y axis
    versus the specificity on the X axis.
    """

    identifier = "sespe"

    def __init__(self, data):
        """Constructs a sensitivity-specificity curve from the given dataset.

        The dataset must contain ``(x, y)`` pairs where `x` is a predicted
        value and `y` defines whether the example is positive or negative.
        When `y` is less than or equal to zero, it is considered a negative
        example, otherwise it is positive. ``False`` also means a negative
        and ``True`` also means a positive example. The dataset can also
        be an instance of `BinaryClassifierData`.
        """
        super(SensitivitySpecificityCurve, self).__init__(
            data, BinaryConfusionMatrix.tnr, BinaryConfusionMatrix.recall
        )

    @classmethod
    def get_friendly_name(cls):
        """Returns a human-readable name of the curve that can be
        used in messages."""
        return "sensitivity-specificity plot"


class AccumulationCurve(BinaryClassifierPerformanceCurve):
    """Class representing an accumulation curve.

    An accumulation curve plots the true positive rate on the Y axis
    versus the fraction of data classified as positive on the X axis.
    """

    identifier = "ac"

    def __init__(self, data):
        """Constructs an accumulation curve from the given dataset.

        The dataset must contain ``(x, y)`` pairs where `x` is a predicted
        value and `y` defines whether the example is positive or negative.
        When `y` is less than or equal to zero, it is considered a negative
        example, otherwise it is positive. ``False`` also means a negative
        and ``True`` also means a positive example. The dataset can also
        be an instance of `BinaryClassifierData`.
        """
        super(AccumulationCurve, self).__init__(
            data, BinaryConfusionMatrix.fdp, BinaryConfusionMatrix.tpr
        )

    @classmethod
    def get_friendly_name(cls):
        """Returns a human-readable name of the curve that can be
        used in messages."""
        return "accumulation curve"


class CROCCurve(BinaryClassifierPerformanceCurve):
    """Class representing a concentrated ROC curve.

    A CROC curve plots the true positive rate on the Y axis versus
    the false positive rate on the X axis, but it transforms the X axis
    in order to give more emphasis to the left hand side of the X axis
    (close to zero).
    """

    identifier = "croc"

    def __init__(self, data, alpha=7):
        """Constructs a CROC curve from the given dataset.

        The dataset must contain ``(x, y)`` pairs where `x` is a predicted
        value and `y` defines whether the example is positive or negative.
        When `y` is less than or equal to zero, it is considered a negative
        example, otherwise it is positive. ``False`` also means a negative
        and ``True`` also means a positive example. The dataset can also
        be an instance of `BinaryClassifierData`.

        `alpha` is the magnification factor that defines how much do we want
        to focus on the left side of the X axis. The default `alpha`=7
        transforms a FPR of 0.1 to 0.5.
        """
        self._transformation = ExponentialTransformation(alpha)
        super(CROCCurve, self).__init__(
            data, self._transformed_fpr, BinaryConfusionMatrix.tpr
        )

    def auc(self):
        """Constructs the area under the ROC curve by the average of the
        FPRs at thresholds equal to each positive instance."""
        pos_ranks = self.data.get_positive_ranks()
        return self.auc_from_pos_ranks(pos_ranks, len(self.data))

    def auc_from_pos_ranks(self, pos_ranks, total):
        """Returns the AUC under a CROC curve, given the ranks of the positive
        examples and the total number of examples.

        This method can be used to calculate an AUC value quickly without
        constructing the curve itself if you have the positive ranks.
        """
        pos_count = len(pos_ranks)
        neg_count = float(total - pos_count)
        if neg_count == 0.0:
            return 1.0

        trans = self._transformation
        fprs = [1.0 - (rank - i - 1) / neg_count for i, rank in enumerate(pos_ranks)]
        return 1.0 - sum(trans(fprs)) / pos_count

    @axis_label("Transformed false positive rate")
    def _transformed_fpr(self, matrix):
        """Internal function that returns the transformed FPR value from the
        given confusion matrix that should be plotted on the X axis."""
        return self._transformation(matrix.fpr())

    def get_empty_figure(self, *args, **kwds):
        """Returns an empty `matplotlib.Figure` that can be used
        to show the ROC curve. The arguments of this function are
        passed on intact to the constructor of `matplotlib.Figure`,
        except these (which are interpreted here):

            - `title`: the title of the figure.
            - `xlabel`: the label of the X axis.
            - `ylabel`: the label of the Y axis.
            - `no_discrimination_curve`: if ``True``, the no discrimination
              curve will be drawn. If ``False``, it won't be drawn. If
              a string, it is interpreted as a line style by
              ``matplotlib`` and this line style will be used to draw
              the no discrimination line. If it is a tuple, the first
              element of the tuple will be interpreted as the color
              and the second will be interpreted as the line style
              by ``matplotlib``.

        These must be given as keyword arguments.
        """
        if "no_discrimination_curve" in kwds:
            no_discrimination_curve = kwds["no_discrimination_curve"]
            del kwds["no_discrimination_curve"]
        else:
            no_discrimination_curve = ("#444444", ":")

        # Create the figure by calling the superclass
        fig = super(CROCCurve, self).get_empty_figure(*args, **kwds)
        axes = fig.get_axes()[0]

        # Plot the no-discrimination curve
        if no_discrimination_curve:
            ys = [y / 100.0 for y in range(101)]
            xs = [self._transformation(y) for y in ys]
            if isinstance(no_discrimination_curve, (tuple, list)):
                color, linestyle = no_discrimination_curve
                axes.plot(xs, ys, color=color, linestyle=linestyle)
            else:
                axes.plot(xs, ys, no_discrimination_curve)

        return fig

    def get_interpolated_point(self, x):
        """Returns an interpolated point on this curve at the given
        X position.

        This method performs the proper non-linear interpolation that
        is required for concentrated ROC curves. Basically, for each
        point, we find the two nearest known points, transform the
        X coordinates back to obtain the original FPRs, interpolate
        them, then transform then again.

        It is assumed that `self._points` is sorted in ascending order.
        If not, this function will produce wrong results.
        """
        points = self.points
        pos = bisect(points, (x, 0))

        # Do we have an exact match?
        try:
            if points[pos][0] == x:
                return points[pos]
        except IndexError:
            pass

        # Nope, so we have to interpolate
        if pos == 0:
            # Extrapolation is not possible, just return the
            # first element from points
            return points[0]
        elif pos == len(points):
            # Extrapolation is not possible, just return the
            # last element from points
            return points[-1]

        if pos == 0:
            # Extrapolating instead
            (x1, y1), (x2, y2) = points[:2]
        elif pos == len(points):
            # Extrapolating instead
            (x1, y1), (x2, y2) = points[-2:]
        else:
            # Truly interpolating
            (x1, y1), (x2, y2) = points[pos - 1 : pos + 1]

        trans_inv = self._transformation.inverse

        fpr1, fpr2, fpr_mid = trans_inv(x1), trans_inv(x2), trans_inv(x)
        r = (fpr2 - fpr_mid) / (fpr2 - fpr1)
        return (x, y1 * r + y2 * (1 - r))

    @classmethod
    def get_friendly_name(cls):
        """Returns a human-readable name of the curve that can be
        used in messages."""
        return "concentrated ROC curve"


class FScoreCurve(BinaryClassifierPerformanceCurve):
    """Class representing an F-score curve.

    An F-score curve plots the F-score on the Y axis versus the fraction
    of data classified as positive on the X axis.
    """

    identifier = "fscore"

    def __init__(self, data, f=1.0):
        """Constructs an F-score curve from the given dataset.

        The dataset must contain ``(x, y)`` pairs where `x` is a predicted
        value and `y` defines whether the example is positive or negative.
        When `y` is less than or equal to zero, it is considered a negative
        example, otherwise it is positive. ``False`` also means a negative
        and ``True`` also means a positive example. The dataset can also
        be an instance of `BinaryClassifierData`.

        The value of `f` controls the weighting between precision and recall
        in the F-score formula. `f` = 1 means that equal importance is attached
        to precision and recall. In general, recall is considered `f` times more
        important than precision.
        """

        @axis_label("F-score")
        def f_score(matrix):
            """Internal function that binds the `f` parameter of
            `BinaryConfusionMatrix.f_score` to the value specified in the constructor.
            """
            return BinaryConfusionMatrix.f_score(matrix, f)

        super(FScoreCurve, self).__init__(data, BinaryConfusionMatrix.fdp, f_score)

    @classmethod
    def get_friendly_name(cls):
        """Returns a human-readable name of the curve that can be
        used in messages."""
        return "F-score curve"
