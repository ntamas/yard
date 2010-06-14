"""
Curve classes used in YARD.

This package contains implementations for all the curves YARD can plot.
At the time of writing, this includes:

    - ROC curves
    - CROC curves
    - Precision-recall curves
    - Accumulation curves
"""

__author__  = "Tamas Nepusz"
__email__   = "tamas@cs.rhul.ac.uk"
__copyright__ = "Copyright (c) 2010, Tamas Nepusz"
__license__ = "MIT"


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
        auc = sum((y0+y1) / 2. * (x0-x1) \
                  for (x0, y0), (x1, y1) in izip(points, points[1:]))
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
        step = (n-1) / (k-1.)
        result = [points[int(idx*step)] for idx in xrange(1, k-1)]
        result.append(points[-1])
        self._points = result

    def get_empty_figure(self, *args, **kwds):
        """Returns an empty `matplotlib.Figure` that can be used to show the
        curve. The arguments of this function are passed on intact to the
        constructor of `matplotlib.Figure`, except these (which are interpreted
        here):

            - `title`: the title of the figure.

            - `xlabel`: the label of the X axis. If omitted, we will try to
              infer it from `self.x_method_name`.

            - `ylabel`: the label of the Y axis. If omitted, we will try to
              infer it from `self.y_method_name`.

        These must be given as keyword arguments.
        """
        import matplotlib.pyplot as plt

        # Construct the figure
        fig = plt.figure(*args, **kwds)

        # Create the axes, set the axis labels
        axes = fig.add_subplot(111)

        if "xlabel" in kwds:
            axes.set_xlabel(kwds["xlabel"])
            del kwds["xlabel"]
        if "ylabel" in kwds:
            axes.set_ylabel(kwds["ylabel"])
            del kwds["ylabel"]
        if "title" in kwds:
            axes.set_title(title)
            del kwds["title"]

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
        self._points = [tuple(point) for point in points]

    def show(self, *args, **kwds):
        """Constructs and shows a `matplotlib.Figure` that plots the
        curve. If you need the figure itself for further manipulations,
        call `get_figure()` instead of this method.

        The arguments of this function are passed on intact to
        `get_figure()`.
        """
        self.get_figure(*args, **kwds).show()


class BinaryClassifierPerformanceCurve(Curve):
    """Class representing a broad class of binary classifier performance
    curves.

    By using this class diretly, you are free to specify what's on the X
    and Y axes of the plot. If you are interested in ROC curves, see
    `ROCCurve`, which is a subclass of this class. If you are interested
    in precision-recall curves, see `PrecisionRecallCurve`, which is also
    a subclass. Accumulation curves are implemented in `AccumulationCurve`.
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

    def _calculate_points(self):
        """Returns the actual points of the curve as a list of tuples."""
        meth_x = getattr(BinaryConfusionMatrix, self.x_method_name)
        meth_y = getattr(BinaryConfusionMatrix, self.y_method_name)
        self._points = [(meth_x(mat), meth_y(mat)) for _, mat in \
                self._data.iter_confusion_matrices()]

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
              try to infer it from `self.x_method_name`.

            - `ylabel`: the label of the Y axis. If omitted, we will
              try to infer it from `self.y_method_name`.

        These must be given as keyword arguments.
        """
        # Set up the dict mapping method names to labels
        known_labels = dict(\
                accuracy="Accuracy", \
                fdn="Fraction of data classified negative", \
                fdp="Fraction of data classified positive", \
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

        # Infer the labels of the X and Y axes
        if "xlabel" not in kwds:
            kwds["xlabel"] = known_labels.get(self.x_method_name,
                                              self.x_method_name)
        if "ylabel" not in kwds:
            kwds["ylabel"] = known_labels.get(self.y_method_name,
                                              self.y_method_name)

        super(BinaryClassifierPerformanceCurve, self).get_empty_figure(
                *args, **kwds)


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

