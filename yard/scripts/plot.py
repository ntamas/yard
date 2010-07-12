"""Standalone command-line application that plots ROC, precision-recall
and accumulation curves."""

import sys

from itertools import cycle, izip
from yard.data import BinaryClassifierData
from yard.curve import ROCCurve, CROCCurve, AccumulationCurve, \
                       PrecisionRecallCurve
from yard.scripts import CommandLineAppForClassifierData

__author__  = "Tamas Nepusz"
__email__   = "tamas@cs.rhul.ac.uk"
__copyright__ = "Copyright (c) 2010, Tamas Nepusz"
__license__ = "MIT"

class ROCPlotterApplication(CommandLineAppForClassifierData):
    """\
    %prog input_file
    
    Standalone command-line application that plots ROC, precision-recall
    and accumulation curves.

    The input file must contain one observation per line, the first column
    being the expected class (1 for positive examples, -1 for negatives),
    the second being the prediction itself. You can also use the -c switch
    to use different column indices and multiple datasets. Columns are
    separated by whitespace per default.\
    """

    short_name = "yard-plot"

    def __init__(self):
        super(ROCPlotterApplication, self).__init__()

    def add_parser_options(self):
        """Creates the command line parser object for the application"""
        super(ROCPlotterApplication, self).add_parser_options()

        parser = self.parser

        parser.add_option("-t", "--curve-type", dest="curve_types",
                metavar="TYPE", choices=("roc", "pr", "ac", "croc"),
                action="append", default=[],
                help="sets the TYPE of the curve to be plotted "
                     "(roc, pr, ac or croc). May be specified "
                     "multiple times if the output is PDF.")
        parser.add_option("-l", "--log-scale", dest="log_scale",
                metavar="AXES",
                help="use logarithmic scale on the given AXES. "
                     "Valid values: none, x, y and xy",
                choices=["none", "x", "y", "xy"], default="none")
        parser.add_option("-o", "--output", dest="output", metavar="FILE",
                help="saves the plot to the given FILE instead of showing it",
                default=None)
        parser.add_option("--show-auc", dest="show_auc", action="store_true",
                default=False, help="shows the AUC scores in the legend")
        parser.add_option("--no-resampling", dest="resampling", action="store_false",
                default=True, help="don't resample curves before "
                                   "plotting and AUC calculation")

    def run_real(self):
        """Runs the main application"""

        # Do we need headless mode for matplotlib?
        if self.options.output:
            import matplotlib
            matplotlib.use("agg")

        # Get the types of the curves to be plotted
        curve_name_to_class = dict(
                roc=ROCCurve, croc=CROCCurve,
                ac=AccumulationCurve,
                pr=PrecisionRecallCurve
        )
        curve_classes = []
        for name in self.options.curve_types:
            try:
                curve_classes.append(curve_name_to_class[name])
            except KeyError:
                self.parser.error("Unknown curve type: %s" % name)

        # If no curve type was given, assume a ROC curve
        if not curve_classes:
            curve_classes.append(ROCCurve)

        # Do we have multiple curve types? If so, we need PDF output
        if len(curve_classes) > 1:
            if not self.options.output or \
               not self.options.output.endswith(".pdf"):
                self.parser.error("multiple curves can only be plotted to PDF")

            from matplotlib.backends.backend_pdf import PdfPages
            pp = PdfPages(self.options.output)
            def figure_saver(figure):
                pp.savefig(figure)
        elif self.options.output:
            # Figure with a single plot will be created
            pp = None
            def figure_saver(figure):
                self.log.info("Saving plot to %s..." % self.options.output)
                figure.savefig(self.options.output)
        else:
            # Figure will be shown on screen
            def figure_saver(figure):
                figure.show()
                print "Press Enter..."
                raw_input()

        self.process_input_files()

        self.log.info("Plotting results...")
        for curve_class in curve_classes:
            fig = self.get_figure_for_curves(curve_class)
            figure_saver(fig)

        # For multi-page output, we have to close it explicitly
        if pp is not None:
            pp.close()

    def get_figure_for_curves(self, curve_class):
        """Plots curves given by `curve_class` for all the data in `self.data`.
        `curve_class` is a subclass of `BinaryClassifierPerformanceCurve`.
        `self.data` must be a dict of lists, and the ``__class__`` key of
        `self.data` must map to the expected classes of elements. Returns an
        instance of `matplotlib.figure.Figure`."""
        fig, axes = None, None

        data = self.data
        expected = data["__class__"]

        keys = sorted(data.keys())
        keys.remove("__class__")

        styles = ["r-",  "b-",  "g-",  "c-",  "m-",  "y-",  "k-", \
                  "r--", "b--", "g--", "c--", "m--", "y--", "k--"]

        # Plot the curves
        line_handles, labels, aucs = [], [], []
        for key, style in izip(keys, cycle(styles)):
            self.log.info("Calculating %s for %s..." %
                    (curve_class.get_friendly_name(), key))
            observed = data[key]

            bc_data = BinaryClassifierData(zip(observed, expected), title=key)
            curve = curve_class(bc_data)

            if self.options.resampling:
                curve.resample([x/2000. for x in xrange(2001)])

            if self.options.show_auc:
                aucs.append(curve.auc())
                labels.append("%s, AUC=%.4f" % (key, aucs[-1]))
            else:
                labels.append(key)

            if not fig:
                fig = curve.get_empty_figure()
                axes = fig.get_axes()[0]

            line_handle = curve.plot_on_axes(axes, style=style, legend=False)
            line_handles.append(line_handle)

        if aucs:
            # Sort the labels of the legend in decreasing order of AUC
            indices = sorted(range(len(aucs)), key=aucs.__getitem__,
                             reverse=True)
            line_handles = [line_handles[i] for i in indices]
            labels = [labels[i] for i in indices]
            aucs = [aucs[i] for i in indices]

        if axes:
            legend_pos = "best"

            # Set logarithmic axes if needed
            if "x" in self.options.log_scale:
                axes.set_xscale("log")
                legend_pos = "upper left"
            if "y" in self.options.log_scale:
                axes.set_yscale("log")

            # Plot the legend
            axes.legend(line_handles, labels, loc = legend_pos)

        return fig


def main():
    """Entry point for the plotter script"""
    sys.exit(ROCPlotterApplication().run())

if __name__ == "__main__":
    main()
