"""Standalone command-line application that plots ROC, precision-recall
and accumulation curves."""

import sys

from itertools import cycle

from yard.data import BinaryClassifierData
from yard.curve import CurveFactory
from yard.scripts import CommandLineAppForClassifierData
from yard.utils import parse_size

__author__ = "Tamas Nepusz"
__email__ = "tamas@cs.rhul.ac.uk"
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

        parser.add_option(
            "-t",
            "--curve-type",
            dest="curve_types",
            metavar="TYPE",
            choices=CurveFactory.get_curve_names(),
            action="append",
            default=[],
            help="sets the TYPE of the curve to be plotted "
            "(roc, pr, ac, sespe or croc). May be specified "
            "multiple times.",
        )
        parser.add_option(
            "-l",
            "--log-scale",
            dest="log_scale",
            metavar="AXES",
            help="use logarithmic scale on the given AXES. "
            "Valid values: none, x, y and xy",
            choices=["none", "x", "y", "xy"],
            default="none",
        )
        parser.add_option(
            "-o",
            "--output",
            dest="output",
            metavar="FILE",
            help="saves the plot to the given FILE instead of showing it",
            default=None,
        )
        parser.add_option(
            "-s",
            "--size",
            dest="size",
            metavar="WIDTHxHEIGHT",
            help="sets the size of the figure to WIDTHxHEIGHT, where "
            "WIDTH and HEIGHT are measures in inches. You may "
            "specify alternative measures (cm or mm) by adding "
            'them as a suffix; e.g., "6cmx4cm" or "6cm x 4cm"',
            default=None,
        )
        parser.add_option(
            "--dpi",
            dest="dpi",
            metavar="DPI",
            type=float,
            default=72.0,
            help="specifies the dpi value (dots per inch) when "
            "converting pixels to inches and vice versa "
            "in figure and font size calculations. "
            "Default: %default",
        )
        parser.add_option(
            "--font-size",
            dest="font_size",
            metavar="SIZE",
            type=float,
            default=None,
            help="overrides the font size to be used on figures, " "in points (pt).",
        )
        parser.add_option(
            "--show-auc",
            dest="show_auc",
            action="store_true",
            default=False,
            help="shows the AUC scores in the legend",
        )
        parser.add_option(
            "--no-resampling",
            dest="resampling",
            action="store_false",
            default=True,
            help="don't resample curves before " "plotting and AUC calculation",
        )

    def run_real(self):
        """Runs the main application"""
        import matplotlib

        # Do we need headless mode for matplotlib?
        if self.options.output:
            matplotlib.use("agg")

        # If no curve type was given, assume a ROC curve
        if not self.options.curve_types:
            self.options.curve_types = ["roc"]

        # Set up the font size
        if self.options.font_size is not None:
            for param in ["font.size", "legend.fontsize"]:
                matplotlib.rcParams[param] = self.options.font_size

        # Get the types of the curves to be plotted
        curve_classes = []
        for name in self.options.curve_types:
            try:
                curve_classes.append(CurveFactory.find_class_by_name(name))
            except ValueError:
                self.parser.error("Unknown curve type: %s" % name)

        # Do we have multiple curve types? If so, we need PDF output
        pp = None
        if len(curve_classes) > 1:
            if not self.options.output or not self.options.output.endswith(".pdf"):
                self.parser.error("multiple curves can only be plotted to PDF")

            try:
                from matplotlib.backends.backend_pdf import PdfPages
            except ImportError:
                self.parser.error(
                    "Matplotlib is too old and does not have "
                    "multi-page PDF support yet. Please upgrade it to "
                    "Matplotlib 0.99 or later"
                )

            pp = PdfPages(self.options.output)

            def figure_saver(figure):
                pp.savefig(figure, bbox_inches="tight")

        elif self.options.output:
            # Figure with a single plot will be created
            def figure_saver(figure):
                self.log.info("Saving plot to %s..." % self.options.output)
                figure.savefig(self.options.output, bbox_inches="tight")

        else:
            # Figure will be shown on screen
            def figure_saver(figure):
                import matplotlib.pyplot as plt

                plt.show()

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

        styles = [
            "r-",
            "b-",
            "g-",
            "c-",
            "m-",
            "y-",
            "k-",
            "r--",
            "b--",
            "g--",
            "c--",
            "m--",
            "y--",
            "k--",
        ]

        # Plot the curves
        line_handles, labels, aucs = [], [], []
        for key, style in zip(keys, cycle(styles)):
            self.log.info(
                "Calculating %s for %s..." % (curve_class.get_friendly_name(), key)
            )
            observed = data[key]

            bc_data = BinaryClassifierData(zip(observed, expected), title=key)
            curve = curve_class(bc_data)

            if self.options.resampling:
                curve.resample(x / 2000.0 for x in range(2001))

            if self.options.show_auc:
                aucs.append(curve.auc())
                labels.append("%s, AUC=%.4f" % (key, aucs[-1]))
            else:
                labels.append(key)

            if not fig:
                dpi = self.options.dpi
                fig = curve.get_empty_figure(
                    dpi=dpi, figsize=parse_size(self.options.size, dpi=dpi)
                )
                axes = fig.get_axes()[0]

            line_handle = curve.plot_on_axes(axes, style=style, legend=False)
            line_handles.append(line_handle)

        if aucs:
            # Sort the labels of the legend in decreasing order of AUC
            indices = sorted(range(len(aucs)), key=aucs.__getitem__, reverse=True)
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
            axes.legend(line_handles, labels, loc=legend_pos)

        return fig


def main():
    """Entry point for the plotter script"""
    sys.exit(ROCPlotterApplication().run())


if __name__ == "__main__":
    main()
