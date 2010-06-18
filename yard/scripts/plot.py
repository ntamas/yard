"""Standalone command-line application that plots ROC, precision-recall
and accumulation curves."""

import sys

from collections import defaultdict
from itertools import cycle, izip
from yard.data import BinaryClassifierData
from yard.curve import ROCCurve, CROCCurve, AccumulationCurve, \
                       PrecisionRecallCurve

class ROCPlotterApplication(object):
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
        self.parser = self.create_parser()
        self.log = self.create_logger()
        self.curve_class = None
        self.cols = None
        self.sep = None
        self.options = None

    def create_logger(self):
        """Creates the logger object for the application"""
        import logging

        log = logging.getLogger(self.short_name)
        log.setLevel(logging.DEBUG)

        return log

    def create_parser(self):
        """Creates the command line parser object for the application"""
        import optparse
        from textwrap import dedent

        parser = optparse.OptionParser(usage=dedent(self.__class__.__doc__))
        parser.add_option("-c", "--columns", dest="columns", metavar="COLUMNS",
                help="use the given COLUMNS from the input file. Column indices "\
                     "are separated by commas. The first index specifies the "\
                     "column containing the class of the datapoint (positive "\
                     "or negative), the remaining indices specify predictions "\
                     "according to various prediction methods. If the class "\
                     "column does not contain a numeric value, the whole row "\
                     "is considered as a header.",
                default="1,2")
        parser.add_option("-f", "--field-separator", dest="sep",
                metavar="CHAR",
                help="use the given separator CHARacter between columns. "\
                     "If omitted, all whitespace characters are separators.",
                default=None)
        parser.add_option("-l", "--log-scale", dest="log_scale",
                metavar="AXES",
                help="use logarithmic scale on the given AXES. "
                     "Valid values: none, x, y and xy",
                choices=["none", "x", "y", "xy"], default="none")
        parser.add_option("-o", "--output", dest="output", metavar="FILE",
                help="saves the plot to the given FILE instead of showing it",
                default=None)
        parser.add_option("-t", "--curve-type", dest="curve_type",
                metavar="TYPE", choices=("roc", "pr", "ac", "croc"),
                default="roc", 
                help="sets the TYPE of the curve to be plotted "
                     "(roc, pr, ac or croc)")
        parser.add_option("-v", "--verbose", dest="verbose",
                action="store_true", default=False,
                help="verbose mode, shows progress while calculating curves")
        parser.add_option("--show-auc", dest="show_auc", action="store_true",
                default=False, help="shows the AUC scores in the legend")
        parser.add_option("--no-resampling", dest="resampling", action="store_false",
                default=True, help="don't resample curves before "
                                   "plotting and AUC calculation")
        return parser

    @staticmethod
    def parse_column_indices(indices):
        """Parses the column indices passed to the ``-c`` argument on the
        command line. The ``indices`` variable is a string containing the
        value of the ``-c`` argument. It must be a comma-separated list of
        integers or integer intervals (``from-to``). The result is a list
        of integers."""
        parts = indices.split(",")
        result = []
        for part in parts:
            if "-" in part:
                start, end = [int(idx) for idx in part.split("-", 1)]
                result.extend(range(start-1, end))
            else:
                result.append(int(part)-1)
        return result

    def run(self):
        """Runs the main application"""
        self.options, args = self.parser.parse_args()

        # Set logging level
        from logging import basicConfig, WARN, INFO
        basicConfig(level=[WARN, INFO][self.options.verbose], \
                    format="%(message)s")

        # Process self.options.sep
        sep = self.options.sep
        if sep is not None:
            if len(sep) == 2 and sep[0] == '\\':
                sep = eval(r'"%s"' % sep)
            elif len(sep) != 1:
                self.parser.error("Column separator must be a single character")
        self.sep = sep

        # Process self.options.columns
        self.cols = self.options.columns
        try:
            self.cols = self.parse_column_indices(self.cols)
        except ValueError:
            self.parser.error("Format error in column specification: %r" % self.cols)
        if len(self.cols) == 1:
            self.parser.error("Must specify at least two column indices")
        if min(self.cols) < 0:
            self.parser.error("Column indices must be positive")

        # Get the type of the curve to be plotted
        try:
            self.curve_class = dict(
                    roc=ROCCurve, croc=CROCCurve,
                    ac=AccumulationCurve,
                    pr=PrecisionRecallCurve
            )[self.options.curve_type]
        except KeyError:
            self.parser.error("Unknown curve type: %s" % self.options.curve_type)

        # Do we need headless mode for matplotlib?
        if self.options.output:
            import matplotlib
            matplotlib.use("agg")

        if not args:
            args = ["-"]

        data = defaultdict(list)
        for arg in args:
            if arg == "-":
                handle = sys.stdin
                arg = "standard input"
            else:
                handle = open(arg)
            self.log.info("Processing %s..." % arg)
            self.process_file(handle, data)

        if len(data) == 0:
            self.parser.error("No data columns in input file")

        self.plot_curves(data)

    def process_file(self, stream, data):
        """Processes the given input `stream` and stores the results in `data`,
        which must be a ``defaultdict(list)``"""
        cols, sep = self.cols, self.sep

        ncols = len(cols)
        headers = ["Dataset %d" % idx for idx in xrange(ncols)]
        headers[0] = "__class__"

        seen_header = False

        for line in stream:
            line = line.strip()
            if not line:
                continue

            parts = line.split(sep)
            try:
                int(float(parts[cols[0]]))
            except (IndexError, ValueError):
                # This is a header row
                if seen_header:
                    raise ValueError("duplicate header row in input file")
                seen_header = True
                headers[1:] = [parts[idx] for idx in cols[1:]]
                anon_dataset_idx = 1
                for idx, header in enumerate(headers):
                    if not header:
                        while ("Dataset %d" % anon_dataset_idx) in data:
                            anon_dataset_idx += 1
                        headers[idx] = "Dataset %d" % anon_dataset_idx
                continue

            # This is a data row
            for i in xrange(ncols):
                data[headers[i]].append(float(parts[cols[i]]))

    def plot_curves(self, data):
        """Plots all the ROC curves in the given `data`. `data` must be a
        dict of lists, and the ``__class__`` key of `data` must map to
        the expected classes of elements."""
        expected = data["__class__"]

        fig, axes = None, None

        keys = sorted(data.keys())
        keys.remove("__class__")
        styles = ["r-",  "b-",  "g-",  "c-",  "m-",  "y-",  "k-", \
                  "r--", "b--", "g--", "c--", "m--", "y--", "k--"]

        # Plot the curves
        line_handles, labels, aucs = [], [], []
        for key, style in izip(keys, cycle(styles)):
            self.log.info("Calculating curve for %s..." % key)
            observed = data[key]

            bc_data = BinaryClassifierData(zip(observed, expected), title=key)
            curve = self.curve_class(bc_data)

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
            axes.legend(line_handles, labels, loc = legend_pos)

        if not self.options.output:
            self.log.info("Plotting results...")
            fig.show()
            print "Press Enter to exit..."
            raw_input()
        else:
            self.log.info("Saving plot to %s..." % self.options.output)
            fig.savefig(self.options.output)

if __name__ == "__main__":
    sys.exit(ROCPlotterApplication().run())

