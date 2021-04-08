"""Standalone command-line application that calculates the AUC
(Area Under Curve) statistic for various curves."""

import sys

from yard.data import BinaryClassifierData
from yard.curve import CurveFactory
from yard.scripts import CommandLineAppForClassifierData

__author__ = "Tamas Nepusz"
__email__ = "tamas@cs.rhul.ac.uk"
__copyright__ = "Copyright (c) 2010, Tamas Nepusz"
__license__ = "MIT"


class AUCCalculatorApplication(CommandLineAppForClassifierData):
    """\
    %prog input_file
    
    Standalone command-line application that calculates the AUC
    (Area Under Curve) statistic for various curves.

    The input file must contain one observation per line, the first column
    being the expected class (1 for positive examples, -1 for negatives),
    the second being the prediction itself. You can also use the -c switch
    to use different column indices and multiple datasets. Columns are
    separated by whitespace per default.\
    """

    short_name = "yard-auc"

    def __init__(self):
        super(AUCCalculatorApplication, self).__init__()

    def add_parser_options(self):
        """Creates the command line parser object for the application"""
        super(AUCCalculatorApplication, self).add_parser_options()

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

    def run_real(self):
        """Runs the main application"""

        # If no curve type was given, assume a ROC curve
        if not self.options.curve_types:
            self.options.curve_types = ["roc"]

        # Get the types of the curves to be plotted
        curve_classes = []
        for name in self.options.curve_types:
            try:
                curve_classes.append(CurveFactory.find_class_by_name(name))
            except ValueError:
                self.parser.error("Unknown curve type: %s" % name)

        self.process_input_files()
        for curve_class in curve_classes:
            self.print_scores_for_curve(curve_class)

    def print_scores_for_curve(self, curve_class):
        """Calculates AUC scores for curves given by `curve_class` for all
        the data in `self.data`.

        `curve_class` is a subclass of `BinaryClassifierPerformanceCurve`.
        `self.data` must be a dict of lists, and the ``__class__`` key of
        `self.data` must map to the expected classes of elements.
        """
        data = self.data
        expected = data["__class__"]

        keys = sorted(data.keys())
        keys.remove("__class__")

        print("Calculating AUCs for %s..." % curve_class.get_friendly_name())
        for key in keys:
            observed = data[key]

            bc_data = BinaryClassifierData(zip(observed, expected), title=key)
            auc = curve_class(bc_data).auc()
            print("  AUC[%s] = %.4f" % (key, auc))
        print("")


def main():
    """Entry point for the plotter script"""
    sys.exit(AUCCalculatorApplication().run())


if __name__ == "__main__":
    main()
