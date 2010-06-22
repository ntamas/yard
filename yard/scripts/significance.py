"""Standalone command-line application that performs statistical tests
to determine whether the difference between AUC scores of various curves and
datasets are statistically significant or not."""

import itertools
import sys

from yard.data import BinaryClassifierData
from yard.scripts import CommandLineAppForClassifierData
from yard.significance import PairedPermutationTest

class SignificanceTestApplication(CommandLineAppForClassifierData):
    """\
    %prog input_file
    
    Standalone command-line application that tests for significant
    differences between the AUC scores of ROC curves.

    The input file must contain one observation per line, the first column
    being the expected class (1 for positive examples, -1 for negatives),
    the second being the prediction itself. You can also use the -c switch
    to use different column indices and multiple datasets. Columns are
    separated by whitespace per default.\
    """

    short_name = "yard-signi"

    def __init__(self):
        super(SignificanceTestApplication, self).__init__()

    def add_parser_options(self):
        """Creates the command line parser object for the application"""
        super(SignificanceTestApplication, self).add_parser_options()

        parser = self.parser

        """
        parser.add_option("-t", "--curve-type", dest="curve_type",
                metavar="TYPE", choices=("roc", "pr", "ac", "croc"),
                default="roc", 
                help="sets the TYPE of the curve to be plotted "
                     "(roc, pr, ac or croc)")
        """

    def run_real(self):
        """Runs the main application"""
        self.process_input_files()
        self.run_tests()

    def run_tests(self):
        """Runs pairwise significance tests on the datasets found in
        ``self.data``."""
        data = self.data
        expected = data["__class__"]

        keys = sorted(data.keys())
        keys.remove("__class__")

        for key in keys:
            self.log.info("Preparing dataset for %s..." % key)
            data[key] = BinaryClassifierData(zip(data[key], expected), title=key)

        significance_test = PairedPermutationTest()
        for key1, key2 in itertools.product(keys, keys):
            if key1 >= key2:
                continue
            diff, p_value = significance_test.test(data[key1], data[key2])
            if p_value < 0.01:
                stars = "***"
            elif p_value < 0.05:
                stars = "**"
            elif p_value < 0.1:
                stars = "*"
            else:
                stars = ""
            self.log.info("%3s   d=%8.3g   p=%8.3g   %s vs %s" %
                          (stars, diff, p_value, key1, key2))



if __name__ == "__main__":
    sys.exit(SignificanceTestApplication().run())

