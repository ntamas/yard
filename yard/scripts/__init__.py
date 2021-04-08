"""
Classes and utilities commonly used in the command line scripts.

Each command line script of ``yard`` is derived from the `CommandLineApp`
class, which takes care of implementing functionality common for all the
scripts, such as:

- providing a command line parser (an instance of `optparse.OptionParser`)
- providing a logger instance
- defining methods for extending the default option parser and for
  signaling fatal errors to the caller
"""

import logging
import sys

from collections import defaultdict
from optparse import OptionParser
from textwrap import dedent

__author__ = "Tamas Nepusz"
__email__ = "tamas@cs.rhul.ac.uk"
__copyright__ = "Copyright (c) 2010, Tamas Nepusz"
__license__ = "MIT"


class CommandLineApp(object):
    """Generic command line application class"""

    def __init__(self, logger=None):
        if logger:
            self.log = logger
        else:
            self.log = self.create_logger()
        self.options, self.args = None, None

    def create_parser(self):
        """Creates a command line parser for the application"""
        doc = self.__class__.__doc__
        parser = OptionParser(usage=dedent(doc).strip())
        parser.add_option(
            "-q",
            "--quiet",
            dest="verbose",
            default=True,
            action="store_false",
            help="quiet output (logs only warnings)",
        )
        parser.add_option(
            "-d",
            "--debug",
            dest="debug",
            action="store_true",
            help="show debug messages",
        )
        return parser

    def add_parser_options(self):
        """Adds command line options to the command line parser"""
        pass

    def create_logger(self):
        """Creates a logger for the application"""
        if hasattr(self.__class__, "short_name"):
            log_name = self.__class__.short_name
        else:
            log_name = self.__class__.__module__

        log = logging.getLogger(log_name)
        log.setLevel(logging.WARNING)
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")

        return log

    def error(self, message):
        """Signals a fatal error and shuts down the application."""
        self.parser.error(message)

    def run(self, args=None):
        """Runs the application. This method processes the command line using the
        command line parser and as such, it should not be overridden in child
        classes unless you know what you are doing. If you want to implement
        the actual logic of your application, override `run_real` instead."""
        self.parser = self.create_parser()
        self.add_parser_options()
        self.options, self.args = self.parser.parse_args(args)

        if self.options.verbose:
            self.log.setLevel(logging.INFO)
        if self.options.debug:
            self.log.setLevel(logging.DEBUG)

        return self.run_real()

    def run_real(self):
        self.log.info("Nothing to do.")
        return 0


class CommandLineAppForClassifierData(CommandLineApp):
    """Subclass of `CommandLineApp` that adds the usual command line options
    for processing tabular classifier data.

    This class can be used as a base class for applications that work from
    flat files containing classifier outputs in columns.
    """

    def __init__(self):
        super(CommandLineAppForClassifierData, self).__init__()
        self.cols, self.sep = None, None
        self.data = defaultdict(list)

    def add_parser_options(self):
        """Adds the usual command line parse options for command line scripts
        working with tabular classifier data."""
        parser = self.parser
        parser.add_option(
            "-c",
            "--columns",
            dest="columns",
            metavar="COLUMNS",
            help="use the given COLUMNS from the input file. Column indices "
            "are separated by commas. The first index specifies the "
            "column containing the class of the datapoint (positive "
            "or negative), the remaining indices specify predictions "
            "according to various prediction methods. If the class "
            "column does not contain a numeric value, the whole row "
            "is considered as a header.",
            default=None,
        )
        parser.add_option(
            "-f",
            "--field-separator",
            dest="sep",
            metavar="CHAR",
            help="use the given separator CHARacter between columns. "
            "If omitted, all whitespace characters are separators.",
            default=None,
        )

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
                result.extend(range(start - 1, end))
            else:
                result.append(int(part) - 1)
        return result

    def process_options(self):
        """Processes the command line options and sets up `self.sep` and
        self.cols`, which are needed later by `process_file()`. You are
        not required to call this function, `process_file()` will do so
        when needed."""

        # Process self.options.sep
        sep = self.options.sep
        if sep is not None:
            if len(sep) == 2 and sep[0] == "\\":
                sep = eval(r'"%s"' % sep)
            elif len(sep) != 1:
                self.parser.error("Column separator must be a single character")
        self.sep = sep

        # Process self.options.columns
        self.cols = self.options.columns
        if self.cols is not None:
            # Using some columns only
            try:
                self.cols = self.parse_column_indices(self.cols)
            except ValueError:
                self.parser.error(
                    "Format error in column specification: %r" % self.cols
                )
            if len(self.cols) == 1:
                self.parser.error("Must specify at least two column indices")
            if min(self.cols) < 0:
                self.parser.error("Column indices must be positive")
        else:
            # Using all columns
            pass

    def process_file(self, stream):
        """Processes the given input `stream` and stores the results in
        `self.data`, which must be a ``defaultdict(list)``"""

        if self.cols is None and self.sep is None:
            self.process_options()

        cols, sep = self.cols, self.sep
        seen_header = False

        if cols is not None:
            # Prepare ncols and headers in advance here
            ncols = len(cols)
            headers = ["Dataset %d" % idx for idx in range(ncols)]
            headers[0] = "__class__"

        for line in stream:
            line = line.strip()
            if not line:
                continue

            parts = line.split(sep)
            if not parts:
                continue

            if not seen_header:
                # This is either a header row or a data row
                if cols is None:
                    colidx = 0
                else:
                    colidx = cols[0]

                try:
                    int(float(parts[colidx]))
                except (IndexError, ValueError):
                    # This is surely a header row
                    seen_header = True
                    if cols is None:
                        # Prepare ncols now that we know the header
                        cols = range(len(parts))
                        ncols = len(parts)
                        headers = list(parts)
                        headers[0] = "__class__"
                    else:
                        headers[1:] = [parts[idx] for idx in cols[1:]]
                    anon_dataset_idx = 1

                    # Add dataset names for empty headers
                    for idx, header in enumerate(headers):
                        if not header:
                            while ("Dataset %d" % anon_dataset_idx) in self.data:
                                anon_dataset_idx += 1
                            headers[idx] = "Dataset %d" % anon_dataset_idx
                    continue

                # This is a data row; there will be no header row. Set up the
                # dataset names for all the columns if we did not specify
                # columns in the input file
                if cols is None:
                    cols = range(len(parts))
                    ncols = len(parts)
                    headers = ["Dataset %d" % idx for idx in range(ncols)]
                    headers[0] = "__class__"

            # This is a data row
            for i in range(ncols):
                self.data[headers[i]].append(float(parts[cols[i]]))

    def process_input_files(self):
        """Processes all the input files passed in the positional command
        line arguments."""

        if not self.args:
            self.args = ["-"]

        for arg in self.args:
            if arg == "-":
                handle = sys.stdin
                arg = "standard input"
            else:
                handle = open(arg)
            self.log.info("Processing %s..." % arg)
            self.process_file(handle)

        if len(self.data) == 0:
            self.parser.error("No data columns in input file")
