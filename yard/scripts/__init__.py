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

from optparse import OptionParser
from textwrap import dedent

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
        parser.add_option("-q", "--quiet", dest="verbose",
                default=True, action="store_false",
                help="quiet output (logs only warnings)")
        parser.add_option("-d", "--debug", dest="debug",
                action="store_true", help="show debug messages")
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
