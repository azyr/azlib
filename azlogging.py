import logging
import sys

class NoDiagnosticsFilter(logging.Filter):
    """Filter used to pick only records that have levelno below cutofflevel."""

    def __init__(self, cutofflevel):
        self.cutofflevel = cutofflevel

    def filter(self, record):
        return record.levelno < self.cutofflevel

def basic_config(**kwargs):

    """Remove all existing logging handlers/filters and setup new ones.

    Running this will make ipython work properly with scripts/modules that use
    logging. This is also a quick way to have a basic logging configuration that
    splits warnings and move severe messages to stderr, others to stdout.

    Warning: This is not thread safe, use only at the initialization phase of
             of the program!

    Keyword arguments:
    level     -- minimum logging level to show
    fmt       -- format for records
    datefmt   -- format for the date in records
    """

    r = logging.root

    # remove existing handlers and filters
    r.handlers.clear()
    r.filters.clear()

    level = logging.NOTSET
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y%m%d %H:%M:%S"

    if "level" in kwargs:
        level = kwargs["level"]
    if "format" in kwargs:
        fmt = kwargs["format"]
    if "datefmt" in kwargs:
        datefmt = kwargs["datefmt"]

    r.level = level

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    stdout = logging.StreamHandler(stream=sys.stdout)
    stdout.name = "stdout"
    stdout.addFilter(NoDiagnosticsFilter(logging.WARNING))
    stdout.setFormatter(formatter)

    stderr = logging.StreamHandler(stream=sys.stderr)
    stderr.name = "stderr"
    stderr.level = logging.WARNING
    stderr.setFormatter(formatter)

    r.addHandler(stdout)
    r.addHandler(stderr)


def quick_config(verbosity, quiet, fmt='%(asctime)s [%(levelname)s] %(message)s', dtfmt='%j-%H:%M:%S'):
    """Wrapper for basic_config for quick configuration.

    Arguments:
    verbosity -- verbosity level: 0 = INFO, 1 = DEBUG, 2 = NOTSET
    quiet     -- if true only level >= WARNING will be shown
    fmt       -- format for records
    dtfmt     -- format for dates in records
    """
    if quiet:
        basic_config(level=logging.WARNING, format=fmt, datefmt=dtfmt)
    else:
        if verbosity == 0:
            basic_config(level=logging.INFO, format=fmt, datefmt=dtfmt)
        elif verbosity == 1:
            basic_config(level=logging.DEBUG, format=fmt, datefmt=dtfmt)
        elif verbosity >= 2:
            basic_config(level=logging.NOTSET, format=fmt, datefmt=dtfmt)
