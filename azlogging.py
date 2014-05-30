import logging
import sys

class NoDiagnosticsFilter(logging.Filter):

    def __init__(self, cutofflevel):
        self.cutofflevel = cutofflevel

    def filter(self, record):
        return record.levelno < self.cutofflevel


# this is NOT thread safe! use only at the initialization phase of the program...
def basic_config(**kwargs):
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