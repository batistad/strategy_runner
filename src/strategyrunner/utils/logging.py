import logging
import sys

FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(stream=sys.stdout, level=getattr(logging, level), format=FMT)
