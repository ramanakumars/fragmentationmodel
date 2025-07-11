import logging

logger = logging.getLogger(__name__)
# define the format for the log messages, here: "level name: module -  message"
formatter = logging.Formatter("[%(levelname)s]: %(name)s -  %(message)s")
if logger.handlers:
    logger.handlers = []

# define console handler to write log messages to stdout
sh = logging.StreamHandler()
sh.setFormatter(formatter)
sh.setLevel(logging.NOTSET)

# add the handlers to the logger
logger.addHandler(sh)
