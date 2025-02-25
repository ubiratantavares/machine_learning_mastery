import logging

# Create `parent.child` logger
logger = logging.getLogger("parent.child")

# Emit a log message of level INFO, by default this is not print to the screen
logger.info("this is info level")

# Create `parent` logger
parentlogger = logging.getLogger("parent")

# Set parent's level to INFO and assign a new handler
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
parentlogger.setLevel(logging.INFO)
parentlogger.addHandler(handler)

# Let child logger emit a log message again
logger.info("this is info level again")
