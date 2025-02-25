import logging

# Set up root logger, and add a file handler to root logger
logging.basicConfig(filename = 'file.log',
                    level = logging.WARNING,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

# Create logger, set level, and add stream handler
parent_logger = logging.getLogger("parent")
parent_logger.setLevel(logging.INFO)
parent_shandler = logging.StreamHandler()
parent_logger.addHandler(parent_shandler)

# Log message of severity INFO or above will be handled
parent_logger.debug('Debug message')
parent_logger.info('Info message')
parent_logger.warning('Warning message')
parent_logger.error('Error message')
parent_logger.critical('Critical message')
