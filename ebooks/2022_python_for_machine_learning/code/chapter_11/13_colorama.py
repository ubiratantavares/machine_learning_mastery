import logging
import colorama
from colorama import Fore, Back, Style

# Initialize the terminal for color
colorama.init(autoreset = True)

# Set up logger as usual
logger = logging.getLogger("color")
logger.setLevel(logging.DEBUG)
shandler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
shandler.setFormatter(formatter)
logger.addHandler(shandler)

# Emit log message with color
logger.debug('Debug message')
logger.info(Fore.GREEN + 'Info message')
logger.warning(Fore.BLUE + 'Warning message')
logger.error(Fore.YELLOW + Style.BRIGHT + 'Error message')
logger.critical(Fore.RED + Back.YELLOW + Style.BRIGHT + 'Critical message')
