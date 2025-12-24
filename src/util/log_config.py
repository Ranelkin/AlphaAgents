import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import os, sys, pytz, datetime


class SecurityFilter(logging.Filter):
    def __init__(self, blacklist=None):
        super().__init__()
        self.blacklist = blacklist or []

    def filter(self, record):
        if isinstance(record.msg, str):
            for item in self.blacklist:
                record.msg = record.msg.replace(item, '*' * len(item))
        return True

def setup_logging(app_name, log_level=logging.INFO, log_format=None, 
                  rotation_type='size', max_bytes=1048576, backup_count=10,
                  when='midnight', interval=1, utc=False, 
                  console_output=True, file_output=True,
                  security_filter=None):
    """
    Setup logging configuration.

    :param app_name: Name of the application (used for log file naming)
    :param log_level: Logging level
    :param log_format: Custom log format (if None, default format is used)
    :param rotation_type: Type of log rotation ('size' or 'time')
    :param max_bytes: Maximum size of log file before rotation (for 'size' rotation)
    :param backup_count: Number of backup files to keep
    :param when: When to rotate logs (for 'time' rotation)
    :param interval: Interval of rotation (for 'time' rotation)
    :param utc: Use UTC time for rotation (for 'time' rotation)
    :param console_output: Whether to output logs to console
    :param file_output: Whether to output logs to file
    :param security_filter: List of sensitive strings to be filtered out
    """
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a logger
    logger = logging.getLogger(app_name)
    logger.setLevel(log_level)

    # Remove any existing handlers to avoid duplicate logging
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    berlin_tz = pytz.timezone('Europe/Berlin')
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S %Z')
    formatter.converter = lambda *args: berlin_tz.localize(datetime.datetime.now()).timetuple()
    handlers = []

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    # File handler
    if file_output:
        log_file = os.path.join(log_dir, f'{app_name}.log')
        if rotation_type == 'size':
            file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        elif rotation_type == 'time':
            file_handler = TimedRotatingFileHandler(log_file, when=when, interval=interval, 
                                                    backupCount=backup_count, utc=utc)
        else:
            raise ValueError("rotation_type must be either 'size' or 'time'")
        
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Add handlers to the logger
    for handler in handlers:
        logger.addHandler(handler)

    # Add security filter if provided
    if security_filter:
        for handler in handlers:
            handler.addFilter(SecurityFilter(security_filter))

    # Exception logging
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    return logger

# Usage example:
if __name__ == '__main__':
    logger = setup_logging('example_app', log_level=logging.DEBUG, 
                           rotation_type='time', when='D', interval=1, 
                           security_filter=['password', 'secret_key'])
    logger.info('This is an info message')
    logger.debug('This is a debug message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')
    logger.info('User password: password123')  # This should be filtered