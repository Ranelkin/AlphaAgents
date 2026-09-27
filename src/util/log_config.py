import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo


class SecurityFilter(logging.Filter):
    def __init__(self, blacklist=None):
        super().__init__()
        self.blacklist = list(blacklist or [])

    def filter(self, record):
        message = record.getMessage()
        for item in self.blacklist:
            message = message.replace(item, "*" * len(item))
        record.msg = message
        record.args = ()

        return True


def setup_logging(
    app_name,
    log_level=logging.INFO,
    log_format=None,
    rotation_type="size",
    max_bytes=1_048_576,
    backup_count=10,
    when="midnight",
    interval=1,
    utc=False,
    console_output=True,
    file_output=True,
    security_filter=None,
):
    logger = logging.getLogger(app_name)
    logger.setLevel(log_level)
    logger.handlers.clear()

    berlin = ZoneInfo("Europe/Berlin")
    formatter = logging.Formatter(
        log_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z",
    )

    formatter.converter = lambda timestamp: datetime.fromtimestamp(
        timestamp, berlin
    ).timetuple()

    # Start with an empty list.
    handlers = []

    if console_output:
        handlers.append(logging.StreamHandler(sys.stderr))
    if file_output:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        path = log_dir / f"{app_name}.log"
        if rotation_type == "size":
            handlers.append(
                RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
            )
        elif rotation_type == "time":
            handlers.append(
                TimedRotatingFileHandler(
                    path,
                    when=when,
                    interval=interval,
                    backupCount=backup_count,
                    utc=utc,
                )
            )
        else:
            raise ValueError("rotation_type must be 'size' or 'time'")

    redact = SecurityFilter(security_filter)
    for handler in handlers:
        handler.setFormatter(formatter)
        if security_filter:
            handler.addFilter(redact)
        logger.addHandler(handler)

    def handle_exception(exc_type, exc_value, traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, traceback))

    sys.excepthook = handle_exception
    return logger
