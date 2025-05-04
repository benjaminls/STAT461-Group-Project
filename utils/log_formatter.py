import logging
import colorama

class ColoredFormatter(logging.Formatter):
    """A colored formatter for the main script."""
    COLORS = {
        logging.DEBUG: colorama.Fore.BLUE,
        logging.INFO: colorama.Fore.WHITE,
        logging.WARNING: colorama.Fore.YELLOW,
        logging.ERROR: colorama.Fore.LIGHTRED_EX,
        logging.CRITICAL: colorama.Fore.RED,
    }
    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        record.msg = f"{color}{record.msg}{colorama.Style.RESET_ALL}"
        return super().format(record)


class ColoredModuleFormatter(logging.Formatter):
    """A colored formatter for modules."""
    COLORS = {
        logging.DEBUG: colorama.Fore.BLUE,
        logging.INFO: colorama.Fore.WHITE,
        logging.WARNING: colorama.Fore.YELLOW,
        logging.ERROR: colorama.Fore.LIGHTRED_EX,
        logging.CRITICAL: colorama.Fore.RED,
    }
    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        if record.levelno == logging.DEBUG:
            record.msg = f"{color}({record.filename}:{record.lineno}) {record.msg}{colorama.Style.RESET_ALL}"
        elif record.levelno == logging.INFO:
            record.msg = f"{color}{record.msg}{colorama.Style.RESET_ALL}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{color}({record.filename}) {record.levelname} - {record.msg}{colorama.Style.RESET_ALL}"
        elif record.levelno in [logging.ERROR, logging.CRITICAL]:
            record.msg = f"{color}({record.filename}:{record.lineno}) {record.levelname} - {record.msg}{colorama.Style.RESET_ALL}"
        else:
            raise ValueError(f"logFormatter failed to handle invalid log level: {record.levelno}")
        return super().format(record)