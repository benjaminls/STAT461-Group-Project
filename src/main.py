import sys
import os
import argparse
import logging
import colorama # for nice colored stdout output

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # now we can import from src

from utils.log_formatter import ColoredFormatter


def _parse_args():
    parser = argparse.ArgumentParser(description="Train a GCN for track edge‐classification")
    parser.add_argument(
        "--hits",
        type=str,
        help="Path to hits CSV",
    )
    parser.add_argument(
        "--truth",
        type=str,
        help="Path to truth CSV",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--logLevel",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    return parser.parse_args()


# class ColoredFormatter(logging.Formatter):
#     """A minimal colored formatter.
#     Reused from """
#     COLORS = {
#         logging.DEBUG: colorama.Fore.BLUE,
#         logging.INFO: colorama.Fore.WHITE,
#         logging.WARNING: colorama.Fore.YELLOW,
#         logging.ERROR: colorama.Fore.LIGHTRED_EX,
#         logging.CRITICAL: colorama.Fore.RED,
#     }
#     def format(self, record):
#         color = self.COLORS.get(record.levelno, "")
#         record.msg = f"{color}{record.msg}{colorama.Style.RESET_ALL}"
#         return super().format(record)



def _init_logger(cl_args: argparse.Namespace):
    """Logger setup for main file.

    Args:
        cl_args (argparse.Namespace): Command line arguments.

    Returns:
        logging.Logger: logger object
    """
    console_handler = logging.StreamHandler()
    level = getattr(logging, cl_args.logLevel.upper(), logging.INFO)
    console_handler.setLevel(level)
    # formatter = logging.Formatter(
    #     "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # )
    formatter = ColoredFormatter("%(message)s") # imported from utils/log_formatter.py
    console_handler.setFormatter(formatter)
    new_logger = logging.getLogger(__name__)
    new_logger.addHandler(console_handler)
    new_logger.setLevel(level)

    # Set logger environment variable
    os.environ["PYTHON_GNN_LOG_LEVEL"] = cl_args.logLevel.upper()

    return new_logger



args = _parse_args()
logger = _init_logger(args)

logger.info("Arguments:")
for arg, value in vars(args).items():
    logger.info(f"  {arg}: {value}")



from utils.network import run_training # utils.network logging is initialized at this point

if __name__ == "__main__":
    run_training(
        args.hits,
        args.truth,
        epochs=args.epochs,
        lr=args.lr,
    )
