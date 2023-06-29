import os
import yaml
import pickle
import inspect
import datetime
import contextlib
import joblib

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def filter_args(func, d):
    """Filter dictionary keys to match the function arguments.
    Arguments:
        - func: function
        - d: dict
    Returns:
        - args: dict
    """
    keys = inspect.getfullargspec(func).args
    args = {key: d[key] for key in keys if key != "self"}
    return args


def load_pickle(filename):
    """Load pickle file with PROTOCOL 5.
    Args:
        - filename: str
    Return:
        - data (loaded file)
    """
    with open(filename, "rb") as inp:
        data = pickle.load(inp)
    return data


def save_pickle(filename, data):
    """Load pickle file with PROTOCOL 5.
    Args:
        - filename: str
        - data (objcet to save)
    """
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)


def check_folder(path):
    """Create adequate folders if necessary."""
    try:
        if not os.path.isdir(path):
            check_folder(os.path.dirname(path))
            os.mkdir(path)
    except:
        pass


def read_yaml(yaml_path):
    """Open and read safely a yaml file."""
    try:
        with open(yaml_path, "r") as stream:
            parameters = yaml.safe_load(stream)
        return parameters
    except:
        print("Couldn't load yaml file: {}.".format(yaml_path))


def save_yaml(data, yaml_path):
    """Open and write safely in a yaml file.
    Arguments:
        - data: list/dict/str/int/float
        -yaml_path: str
    """
    with open(yaml_path, "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def write(path, text, end=""):  # end='\n'
    """Write in the specified text file."""
    with open(path, "a+") as f:
        f.write(text)
        f.write(end)


def get_timestamp(time_format="%d-%b-%Y (%H:%M:%S)"):
    """Return string timestamp.
    Returns:
        - string
    """
    return datetime.datetime.now(tz=None).strftime(time_format)


def format_time(elapsed):
    """Takes a time in seconds and returns a string hh:mm:ss"""
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# `rich` console used throughout the codebase
console = Console()


# `rich` progress bar used throughout the codebase
def get_progress(**kwargs):
    return Progress(
        SpinnerColumn(),
        TaskProgressColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        "<",
        TimeRemainingColumn(),
        **kwargs,
    )


@contextlib.contextmanager
def rich_progress_joblib(description=None, total=None, verbose=False):
    if description is None:
        description = "Processing..."

    progress = get_progress()
    if verbose:
        task_id = progress.add_task(description, total=total)

    class BatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            if verbose:
                progress.update(task_id, advance=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack

    try:
        joblib.parallel.BatchCompletionCallBack = BatchCompletionCallback
        progress.start()

        yield progress
    finally:
        progress.stop()
        joblib.parallel.BatchCompletionCallBack = old_callback
