import sys

def progress_bar(progress:int, total:int, length:int=50):
    """Print a progress bar to the console.

    Args:
        progress (int): The current progress.
        total (int): The total progress.
        length (int): The length of the progress bar.
    """
    percent = 100 * (progress / float(total))
    filled_length = int(length * progress // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r|{bar}| {percent:.2f}%')
    sys.stdout.flush()