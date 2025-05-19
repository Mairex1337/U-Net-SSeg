import time


class Timer:
    """Context manager for measuring elapsed time in seconds."""

    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.time() - self.start