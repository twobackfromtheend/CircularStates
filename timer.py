import time
from contextlib import contextmanager


@contextmanager
def timer(process: str):
    start_time = time.time()
    print(f"{process} started.")
    try:
        yield
    finally:
        print(f"{process} took {time.time() - start_time:.3f}s")
