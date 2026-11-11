import logging
from collections.abc import Callable
from functools import wraps
from typing import Concatenate, Protocol

logger = logging.getLogger("exfa")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)


class SupportsVerbose(Protocol):
    @property
    def verbose(self) -> bool: ...


def log_method[T: SupportsVerbose, **P, R](
    method: Callable[Concatenate[T, P], R],
) -> Callable[Concatenate[T, P], R]:
    """
    Log the method name before execution when verbosity is enabled.
    """

    @wraps(method)
    def wrapper(self: T, /, *args: P.args, **kwargs: P.kwargs) -> R:
        if self.verbose:
            logger.info("Calling %s", wrapper.__qualname__)
        return method(self, *args, **kwargs)

    return wrapper
