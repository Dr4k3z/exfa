import logging
from collections.abc import Callable
from functools import wraps
from typing import Protocol, cast

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


def log_method[Method: Callable[..., object]](method: Method) -> Method:
    """
    Log the method name before execution when verbosity is enabled.
    """

    @wraps(method)
    def wrapper(self: SupportsVerbose, /, *args: object, **kwargs: object) -> object:
        if self.verbose:
            logger.info("Calling %s", wrapper.__qualname__)
        return method(self, *args, **kwargs)

    return cast(Method, wrapper)
