from itertools import tee
from typing import Union, Callable


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def str2callable(cls: Union[str, Callable], restriction: str = "") -> Callable:
    """Dynamically import a callable from a string."""
    if isinstance(cls, Callable):  # type: ignore[arg-type]
        return cls  # type: ignore[return-value]
    if isinstance(cls, str) and not cls.startswith(restriction):
        raise ValueError(
            f"Failed to import '{cls}' because "
            f"imports are restricted to '{restriction}'."
        )

    module_name, class_name = cls.rsplit(".", 1)  # type: ignore[union-attr]
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)

    return cls  # type: ignore[return-value]
