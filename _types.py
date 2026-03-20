from __future__ import annotations
import typing as t
import numpy as np
import numpy.typing as npt

if t.TYPE_CHECKING:
    Float: t.TypeAlias = t.Union[npt.NDArray[np.floating], float]
    Int: t.TypeAlias = t.Union[npt.NDArray[np.integer], int]
else:
    Float = t.Union[float, np.ndarray]
    Int = t.Union[int, np.ndarray]
