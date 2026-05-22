from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CostOpKind = Literal["constant", "addf", "subf", "mulf"]


@dataclass(frozen=True)
class CostOperation:
    result: str
    kind: CostOpKind
    operands: tuple[str, ...] = ()
    value: float | None = None


@dataclass(frozen=True)
class CostFunction:
    variables: list[str]
    operations: list[CostOperation]
    return_value: str
