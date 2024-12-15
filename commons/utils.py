from collections.abc import Iterable
from typing import Any
import re

def regex_find(lst: Iterable[Any], pattern: re.Pattern) -> Iterable[tuple[int, Any]]:
    yield from ((i, x) for i, x in enumerate(lst) if re.fullmatch(pattern, str(x)))
