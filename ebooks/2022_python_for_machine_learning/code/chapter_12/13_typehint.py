from typing import Any, Union, List

def square(x: Union[int, float]) -> Union[int, float]:
    return x * x

def append(x: List[Any], y: Any) -> None:
    x.append(y)
