from typing import Any, TypeVar, Union

_T = TypeVar("_T")
_U = TypeVar("_U")


def get_or(maybe: Union[_T, None], value: _U) -> Union[_T, _U]:
    return value if maybe is None else maybe
