"""YAML config loading and strict dataclass construction."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, Mapping, Type, TypeVar, Union

import yaml

T = TypeVar("T")


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a YAML config file into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _freeze(value: Any) -> Any:
    """Convert (nested) lists to tuples so config dataclasses stay hashable-ish and immutable."""
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def build_config(cls: Type[T], data: Mapping[str, Any]) -> T:
    """Build dataclass `cls` from `data`, rejecting missing or unknown keys.

    Config dataclasses deliberately have no defaults: every value comes from YAML.
    """
    names = {f.name for f in dataclasses.fields(cls)}
    missing = names - set(data)
    unknown = set(data) - names
    if missing or unknown:
        raise ValueError(
            f"{cls.__name__} config mismatch: missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    return cls(**{k: _freeze(v) for k, v in data.items()})
