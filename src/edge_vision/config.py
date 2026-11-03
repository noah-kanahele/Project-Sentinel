"""YAML config loading and strict dataclass construction."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, Mapping, Type, TypeVar, Union

import yaml

T = TypeVar("T")


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively overlay `override` onto `base` (returns a new dict)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a YAML config file into a plain dict.

    A config may set `extends: other.yaml` (path relative to the file) to inherit another
    config and override only some values.
    """
    path = Path(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    parent = data.pop("extends", None)
    if parent is not None:
        return _deep_merge(load_config(path.parent / parent), data)
    return data


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
