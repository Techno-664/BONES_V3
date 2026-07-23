from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _prompt_raw(msg: str, default: Any = None) -> str:
    hint = f" [{default}]" if default is not None else ""
    raw = input(f"{msg}{hint}: ").strip()
    return raw


def prompt_int(msg: str, default: int | None = None, min_val: int | None = None, max_val: int | None = None) -> int | None:
    while True:
        raw = _prompt_raw(msg, default)
        if not raw:
            return default
        try:
            val = int(raw)
        except ValueError:
            print("  Invalid input: expected an integer")
            continue
        if min_val is not None and val < min_val:
            print(f"  Value must be >= {min_val}")
            continue
        if max_val is not None and val > max_val:
            print(f"  Value must be <= {max_val}")
            continue
        return val


def prompt_float(msg: str, default: float | None = None, min_val: float | None = None, max_val: float | None = None) -> float:
    while True:
        raw = _prompt_raw(msg, default)
        if not raw and default is not None:
            return default
        try:
            val = float(raw)
        except ValueError:
            print("  Invalid input: expected a number")
            continue
        if min_val is not None and val < min_val:
            print(f"  Value must be >= {min_val}")
            continue
        if max_val is not None and val > max_val:
            print(f"  Value must be <= {max_val}")
            continue
        return val


def prompt_choice(msg: str, options: dict[str, str], default: str | None = None) -> str:
    print(msg)
    keys = list(options.keys())
    for i, k in enumerate(keys, 1):
        print(f"  {i}) {options[k]}")
    while True:
        raw = _prompt_raw("Enter number", default)
        if not raw and default is not None:
            return default
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
        except ValueError:
            pass
        print(f"  Invalid choice, enter 1-{len(keys)}")


def prompt_bool(msg: str, default: bool = False) -> bool:
    hint = " [Y/n]" if default else " [y/N]"
    raw = input(f"{msg}{hint}: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def prompt_path(msg: str, must_exist: bool = False, default: str | None = None) -> Path | None:
    while True:
        raw = _prompt_raw(msg, default)
        if not raw and default is not None:
            p = Path(default)
            if must_exist and not p.exists():
                print(f"  Path not found: {p}")
                continue
            return p
        if not raw:
            return None
        p = Path(raw)
        if must_exist and not p.exists():
            print(f"  Path not found: {raw}")
            continue
        return p


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)
