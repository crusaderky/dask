from __future__ import annotations

from types import ModuleType

import pytest

pytestmark = pytest.mark.normal_and_array_expr

pytest.importorskip("numpy")

DA_EXPORTED_SUBMODULES = {"backends", "fft", "lib", "linalg", "ma", "overlap", "random"}


def test_api():
    """Tests that `dask.array.__all__` is correct"""
    import dask.array as da

    member_dict = vars(da)
    members = set(member_dict)
    # unexported submodules
    ignore_modules = {
        m
        for m, mod in member_dict.items()
        if m not in DA_EXPORTED_SUBMODULES
        and isinstance(mod, ModuleType)
    }
    members -= ignore_modules
    # private utilities and `__dunder__` members
    members -= {"annotations", "ARRAY_EXPR_ENABLED", "raise_not_implemented_error"}
    members -= {m for m in members if m.startswith("_")}

    assert set(da.__all__) == members
