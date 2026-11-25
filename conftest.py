"""pytest 入口：每个测试函数通过参数 `m` 拿到被测模块。

默认测你自己写的 template.py；想先确认参考答案能过，用
    RH_IMPL=solution uv run pytest
"""
import importlib.util
import os
import pathlib

import pytest

_CACHE = {}


def load_impl(exercise_dir, which):
    path = pathlib.Path(exercise_dir) / f"{which}.py"
    key = str(path)
    if key not in _CACHE:
        name = f"rh_{path.parent.parent.name}_{path.parent.name}_{which}"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CACHE[key] = mod
    return _CACHE[key]


@pytest.fixture
def m(request):
    which = os.environ.get("RH_IMPL", "template")
    return load_impl(pathlib.Path(str(request.fspath)).parent, which)
