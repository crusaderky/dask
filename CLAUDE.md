# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install for development:**
```bash
pip install -e ".[complete,test]"
pre-commit install
```

**Run tests:**
```bash
# All tests
python -m pytest dask --runslow

# Single test file
python -m pytest dask/array/tests/test_reductions.py

# Single test
python -m pytest dask/array/tests/test_reductions.py::test_nanquantile

# With array-expr backend (new expression-based array backend)
python -m pytest dask --runarrayexpr

# Parallel execution
python -m pytest dask -n4
```

**Linting and formatting:**
```bash
pre-commit run --all-files   # ruff-check, black, mypy, codespell
ruff check dask/
black dask/
mypy dask/                   # type-checking (excludes dask/dataframe/dask_expr/)
```

## Architecture

Dask is a parallel computing library built around **lazy task graphs**. The core abstraction is that collections (Array, DataFrame, Bag, Delayed) build a graph of tasks without executing them; calling `.compute()` triggers execution.

### Task Graph Layer (`dask/`)

- **`_task_spec.py`** — Low-level task primitives: `Task`, `DataNode`, `TaskRef`, `Alias`. A `Task` wraps a callable with arguments; `TaskRef` is a lazy reference to another task by key. This is the execution substrate.
- **`core.py`** — Primitive graph operations: `get()` (synchronous scheduler), `ishashable()`, `istask()`, `flatten()`, `reverse_dict()`, etc.
- **`base.py`** — `DaskMethodsMixin`, `compute()`, `persist()`, `tokenize()`, `normalize_token()`. The `DaskMethodsMixin` is the base class for all dask collections.
- **`tokenize.py`** — Deterministic hashing of Python objects for cache-key and graph-name generation. Custom tokenizers registered via `normalize_token`.
- **`highlevelgraph.py`** — `HighLevelGraph` (dict of named `Layer` objects) and the `Layer` ABC. Layers stay symbolic until materialized; `Blockwise` is the most important layer type.
- **`blockwise.py`** — `Blockwise` layer for element-wise operations across chunks/partitions. Central to how array and dataframe operations are expressed without materializing the full graph.
- **`order.py`** — Static task ordering algorithm that minimizes peak memory usage during execution.
- **`local.py`** — Synchronous and threaded schedulers (used by default).

### Expression IR (`dask/_expr.py`, `dask/*/dask_expr/`, `dask/array/_array_expr/`)

Both DataFrame and Array now have an **expression-based backend** that adds a logical/physical optimizer pipeline before lowering to a task graph. The `Expr` base class in `dask/_expr.py` defines the expression tree protocol with optimizer stages: `logical → simplified-logical → tuned-logical → physical → simplified-physical → fused`.

- **DataFrame expressions**: `dask/dataframe/dask_expr/` — `_expr.py` defines the `Expr` subclass for dataframes with pandas metadata (`_meta`), divisions, and partition-aware operations.
- **Array expressions**: `dask/array/_array_expr/` — `_expr.py` defines `ArrayExpr` with chunk/dtype/shape properties.
- **`SingletonExpr`** (in `dask/_expr.py`) — Ensures structural deduplication: identical expression trees return the same object.

### Collections

- **`dask/array/`** — `dask.array.Array`. `core.py` holds the `Array` class and most public API. The `_array_expr/` subdirectory contains the new expression-based implementation; `dask/array/core.py` bridges both backends.
- **`dask/dataframe/`** — `dask.dataframe.DataFrame/Series/Index`. `core.py` holds the legacy implementation; `dask_expr/` holds the expression-based one (now the default). Divisions track the pandas index range per partition.
- **`dask/bag/`** — `dask.bag.Bag` for unordered collections of Python objects.
- **`dask/delayed.py`** — `dask.delayed` decorator and `Delayed` class for arbitrary Python functions.

### Schedulers

Dask ships three built-in schedulers (selected via `dask.config` or the `scheduler=` argument to `.compute()`):
- `synchronous` / `single-threaded` — `dask/local.py`, for debugging.
- `threads` — `dask/local.py` using `concurrent.futures.ThreadPoolExecutor`.
- `processes` — `dask/multiprocessing.py`.
- `distributed` — External `distributed` package (optional dependency).

### Configuration

`dask/config.py` provides a layered config system (YAML files + env vars `DASK_*` + `dask.config.set()`). The schema lives in `dask/dask-schema.yaml`.

## Key Conventions

- **Test files**: named `test_*.py` inside `tests/` subdirectory of each subpackage.
- **pytest markers**: `slow` (skipped by default; use `--runslow`), `network`, `gpu`, `array_expr`, `normal_and_array_expr`.
- **`filterwarnings = "error"`** in pytest config — all warnings are errors in tests. New warnings in tested code must be either fixed or explicitly suppressed in `pyproject.toml`.
- **`xfail_strict = true`** — xfail tests that start passing will fail CI.
- **Line length**: 120 (ruff/black).
- **`mypy`** is run in pre-commit but only covers `dask/` excluding `dask/dataframe/dask_expr/`.
- **Tokenization** for graph keys uses `dask.base.tokenize()` — always deterministic. Custom objects need a `normalize_token` dispatch or `__dask_tokenize__` method.
