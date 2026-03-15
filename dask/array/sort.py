from __future__ import annotations

from collections import deque
from collections.abc import Generator, Iterable, Sequence
from itertools import product
from operator import getitem
from typing import Any

import numpy as np

from dask._task_spec import Alias, GraphNode, Task
from dask.array.core import Array
from dask.array.creation import arange, empty_like
from dask.array.dispatch import concatenate_lookup, partition_lookup, sort_lookup
from dask.array.numpy_compat import normalize_axis_index
from dask.array.slicing import normalize_index
from dask.base import tokenize
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
from dask.typing import Key, NestedKeys


def sort(a: Array, axis: int = -1, *, descending: bool = False, **kwargs: Any) -> Array:
    axis = normalize_axis_index(axis, a.ndim)
    out = _sort_impl(a, axis=axis, descending=descending, **kwargs)
    return out.map_blocks(
        sort_lookup,  # type: ignore[arg-type]
        axis=axis,
        dtype=a.dtype,
        meta=a._meta,
        **kwargs,
    )


def _normalize_kth(kth: int | Sequence[int], size: int) -> tuple[int, ...]:
    if isinstance(kth, Sequence):
        return normalize_index(tuple(sorted(kth)), (size,) * len(kth))
    else:
        return normalize_index(kth, (size,))


def partition(
    a: Array,
    kth: int | Sequence[int],
    axis: int = -1,
    *,
    descending: bool = False,
    **kwargs: Any,
) -> Array:
    axis = normalize_axis_index(axis, a.ndim)
    kth = _normalize_kth(kth, a.shape[axis])

    chunks_offsets = np.cumsum(a.chunks[axis])
    # Index of the chunk containing the highest kth point
    kth_chunks = np.searchsorted(chunks_offsets, kth)  #
    # Index of the kth points within their chunks
    kth_in_chunk = {
        kth_chunk_i: kth_i - chunks_offsets[kth_chunk_i - 1] if kth_chunk_i > 0 else 0
        for kth_i, kth_chunk_i in zip(kth, kth_chunks)
    }

    out = _sort_impl(
        a,
        axis=axis,
        kth_chunk=int(kth_chunks[-1]),
        descending=descending,
        **kwargs,
    )

    if not any(kth_in_chunk.values()):
        # All kth points are at index 0 within their chunk, nothing to do
        return out

    # Add an extra layer that calls np.partition internally on the chunk(s) that
    # contains the kth elements, and just passes through the other chunks.
    name = f"partition-{tokenize(out, kth)}"
    layer = {}
    for key in flatten(out.__dask_keys__()):
        assert isinstance(key, tuple)
        kth_i = kth_in_chunk.get(key[axis + 1], 0)
        if kth_i != 0:
            node: GraphNode = Task(
                (name, *key[1:]),
                partition_lookup,
                key,
                kth_i,
                axis=axis,
                **kwargs,
            )
        else:
            node = Alias((name, *key[1:]), key)
        layer[node.key] = node

    hlg = HighLevelGraph.from_collections(name, layer, [out])
    return Array(hlg, name, out.chunks, dtype=a.dtype, meta=a._meta, shape=a.shape)


def argsort(a: Array, axis: int = -1, *, descending: bool = False, **kwargs) -> Array:
    b = _add_idx_field(a, axis)
    c = sort(b, axis=axis, descending=descending, **kwargs)
    return c["__tmp_idx"]


def argpartition(
    a: Array, kth: int, axis: int = -1, *, descending: bool = False, **kwargs: Any
) -> Array:
    b = _add_idx_field(a, axis)
    c = partition(b, kth=kth, axis=axis, descending=descending, **kwargs)
    return c["__tmp_idx"]


def _add_idx_field(a: Array, axis: int) -> Array:
    """Convert a to a structured dtype and append a field containing
    the index along the given axis
    """
    if not isinstance(a._meta, np.ndarray):
        raise NotImplementedError("argsort and argpartition for non-NumPy arrays")

    if hasattr(a.dtype, "fields"):
        dtype = [(k, v[0]) for k, v in a.dtype.fields.items()]
    else:
        dtype = [("__orig", a.dtype)]
    dtype.append(("__tmp_idx", np.intp))
    b = empty_like(a, dtype=dtype)
    b[[d[0] for d in dtype[:-1]]] = a
    b["__tmp_idx"] = arange(
        a.shape[axis], chunks=a.chunks[axis], dtype=np.intp
    ).reshape(-1, *((1,) * (a.ndim - axis - 1)))
    return b


def _sort_impl(
    a: Array,
    axis: int,
    descending: bool,
    kth_chunk: int | None = None,
    **kwargs: Any,
) -> Array:
    """Common implementation of

    - sort / argsort
    - partition / argpartition
    - topk / argtopk for k > chunk size
    - topk / argtopk for k < -chunk size

    Return a reordered array where all points of the n-th chunk are <= the points of the
    next chunks, but the chunks are not internally ordered.

    Parameters
    ----------
    descending: bool
        Set to True to sort in descending order. It's important to implement this at low
        level and not just do [::-1] on the output when this is used to implement topk
        with n>0, as it relies on culling to remove unnecessary chunks, and the nth
        output chunk depends on all previous chunks.
    kth_chunk: int
        Index of the chunk containing the kth point of partition (or the max kth point,
        when kth is a sequence). Chunks beyond this index will be replicated verbatim.
    kwargs:
        Passed to the partition function, e.g. `order` for structured arrays.

    The algorithm is embarassingly parallel along all other axes.
    """
    if len(a.chunks[axis]) < 2:
        return a

    if kth_chunk is None:
        kth_chunk = len(a.chunks[axis])
    chunks_idxs = [[i for i, _ in enumerate(c)] for c in a.chunks]
    chunks_idxs[axis] = [slice(None)]  # type: ignore[list-item]
    parent_nodes = np.array(_nested_keys_to_aliases(a.__dask_keys__()))
    name = f"sort-partial-{tokenize(a, axis, descending, kwargs)}"
    # TODO write a specialized Layer to delay materialization
    layer = {
        task.key: task
        for idx in product(*chunks_idxs)
        for task in _sort_impl_row(
            name, axis, descending, kth_chunk, idx, parent_nodes[idx], **kwargs
        )
    }
    hlg = HighLevelGraph.from_collections(name, layer, [a])
    return Array(hlg, name, a.chunks, dtype=a.dtype, meta=a._meta, shape=a.shape)


def _nested_keys_to_aliases(key: Key | NestedKeys) -> Any:
    if isinstance(key, list):
        return [_nested_keys_to_aliases(k) for k in key]
    return Alias(key)


def _sort_impl_row(
    name: str,
    axis: int,
    descending: bool,
    kth_chunk: int,
    idx: tuple[int, ...],
    input_nodes: Iterable[GraphNode],
    **kwargs: Any,
) -> Generator[GraphNode]:

    def gen_key(i: int) -> Generator[Key]:
        while True:
            yield (name, *idx[:axis], i, *idx[axis + 1 :])
            i += 1

    left = deque(input_nodes)
    right: list[Task] = []
    out_key_iter = gen_key(0)
    mid_key_iter = gen_key(len(left))
    n_out = 0

    while left:
        while len(left) > 1:
            step_task = Task(
                next(mid_key_iter),
                _sort_partial,
                left.popleft().ref(),
                left.popleft().ref(),
                axis=axis,
                descending=descending,
                **kwargs,
            )
            left_task = Task(next(mid_key_iter), getitem, step_task.ref(), 0)
            right_task = Task(next(mid_key_iter), getitem, step_task.ref(), 1)
            yield step_task
            yield left_task
            yield right_task
            left.append(left_task)
            right.append(right_task)

        yield Alias(next(out_key_iter), left.popleft().ref())
        if n_out == kth_chunk:
            for node in right:
                yield Alias(next(out_key_iter), node.ref())
            return
        n_out += 1

        left.extend(right)
        right.clear()


def _sort_partial(
    a: np.ndarray,
    b: np.ndarray,
    axis: int,
    descending: bool,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Task kernel for all sort algorithms, with the exception of
    topk when k is smaller than a single chunk.

    Parameters
    ----------
    a, b: array
        Two arrays to sort
    axis: int
        Axis along which to sort
    descending: bool
        Set to True to sort in descending order
    kwargs:
        Passed to the partition function, e.g. `order` for structured arrays.

    Returns
    -------
    - left: array
        The smallest (or largest, if descending) a.shape[axis] elements of
        a and b, in arbitrary internal order
    - right: array
        The remaining elements of a and b, in arbitrary internal order
    """
    concatenate = concatenate_lookup.dispatch(
        type(max([a, b], key=lambda x: getattr(x, "__array_priority__", 0)))
    )
    c = concatenate([a, b], axis=axis)
    kth = a.shape[axis]
    idx_pre = [slice(None)] * axis
    idx_left = (*idx_pre, slice(None, kth))
    idx_right = (*idx_pre, slice(kth, None))
    if descending:
        kth = c.shape[axis] - kth
        idx_left, idx_right = idx_right, idx_left

    if hasattr(c, "partition"):
        # NumPy and CuPy can partition in-place
        c.partition(kth, axis=axis, **kwargs)
    else:
        c = partition_lookup(c, kth, axis=axis, **kwargs)

    return c[idx_left], c[idx_right]
