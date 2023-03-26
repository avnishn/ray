import pytest
import time

import numpy as np
import pandas as pd
import pyarrow as pa

import ray
from ray.data._internal.block_batching.util import (
    make_async_gen,
    blocks_to_batches,
    format_batches,
    collate,
    resolve_block_refs,
)
from ray.data._internal.block_batching.interfaces import Batch


def block_generator(num_rows: int, num_blocks: int):
    for _ in range(num_blocks):
        yield pa.table({"foo": [1] * num_rows})


def test_resolve_block_refs(ray_start_regular_shared):
    block_refs = [ray.put(0), ray.put(1), ray.put(2)]

    resolved_iter = resolve_block_refs(iter(block_refs))
    assert list(resolved_iter) == [0, 1, 2]


@pytest.mark.parametrize("block_size", [1, 10])
@pytest.mark.parametrize("drop_last", [True, False])
def test_blocks_to_batches(block_size, drop_last):
    num_blocks = 5
    block_iter = block_generator(num_rows=block_size, num_blocks=num_blocks)

    batch_size = 3
    batch_iter = list(
        blocks_to_batches(block_iter, batch_size=batch_size, drop_last=drop_last)
    )

    if drop_last:
        for batch in batch_iter:
            assert len(batch.data) == batch_size
    else:
        full_batches = 0
        leftover_batches = 0

        dataset_size = block_size * num_blocks
        for batch in batch_iter:
            if len(batch.data) == batch_size:
                full_batches += 1
            if len(batch.data) == (dataset_size % batch_size):
                leftover_batches += 1

        assert leftover_batches == 1
        assert full_batches == (dataset_size // batch_size)

    assert [batch.batch_idx for batch in batch_iter] == list(range(len(batch_iter)))


@pytest.mark.parametrize("batch_format", ["pandas", "numpy", "pyarrow"])
def test_format_batches(batch_format):
    block_iter = block_generator(num_rows=2, num_blocks=2)
    batch_iter = (Batch(i, block) for i, block in enumerate(block_iter))
    batch_iter = list(format_batches(batch_iter, batch_format=batch_format))

    for batch in batch_iter:
        if batch_format == "pandas":
            assert isinstance(batch.data, pd.DataFrame)
        elif batch_format == "arrow":
            assert isinstance(batch.data, pa.Table)
        elif batch_format == "numpy":
            assert isinstance(batch.data, dict)
            assert isinstance(batch.data["foo"], np.ndarray)

    assert [batch.batch_idx for batch in batch_iter] == list(range(len(batch_iter)))


def test_collate():
    def collate_fn(batch):
        return pa.table({"bar": [1] * 2})

    batches = [
        Batch(i, data)
        for i, data in enumerate(block_generator(num_rows=2, num_blocks=2))
    ]
    batch_iter = collate(batches, collate_fn=collate_fn)

    for i, batch in enumerate(batch_iter):
        assert batch.batch_idx == i
        assert batch.data == pa.table({"bar": [1] * 2})


def test_make_async_gen_fail():
    """Tests that any errors raised in async threads are propagated to the main
    thread."""

    def gen(base_iterator):
        raise ValueError("Fail")

    iterator = make_async_gen(base_iterator=iter([1]), fn=gen)

    with pytest.raises(ValueError) as e:
        for _ in iterator:
            pass

    assert e.match("Fail")


def test_make_async_gen():
    """Tests that make_async_gen overlaps compute."""

    num_items = 10

    def gen(base_iterator):
        for i in base_iterator:
            time.sleep(2)
            yield i

    def sleep_udf(item):
        time.sleep(3)
        return item

    iterator = make_async_gen(
        base_iterator=iter(range(num_items)), fn=gen, num_workers=1
    )

    start_time = time.time()
    outputs = []
    for item in iterator:
        print(item)
        outputs.append(sleep_udf(item))
    end_time = time.time()

    assert outputs == list(range(num_items))

    # Three second buffer.
    assert end_time - start_time < num_items * 3 + 3


def test_make_async_gen_multiple_threads():
    """Tests that using multiple threads can overlap compute even more."""

    num_items = 5

    def gen(base_iterator):
        for i in base_iterator:
            time.sleep(4)
            yield i

    def sleep_udf(item):
        time.sleep(5)
        return item

    # All 5 items should be fetched concurrently.
    iterator = make_async_gen(
        base_iterator=iter(range(num_items)), fn=gen, num_workers=5
    )

    start_time = time.time()

    # Only sleep for first item.
    sleep_udf(next(iterator))

    # All subsequent items should already be prefetched and should be ready.
    for _ in iterator:
        pass
    end_time = time.time()

    # 4 second for first item, 5 seconds for udf, 0.5 seconds buffer
    assert end_time - start_time < 9.5


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
