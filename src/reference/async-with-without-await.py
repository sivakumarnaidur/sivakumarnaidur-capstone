#!/usr/bin/env python3
"""
Quick demo: calling two async functions one after another,
WITH await vs WITHOUT await.
Run: python3 src/reference/async-with-without-await.py
"""

import asyncio


async def task_one():
    print("task_one: started")
    await asyncio.sleep(1)
    print("task_one: finished")
    return "result-1"


async def task_two():
    print("task_two: started")
    await asyncio.sleep(1)
    print("task_two: finished")
    return "result-2"


async def with_await_example():
    print("\n=== WITH await ===")
    r1 = await task_one()   # runs task_one fully, waits for it to complete
    r2 = await task_two()   # then runs task_two fully
    print("results:", r1, r2)


def without_await_example():
    print("\n=== WITHOUT await ===")
    r1 = task_one()   # just creates a coroutine object, body does NOT run
    r2 = task_two()   # same here
    print("r1 is:", r1)   # prints something like <coroutine object task_one at 0x...>
    print("r2 is:", r2)
    # Neither "started"/"finished" print happens above, and r1/r2 are NOT "result-1"/"result-2".
    r1.close()   # avoid "coroutine was never awaited" warnings since we never run them
    r2.close()


if __name__ == "__main__":
    without_await_example()          # plain call: bodies never execute
    asyncio.run(with_await_example())  # proper call: bodies execute in order
