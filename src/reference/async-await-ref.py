#!/usr/bin/env python3
"""
Practical async/await/asyncio reference with runnable examples.
Run: python3 src/reference/async-await-ref.py
"""

import asyncio
import os
import time


# ---------------------------------------------------------------------------
# 1. asyncio.run(): the standard entry point that creates an event loop,
#    runs the given coroutine to completion, and closes the loop.
#    It can only be called once, and NOT from inside an already-running loop.
# ---------------------------------------------------------------------------
async def greet() -> str:
    await asyncio.sleep(0.2)
    return "asyncio.run() executed this coroutine"


def asyncio_run_example():
    result = asyncio.run(greet())   # standalone call: no event loop exists yet here
    print(result)

    # If a loop is ALREADY running (e.g. called from inside another coroutine,
    # or in a Jupyter notebook), asyncio.run() is not available/raises RuntimeError.
    # Fallback: reuse the running loop instead of starting a new one.
    async def _inside_running_loop():
        coro = greet()
        try:
            asyncio.run(coro)             # not allowed here -> raises RuntimeError
        except RuntimeError as e:
            coro.close()                  # avoid "coroutine was never awaited" warning
            print(f"asyncio.run() not available here ({e}); awaiting directly instead")
            result = await greet()        # fallback: just await it directly
            print(result)

    asyncio.run(_inside_running_loop())


# ---------------------------------------------------------------------------
# 2. A coroutine: defined with `async def`, paused/resumed with `await`.
#    Calling it directly does NOT run it -> it just returns a coroutine object.
# ---------------------------------------------------------------------------
async def say_hello(name: str) -> str:
    await asyncio.sleep(1)   # non-blocking "wait" -> lets other tasks run meanwhile
    return f"Hello, {name}!"


async def basic_example():
    result = await say_hello("Siva")   # `await` runs the coroutine and gets its result
    print(result)


# ---------------------------------------------------------------------------
# 3. Sequential await vs concurrent gather().
#    Two 1-second sleeps: sequential -> ~2s total, concurrent -> ~1s total.
# ---------------------------------------------------------------------------
async def fetch_item(item_id: int) -> str:
    await asyncio.sleep(1)
    return f"item-{item_id}"


async def sequential_example():
    start = time.perf_counter()
    r1 = await fetch_item(1)
    r2 = await fetch_item(2)
    elapsed = time.perf_counter() - start
    print(f"sequential: {r1}, {r2} (took {elapsed:.2f}s)")


async def concurrent_example():
    start = time.perf_counter()
    r1, r2 = await asyncio.gather(fetch_item(1), fetch_item(2))
    elapsed = time.perf_counter() - start
    print(f"concurrent: {r1}, {r2} (took {elapsed:.2f}s)")


# ---------------------------------------------------------------------------
# 4. Tasks: asyncio.create_task() schedules a coroutine to start running
#    in the background immediately, instead of waiting for `await` to reach it.
# ---------------------------------------------------------------------------
async def background_counter():
    for i in range(3):
        print(f"  ...background tick {i}")
        await asyncio.sleep(0.3)


async def task_example():
    task = asyncio.create_task(background_counter())   # starts running now
    print("main: doing other work while background task runs")
    await asyncio.sleep(0.5)
    await task                                          # wait for it to finish


# ---------------------------------------------------------------------------
# 5. Timeouts: don't let a slow coroutine hang forever.
# ---------------------------------------------------------------------------
async def slow_operation():
    await asyncio.sleep(5)
    return "done"


async def timeout_example():
    try:
        await asyncio.wait_for(slow_operation(), timeout=1)
    except asyncio.TimeoutError:
        print("slow_operation timed out after 1s")


# ---------------------------------------------------------------------------
# 6. Async context managers (`async with`) e.g. asyncio.Lock for shared state.
# ---------------------------------------------------------------------------
async def worker(name: str, lock: asyncio.Lock, shared: list):
    async with lock:
        shared.append(name)
        await asyncio.sleep(0.1)   # simulate work while holding the lock


async def lock_example():
    lock = asyncio.Lock()
    shared: list[str] = []
    await asyncio.gather(*(worker(f"worker-{i}", lock, shared) for i in range(3)))
    print("shared list built safely:", shared)


# ---------------------------------------------------------------------------
# 7. Async generators (`async for`) for streaming results one at a time,
#    similar to how streaming LLM tokens are consumed.
# ---------------------------------------------------------------------------
async def stream_tokens(text: str):
    for word in text.split():
        await asyncio.sleep(0.05)
        yield word


async def async_generator_example():
    tokens = []
    async for token in stream_tokens("Pydantic and asyncio work great together"):
        tokens.append(token)
    print("streamed:", tokens)


# ---------------------------------------------------------------------------
# 8. Practical tie-in: concurrent OpenAI API calls for this project's agent.py.
#    agent.py uses the sync `openai` client (one request at a time). Using
#    AsyncOpenAI lets you fire multiple prompts concurrently instead of
#    waiting for each response before starting the next.
# ---------------------------------------------------------------------------
async def openai_concurrent_example():
    if not os.getenv("OPENAI_API_KEY"):
        print("skipped: set OPENAI_API_KEY to run this example")
        return

    from openai import AsyncOpenAI

    client = AsyncOpenAI()

    async def ask(prompt: str) -> str:
        resp = await client.chat.completions.create(
            model=os.getenv("MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
        )
        return resp.choices[0].message.content.strip()

    prompts = ["Say hi in French", "Say hi in Spanish", "Say hi in Tamil"]
    results = await asyncio.gather(*(ask(p) for p in prompts))
    for prompt, result in zip(prompts, results):
        print(f"{prompt} -> {result}")


async def main():
    print("\n--- 2. basic_example ---")
    await basic_example()

    print("\n--- 3. sequential_example ---")
    await sequential_example()

    print("\n--- 3. concurrent_example ---")
    await concurrent_example()

    print("\n--- 4. task_example ---")
    await task_example()

    print("\n--- 5. timeout_example ---")
    await timeout_example()

    print("\n--- 6. lock_example ---")
    await lock_example()

    print("\n--- 7. async_generator_example ---")
    await async_generator_example()

    print("\n--- 8. openai_concurrent_example ---")
    await openai_concurrent_example()


if __name__ == "__main__":
    print("\n--- 1. asyncio_run_example ---")
    asyncio_run_example()   # standalone: demonstrates asyncio.run() itself

    asyncio.run(main())     # the single entry point that starts the event loop for the rest
