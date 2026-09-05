#!/usr/bin/env python3
"""
Tricky asyncio.gather() questions, answered with runnable proof.
Run: python3 src/reference/asyncio-gather-tricky.py
"""

import asyncio


# ---------------------------------------------------------------------------
# Q1: If one gathered coroutine raises, are the OTHER coroutines cancelled?
#     Answer: NO (by default). They keep running in the background even
#     though gather() has already raised and your except block moved on.
# ---------------------------------------------------------------------------
async def q1_sibling_not_cancelled():
    print("\nQ1: does a failing task cancel its siblings?")

    async def bad():
        await asyncio.sleep(0.2)
        raise ValueError("bad task failed")

    async def slow():
        await asyncio.sleep(0.6)
        print("  slow: I finished AFTER gather() already raised!")
        return "slow-done"

    try:
        await asyncio.gather(bad(), slow())
    except ValueError as e:
        print(f"  gather raised immediately: {e}")

    await asyncio.sleep(0.6)   # give the orphaned 'slow' task time to finish
    print("  -> answer: siblings are NOT auto-cancelled; they run to completion")


# ---------------------------------------------------------------------------
# Q2: What's the practical difference between return_exceptions=True/False?
#     Default (False): first exception propagates, you lose the other results.
#     True: exceptions come back as regular items in the results list.
# ---------------------------------------------------------------------------
async def q2_return_exceptions():
    print("\nQ2: return_exceptions=True vs False")

    async def bad():
        raise ValueError("bad")

    async def good():
        return "good"

    results = await asyncio.gather(good(), bad(), return_exceptions=True)
    print(f"  results (mixed success/failure): {results}")
    print("  -> answer: with True, you must check each item's type yourself")


# ---------------------------------------------------------------------------
# Q3: Are results returned in completion order or input order?
#     Answer: INPUT order, regardless of which one finishes first.
# ---------------------------------------------------------------------------
async def q3_result_ordering():
    print("\nQ3: result ordering vs completion ordering")

    async def task(label, delay):
        await asyncio.sleep(delay)
        print(f"  {label}: completed (delay={delay})")
        return label

    # "B" finishes first (shorter delay), but is passed second.
    results = await asyncio.gather(task("A", 0.3), task("B", 0.1))
    print(f"  -> answer: results preserve input order: {results}")


# ---------------------------------------------------------------------------
# Q4: If the gather() call itself is cancelled (e.g. via wait_for timeout),
#     what happens to the still-running children? Answer: they ARE cancelled.
#     (Contrast with Q1, where a sibling raising an exception did NOT cancel
#     the others -- cancelling the outer awaitable is what cancels children.)
# ---------------------------------------------------------------------------
async def q4_outer_cancel_cancels_children():
    print("\nQ4: cancelling gather() from outside (e.g. wait_for timeout)")

    async def long_task(label):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            print(f"  {label}: cancelled!")
            raise

    try:
        await asyncio.wait_for(
            asyncio.gather(long_task("A"), long_task("B")), timeout=0.3
        )
    except asyncio.TimeoutError:
        print("  -> answer: wait_for timeout cancels gather(), which cancels all children")


# ---------------------------------------------------------------------------
# Q5: Common bug -- passing a list instead of unpacking it with *.
# ---------------------------------------------------------------------------
async def q5_forgetting_unpack():
    print("\nQ5: gather(coros) vs gather(*coros)")

    async def noop():
        return 1

    coros = [noop(), noop()]

    try:
        asyncio.gather(coros)   # WRONG: a list is not itself an awaitable
    except TypeError as e:
        print(f"  gather(coros) raises immediately: {e}")

    results = await asyncio.gather(*coros)   # correct: unpack with *
    print(f"  -> answer: must unpack with *: gather(*coros) = {results}")


# ---------------------------------------------------------------------------
# Q6: Does gather() have a built-in timeout? Answer: NO. A hung coroutine
#     blocks forever unless you wrap the whole gather() in wait_for().
# ---------------------------------------------------------------------------
async def q6_no_builtin_timeout():
    print("\nQ6: does gather() support a timeout parameter?")

    async def hangs():
        await asyncio.sleep(5)
        return "done"

    try:
        await asyncio.wait_for(asyncio.gather(hangs(), hangs()), timeout=0.3)
    except asyncio.TimeoutError:
        print("  -> answer: gather() has no `timeout=` kwarg; wrap it in asyncio.wait_for()")


async def main():
    await q1_sibling_not_cancelled()
    await q2_return_exceptions()
    await q3_result_ordering()
    await q4_outer_cancel_cancels_children()
    await q5_forgetting_unpack()
    await q6_no_builtin_timeout()


if __name__ == "__main__":
    asyncio.run(main())
