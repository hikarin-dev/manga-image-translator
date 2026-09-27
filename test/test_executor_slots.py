"""An executor with two slots takes a second gallery chunk only once the first has read its pages."""
import asyncio

from server.instance import ExecutorInstance, Executors


def test_second_chunk_waits_for_the_first_to_finish_reading():
    async def scenario():
        pool = Executors()
        inst = ExecutorInstance(ip='127.0.0.1', port=65001, slots=2)
        pool.register(inst)

        first = await pool.find_executor(gallery=True)
        assert first is inst and inst.active == 1 and inst.busy
        assert pool.free_executors(gallery=True) == 0, 'still reading: no second chunk yet'

        pool.chunk_read(inst)
        assert pool.free_executors(gallery=True) == 1 and not inst.busy
        assert pool.free_executors(gallery=False) == 0, 'single-image work still waits for an empty worker'

        second = await pool.find_executor(gallery=True)
        assert second is inst and inst.active == 2 and inst.reading == 1
        assert pool.free_executors(gallery=True) == 0, 'both slots taken'

        await pool.free_executor(inst, read_done=True)       # the first chunk finishes its tail
        assert inst.active == 1 and inst.reading == 1 and inst.busy
        await pool.free_executor(inst, read_done=False)      # the second ends before reporting
        assert inst.active == 0 and inst.reading == 0 and not inst.busy
        assert pool.free_executors(gallery=False) == 1

    asyncio.run(scenario())


def test_single_slot_is_one_chunk_at_a_time():
    async def scenario():
        pool = Executors()
        inst = ExecutorInstance(ip='127.0.0.1', port=65002)
        pool.register(inst)
        await pool.find_executor(gallery=True)
        pool.chunk_read(inst)
        assert pool.free_executors(gallery=True) == 0, 'no overlap without a second slot'
        await pool.free_executor(inst, read_done=True)
        assert pool.free_executors(gallery=True) == 1

    asyncio.run(scenario())
