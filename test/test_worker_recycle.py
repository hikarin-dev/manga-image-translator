"""The local worker leaves the rotation once a finished chunk reports it over its memory budget."""
import server.main as main
from server.instance import ExecutorInstance, executor_instances


def _setup(monkeypatch, budget=24):  # an explicit budget, independent of the default
    inst = ExecutorInstance(ip='127.0.0.1', port=65000)
    executor_instances.register(inst)
    monkeypatch.setattr(main, '_local_worker', {'tree': [(4242, 1.0)], 'instance': inst})
    monkeypatch.setattr(main, 'WORKER_RECYCLE_GB', budget)
    return inst


def _telemetry(pid, commit, children=2.0):
    return {'mem': {'pid': pid, 'commit_end': commit, 'children_max': children, 'trimmed': True}}


def test_over_budget_leaves_rotation(monkeypatch):
    inst = _setup(monkeypatch)
    try:
        main._check_worker_memory(_telemetry(4242, 23.0))
        assert main._local_worker.get('recycle') is True
        assert all(x is not inst for x in executor_instances.list)
    finally:
        executor_instances.unregister(inst)


def test_under_budget_or_other_machine_is_left_alone(monkeypatch):
    inst = _setup(monkeypatch)
    try:
        main._check_worker_memory(_telemetry(4242, 20.0))       # 22 GB with its pool: within budget
        main._check_worker_memory(_telemetry(9999, 40.0))       # an aux node's chunk
        assert not main._local_worker.get('recycle')
        assert any(x is inst for x in executor_instances.list)
    finally:
        executor_instances.unregister(inst)


def test_disabled_with_zero_budget(monkeypatch):
    inst = _setup(monkeypatch, budget=0)
    try:
        main._check_worker_memory(_telemetry(4242, 60.0))
        assert not main._local_worker.get('recycle')
    finally:
        executor_instances.unregister(inst)


def test_untrimmed_report_is_not_judged(monkeypatch):
    inst = _setup(monkeypatch)
    try:
        telemetry = _telemetry(4242, 40.0)
        telemetry['mem']['trimmed'] = False     # caches not yet released: not the worker's real floor
        main._check_worker_memory(telemetry)
        assert not main._local_worker.get('recycle')
    finally:
        executor_instances.unregister(inst)
