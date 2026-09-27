"""Chunks are shared out per client (access key, else address; every local job its own client)."""
import pytest

import server.gallery_jobs as gj


class _Job:
    def __init__(self, token):
        self.token = token
        self.queue = 0


class _Sched:
    """Just what the scheduler's picking reads from a _SchedJob."""
    def __init__(self, token, ip='127.0.0.1', key='local', started=True):
        self.job = _Job(token)
        self.owner_ip, self.owner_key = ip, key
        self.next_page = 16 if started else 0
        self.inflight, self.emitted = set(), {0} if started else set()
        self.has_work = True


@pytest.fixture
def jobs(monkeypatch):
    sched, order = {}, []
    monkeypatch.setattr(gj, '_sched', sched)
    monkeypatch.setattr(gj, '_sched_order', order)
    monkeypatch.setattr(gj, '_last_served', {})

    def add(token, **kw):
        sched[token] = _Sched(token, **kw)
        order.append(token)
        return sched[token]
    return add


def _picks(n):
    return [gj._pick_next().job.token for _ in range(n)]


def test_each_local_job_is_its_own_client(jobs):
    jobs('a'); jobs('b')
    assert gj._served_jobs() == (['a', 'b'], [])
    assert _picks(4) == ['a', 'b', 'a', 'b']


def test_one_remote_address_gets_its_allowance_and_the_rest_waits(jobs):
    jobs('r1', ip='203.0.113.5', key=''); jobs('r2', ip='203.0.113.5', key=''); jobs('l1')
    served, waiting = gj._served_jobs()
    assert served == ['r1', 'l1'] and waiting == ['r2']
    gj._pick_next()
    assert gj._sched['r2'].job.queue == 1


def test_privileged_key_runs_several_jobs_and_clients_share_evenly(jobs, monkeypatch):
    monkeypatch.setattr(gj, 'PRIVILEGED_KEYS', frozenset({'dev'}))
    for t in ('d1', 'd2', 'd3'):
        jobs(t, ip='198.51.100.7', key='dev')
    jobs('x', ip='203.0.113.9', key='')
    assert gj._served_jobs()[0] == ['d1', 'd2', 'd3', 'x']
    # client-level turns alternate dev / x; within dev its jobs take turns
    assert _picks(6) == ['d1', 'x', 'd2', 'x', 'd3', 'x']


def test_client_limit(jobs, monkeypatch):
    monkeypatch.setattr(gj, 'ACTIVE_CLIENTS', 1)
    jobs('a'); jobs('b')
    assert gj._served_jobs() == (['a'], ['b'])
    assert _picks(2) == ['a', 'a']


def test_a_job_that_has_not_started_goes_first(jobs):
    jobs('a'); jobs('b')
    _picks(3)
    jobs('new', started=False)
    assert gj._pick_next().job.token == 'new'
