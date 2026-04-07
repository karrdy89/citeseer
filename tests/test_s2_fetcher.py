"""S2Fetcher 단위 테스트 — httpx mock transport 사용."""
import asyncio

import httpx
import pytest

from app.core.config import Settings
from app.s2.client import S2Fetcher
from conftest import make_s2_response, MockS2Transport


@pytest.mark.asyncio
async def test_basic_pagination():
    """2페이지 페이지네이션 → 큐에 2개 push."""
    transport = MockS2Transport([
        make_s2_response(["p1", "p2"], token="next"),
        make_s2_response(["p3"], token=None),
    ])
    settings = Settings(s2_rate_limit=0)
    queue: asyncio.Queue = asyncio.Queue(maxsize=5)

    async with httpx.AsyncClient(transport=transport) as client:
        fetcher = S2Fetcher(settings, client, queue)
        await fetcher.run()

    # 2 페이지 + sentinel
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())

    assert len(items) == 3  # page0, page1, None sentinel
    assert items[2] is None
    assert len(items[0]) == 2  # p1, p2
    assert len(items[1]) == 1  # p3
    assert fetcher.request_count == 2


@pytest.mark.asyncio
async def test_retry_on_failure():
    """첫 요청 실패 → retry → 성공."""
    transport = MockS2Transport([
        httpx.ConnectError("connection refused"),
        make_s2_response(["p1"], token=None),
    ])
    settings = Settings(s2_rate_limit=0, max_retries=3)
    queue: asyncio.Queue = asyncio.Queue(maxsize=5)

    async with httpx.AsyncClient(transport=transport) as client:
        fetcher = S2Fetcher(settings, client, queue)
        await fetcher.run()

    items = []
    while not queue.empty():
        items.append(queue.get_nowait())

    # 재시도 성공 → 1 페이지 + sentinel
    assert len(items) == 2
    assert items[0] is not None
    assert items[1] is None
    assert fetcher.request_count == 1  # 실패한 건은 카운트 안 함


@pytest.mark.asyncio
async def test_max_retries_exceeded():
    """재시도 상한 초과 → 해당 페이지 스킵."""
    transport = MockS2Transport([
        httpx.ConnectError("fail 1"),
        httpx.ConnectError("fail 2"),
        httpx.ConnectError("fail 3"),
        httpx.ConnectError("fail 4"),  # max_retries=3이면 4번째에 포기
    ])
    settings = Settings(s2_rate_limit=0, max_retries=3)
    queue: asyncio.Queue = asyncio.Queue(maxsize=5)

    async with httpx.AsyncClient(transport=transport) as client:
        fetcher = S2Fetcher(settings, client, queue)
        await fetcher.run()

    items = []
    while not queue.empty():
        items.append(queue.get_nowait())

    # 데이터 없이 sentinel만
    assert len(items) == 1
    assert items[0] is None


@pytest.mark.asyncio
async def test_request_limit():
    """s2_max_requests 도달 시 중단."""
    transport = MockS2Transport([
        make_s2_response(["p1"], token="next"),
        make_s2_response(["p2"], token="next"),
        make_s2_response(["p3"], token=None),  # 여기까지 안 감
    ])
    settings = Settings(s2_rate_limit=0, s2_max_requests=2)
    queue: asyncio.Queue = asyncio.Queue(maxsize=5)

    async with httpx.AsyncClient(transport=transport) as client:
        fetcher = S2Fetcher(settings, client, queue)
        await fetcher.run()

    assert fetcher.request_count == 2

    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    # 2 페이지 + sentinel
    assert items[-1] is None
    assert len(items) == 3


@pytest.mark.asyncio
async def test_backpressure():
    """prefetch buffer가 가득 차면 fetcher가 대기."""
    transport = MockS2Transport([
        make_s2_response(["p1"], token="t1"),
        make_s2_response(["p2"], token="t2"),
        make_s2_response(["p3"], token=None),
    ])
    settings = Settings(s2_rate_limit=0)
    # buffer 1 → 두 번째 put에서 대기
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    async with httpx.AsyncClient(transport=transport) as client:
        fetcher = S2Fetcher(settings, client, queue)
        task = asyncio.create_task(fetcher.run())

        # 첫 페이지가 큐에 들어올 때까지 대기
        page1 = await asyncio.wait_for(queue.get(), timeout=5.0)
        assert len(page1) == 1

        # 두 번째 페이지
        page2 = await asyncio.wait_for(queue.get(), timeout=5.0)
        assert len(page2) == 1

        # 세 번째 + sentinel
        page3 = await asyncio.wait_for(queue.get(), timeout=5.0)
        assert len(page3) == 1

        sentinel = await asyncio.wait_for(queue.get(), timeout=5.0)
        assert sentinel is None

        await task
