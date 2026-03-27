import asyncio
from typing import Coroutine

from src.logger import get_logger

logger = get_logger(__name__)


class BackgroundTasks:
    """
    A helper class to manage background tasks.


    We cannot just fire-and-forget `asyncio.create_task()` because we need to keep
    strong references to tasks. Hence this.

    Usage:
    ```
    from src.bg_tasks import BackgroundTasks

    async def my_async_fn():
        ...

    BackgroundTasks.add(my_async_fn())
    ```
    """

    _tasks: set[asyncio.Task] = set()

    @classmethod
    def add(cls, task_or_coro: asyncio.Task | Coroutine):
        if asyncio.iscoroutine(task_or_coro):
            task_or_coro = asyncio.create_task(task_or_coro)

        cls._tasks.add(task_or_coro)
        task_or_coro.add_done_callback(cls._on_task_done)

    @classmethod
    def _on_task_done(cls, task: asyncio.Task):
        cls._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("background task failed", exc_info=task.exception())

    @classmethod
    async def cancel_all(cls):
        for task in list(cls._tasks):
            task.cancel()

        await asyncio.gather(*cls._tasks, return_exceptions=True)
