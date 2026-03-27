import asyncio
import contextlib
from typing import Callable

from aiortc import MediaStreamError, MediaStreamTrack
from av import AudioFrame

from src.classes import Audio
from src.logger import get_logger

logger = get_logger(__name__)


class InputAudioHandler:
    def __init__(
        self,
        track: MediaStreamTrack,
        on_audio: Callable[[Audio], None],
        on_error: Callable[[Exception, bool], None],
    ) -> None:
        self._track = track
        self._task: asyncio.Task | None = None
        self._on_audio = on_audio
        self._on_error = on_error

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        try:
            if self._task is None:
                return

            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

            self._task = None
        except Exception as e:
            logger.error(f"error stopping audio consumer: {e}", exc_info=e)

    async def _run(self) -> None:
        try:
            while True:
                frame = await self._track.recv()
                # each frame will be 960 samples at 48000Hz, stereo, s16 (int16 PCM)
                assert isinstance(frame, AudioFrame)
                self._on_audio(Audio.from_av_frame(frame))
        except MediaStreamError:
            logger.info("caught MediaStreamError, which means that the track has ended")
        except Exception as e:
            self._on_error(e, True)
