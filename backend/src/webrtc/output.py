import asyncio
import time
from fractions import Fraction

from aiortc import MediaStreamError, MediaStreamTrack
from av import AudioFrame

from src.audio import Audio
from src.logger import get_logger

logger = get_logger(__name__)


class OutputAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int = 48_000) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._samples_per_frame = 960
        self._queue: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._start: float | None = None
        self._timestamp = 0
        self._last_frame_samples = 0

    async def recv(self) -> AudioFrame:
        """
        Receive the next audio frame.
        """

        if self.readyState != "live":
            raise MediaStreamError

        frame = await self._queue.get()
        if frame is None:
            self.stop()
            raise MediaStreamError

        if self._start is None:
            self._start = time.time()
            self._timestamp = 0
        else:
            self._timestamp += self._last_frame_samples
            wait = self._start + (self._timestamp / self._sample_rate) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

        frame.pts = self._timestamp
        self._last_frame_samples = frame.samples

        return frame

    async def enqueue_audio(self, audio: Audio | None) -> None:
        """
        Enqueue audio chunk for sending.
        """

        if self.readyState != "live":
            return

        try:
            if audio is None:
                await self._queue.put(None)
                return

            # audio needs to be 48khz, and sliced into chunks of 20ms
            # each frame will be sent as a separate packet
            frames = audio.resampled(
                self._sample_rate,
                fmt="s16",
                layout="mono",
                chunk_size=self._samples_per_frame,
                output="frame",
            )
            for frame in frames:
                frame.time_base = Fraction(1, self._sample_rate)
                await self._queue.put(frame)
        except asyncio.QueueFull:
            logger.warning("Audio output queue full, dropping frame")

    def clear_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def close(self) -> None:
        """Close the track."""
        self.stop()
        await self._queue.put(None)


class OutputAudioHandler:
    def __init__(self, track: OutputAudioTrack | None = None) -> None:
        self._track = track or OutputAudioTrack()
        self._closed = False

    @property
    def track(self) -> OutputAudioTrack:
        return self._track

    async def enqueue_audio(self, audio: Audio) -> None:
        """
        Enqueue audio chunk for sending.
        """

        if self._closed:
            logger.warning("Output handler closed, ignoring audio")
            return

        await self._track.enqueue_audio(audio)

    def clear_queue(self) -> None:
        self.track.clear_queue()

    async def close(self) -> None:
        """Close the handler."""
        self._closed = True
        await self._track.close()
