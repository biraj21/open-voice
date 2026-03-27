import asyncio
from collections.abc import Callable

from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceCandidate,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)

from src.audio import Audio
from src.bg_tasks import BackgroundTasks
from src.logger import get_logger
from src.schemas.realtime import SessionConfig
from src.vad import VAD, VadEvent, VadSpeechEnded

from .input import InputAudioHandler
from .output import OutputAudioHandler
from .turn_servers import TurnServer

logger = get_logger(__name__)


class WebRTCConnection:
    def __init__(
        self,
        offer_sdp: str,
        session_config: SessionConfig,
        turn_servers: list[TurnServer] | None = None,
        on_close: Callable[["WebRTCConnection"], None] | None = None,
    ) -> None:
        pc_config = None
        if turn_servers is not None:
            pc_config = RTCConfiguration(
                iceServers=[
                    RTCIceServer(
                        urls=ts.urls,
                        username=ts.username,
                        credential=ts.credential,
                    )
                    for ts in turn_servers
                ],
            )

        self._offer_sdp = offer_sdp
        self._session_config = session_config
        self._on_close = on_close
        self._pc = RTCPeerConnection(pc_config)
        self._input_audio_handlers = set[InputAudioHandler]()
        self._closed = False
        self._remote_description_set = False
        self._pending_ice_candidates: list[RTCIceCandidate] = []

        self._output_handler = OutputAudioHandler()
        self._pc.addTrack(self._output_handler.track)
        self._current_speech_id: str | None = None

        def on_vad_event(event: VadEvent):
            logger.info(f"birajlog vad event: {event}")
            if event.type == "speech_started":
                self._output_handler.clear_queue(cancel_id=self._current_speech_id)
                self._current_speech_id = event.speech_id
            elif isinstance(event, VadSpeechEnded):
                # for now, we're basically sending the user's speech as-is
                BackgroundTasks.add(
                    asyncio.create_task(
                        self._output_handler.enqueue_audio(event.speech, id=event.speech_id)
                    )
                )

        self._vad = VAD(on_event=on_vad_event)

        def on_audio(audio: Audio):
            self._vad.process_frame(audio)

        @self._pc.on("track")
        def on_track(track: MediaStreamTrack):
            if track.kind != "audio":
                return

            consumer = InputAudioHandler(
                track,
                on_audio=on_audio,
                on_error=self._on_error,
            )
            consumer.start()
            self._input_audio_handlers.add(consumer)

            @track.on("ended")
            async def on_ended():
                await consumer.stop()
                if consumer in self._input_audio_handlers:
                    self._input_audio_handlers.discard(consumer)

    def _on_error(self, e: Exception, panic: bool = False):
        if panic:
            logger.error(f"webrtc connection panic (closing connection): {e}", exc_info=e)
            BackgroundTasks.add(self.close())
        else:
            logger.error(f"webrtc connection error: {e}", exc_info=e)

    async def answer(self) -> RTCSessionDescription:
        offer = RTCSessionDescription(sdp=self._offer_sdp, type="offer")
        await self._pc.setRemoteDescription(offer)
        self._remote_description_set = True

        pending_ice_candidates = self._pending_ice_candidates.copy()
        self._pending_ice_candidates = []

        for candidate in pending_ice_candidates:
            await self._pc.addIceCandidate(candidate)

        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)
        return self._pc.localDescription

    async def add_or_queue_ice_candidates(self, candidates: list[RTCIceCandidate]) -> None:
        for c in candidates:
            if self._remote_description_set:
                await self._pc.addIceCandidate(c)
            else:
                self._pending_ice_candidates.append(c)

    async def close(self) -> None:
        if self._closed:
            return

        self._closed = True

        for handler in self._input_audio_handlers:
            await handler.stop()

        self._input_audio_handlers.clear()

        await self._output_handler.close()
        await self._pc.close()
        if self._on_close is not None:
            self._on_close(self)

    async def _wait_for_ice_gathering(self) -> None:
        if self._pc.iceGatheringState == "complete":
            return

        done = asyncio.Event()

        @self._pc.on("icegatheringstatechange")
        def on_icegatheringstatechange():
            if self._pc.iceGatheringState == "complete":
                done.set()

        await done.wait()
