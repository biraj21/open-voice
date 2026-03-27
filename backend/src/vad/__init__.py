import math
from pathlib import Path
from typing import Callable, Literal, NamedTuple

import torch

from src.classes import Audio
from src.logger import get_logger

from .utils import OnnxWrapper, VADIterator

SAMPLING_RATE = 16_000
MIN_SAMPLES = 512  # Silero requirement for 16kHz (32ms)
MIN_PREFIX_DURATION_MS = 200

# Define a floor for "detectable" speech
# -30dB to -40dB is a typical starting point for close-mic speech
RMS_THRESHOLD_DB = -40

logger = get_logger(__name__)


class VadSpeechStarted(NamedTuple):
    timestamp_sec: int | float
    type: Literal["speech_started"] = "speech_started"


class VadSpeechEnded(NamedTuple):
    timestamp_sec: int | float
    speech: Audio
    type: Literal["speech_ended"] = "speech_ended"


VadEvent = VadSpeechStarted | VadSpeechEnded


class VAD:
    def __init__(
        self,
        on_event: Callable[[VadEvent], None],
        threshold: float = 0.5,
    ):
        self._model_path = Path(__file__).resolve().parent / "silero_vad.onnx"
        self._model = OnnxWrapper(path=str(self._model_path))
        self._vad_iterator = VADIterator(
            model=self._model,
            threshold=threshold,
            sampling_rate=SAMPLING_RATE,
        )

        # audio objects to create speech segment after end is detected
        self._buffer: list[Audio] = []

        # raw samples for VAD processing
        self._sample_buffer = torch.zeros(0)

        self._state: Literal["idle", "speaking"] = "idle"
        self._on_event = on_event

    def process_frame(self, audio_frame: Audio):
        """
        Processes incoming audio frames.
        Buffers 20ms chunks until 32ms (512 samples) are ready for Silero.
        """

        # 1. silero vad requires 16khz mono audio
        audio_frame = audio_frame.resampled(
            target_sr=SAMPLING_RATE,
            layout="mono",
            fmt="flt",
        )

        # 2. add to both buffers
        self._buffer.append(audio_frame)
        new_samples = audio_frame.as_torch.reshape(-1)
        self._sample_buffer = torch.cat([self._sample_buffer, new_samples])

        # 3. process all available 512-sample windows
        while self._sample_buffer.shape[0] >= MIN_SAMPLES:
            # extract exactly MIN_SAMPLES (512 samples) from the buffer
            current_chunk = self._sample_buffer[:MIN_SAMPLES].reshape(1, -1)

            # remove the current_chunk samples from the buffer
            self._sample_buffer = self._sample_buffer[MIN_SAMPLES:]

            # --- if idle, we check for loudness using RMS ---
            if self._state == "idle" and self.get_loudness_rms_db(current_chunk) < RMS_THRESHOLD_DB:
                # too quiet to be speech; treat as silence manually
                # also saves CPU by skipping the ONNX forward pass lol
                result = None
            else:
                result = self._vad_iterator(current_chunk, return_seconds=True)

            if result is None:
                self._handle_prefix()
                continue

            if result.start is not None:
                self._state = "speaking"
                self._on_event(VadSpeechStarted(timestamp_sec=result.start))
            elif result.end is not None:
                self._state = "idle"

                # emit the collected audio (the speech)
                if self._buffer:
                    speech = Audio.from_list(self._buffer)
                    self._on_event(VadSpeechEnded(timestamp_sec=result.end, speech=speech))

                # reset for next segment
                self._buffer.clear()

                # clear leftover samples to prevent carry-over noise
                self._sample_buffer = torch.zeros(0)
                break

    def _handle_prefix(self):
        """
        Removes old frames during 'idle' state to maintain the prefix window.
        """

        if self._state == "idle":
            current_duration = sum(a.duration_ms for a in self._buffer)
            # If we exceed the prefix duration, drop the oldest frame
            while current_duration > MIN_PREFIX_DURATION_MS and len(self._buffer) > 1:
                dropped = self._buffer.pop(0)
                current_duration -= dropped.duration_ms

    @staticmethod
    def get_loudness_rms_db(audio_tensor: torch.Tensor) -> float:
        """
        Calculates the RMS energy of a tensor in decibels.

        Note: This function is completely AI generated. I don't really understand this calculation.
        """

        # ensure we don't log(0)
        rms = torch.sqrt(torch.mean(audio_tensor**2))
        if rms.item() <= 0:
            return -100.0

        return 20 * math.log10(rms.item())
