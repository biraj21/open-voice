import base64
from fractions import Fraction
from functools import cached_property
from typing import Literal, overload

import numpy as np
import soundfile as sf
import torch
from av import AudioFrame
from av.audio.resampler import AudioResampler


class Audio:
    """
    Internal repr: (samples, channels), s16 or flt depending on source.

    Primary pipeline: aiortc frame (stereo 48kHz) -> mono 16kHz -> VAD -> ...
    """

    def __init__(self, buffer: np.ndarray, sample_rate: int, fmt: str, layout: str) -> None:
        assert buffer.ndim == 2  # (samples, channels)

        # why buffer is of shape (samples, channels) and not (channels, samples)?
        # because it makes it easier to get all channel values in a given time
        # frame[t] -> [L, R]

        self._buffer = buffer
        self._sample_rate = sample_rate
        self._fmt = fmt  # "s16" | "fltp" | "flt"
        self._layout = layout

    @property
    def channels(self) -> int:
        return self._buffer.shape[1]

    @property
    def samples(self) -> int:
        return self._buffer.shape[0]

    @property
    def duration_ms(self) -> float:
        return (self.samples / self._sample_rate) * 1000

    @classmethod
    def from_av_frame(cls, frame: AudioFrame) -> Audio:
        arr = frame.to_ndarray()
        channels = frame.layout.nb_channels

        # Normalize to (samples, channels)
        if frame.format.is_planar:
            # (channels, samples) -> (samples, channels)
            arr = arr.T
        else:
            # packed: let's say the shape is (1, samples * channels), with format [[L0, R0, L1, R1, ...]] for stereo
            # arr[0] is a flat interleaved stream: [L0, R0, L1, R1, ...]
            # reshape(-1, channels) groups consecutive values per timestep:
            #   [
            #     [L0, R0],
            #     [L1, R1],
            #     [L2, R2],
            #     ...
            #   ]  # (samples, channels)
            arr = arr.reshape(-1, channels)

        return cls(
            buffer=arr,
            sample_rate=frame.sample_rate,
            fmt=frame.format.name,
            layout=frame.layout.name,
        )

    def to_av_frame(self, pts: int = 0) -> AudioFrame:
        if self._fmt.endswith("p"):
            # planar → (channels, samples)
            arr = self._buffer.T
        else:
            # packed → (1, samples * channels), interleaved
            arr = self._buffer.reshape(1, -1)

        frame = AudioFrame.from_ndarray(arr, format=self._fmt, layout=self._layout)
        frame.sample_rate = self._sample_rate
        frame.time_base = Fraction(1, self._sample_rate)
        frame.pts = pts
        return frame

    @classmethod
    def from_list(cls, arr: list[Audio]):
        assert all(a._sample_rate == arr[0]._sample_rate for a in arr)

        buf = np.concatenate([a._as_float32() for a in arr], axis=0)
        return cls(
            buffer=buf,
            sample_rate=arr[0]._sample_rate,
            fmt="flt",
            layout=arr[0]._layout,
        )

    @cached_property
    def mono(self) -> Audio:
        if self.channels == 1:
            return self
        # mix down via float32 to avoid int16 overflow on sum
        f32 = self._as_float32()
        mixed = f32.mean(axis=1, keepdims=True)
        buf = self._float32_to_int16(mixed) if self._fmt == "s16" else mixed
        fmt = "s16" if self._fmt == "s16" else "flt"
        return Audio(buffer=buf, sample_rate=self._sample_rate, fmt=fmt, layout="mono")

    @overload
    def resampled(
        self,
        target_sr: int,
        fmt: str | None = None,
        layout: str | None = None,
        *,
        chunk_size: None = None,
        output: Literal["audio"] = "audio",
    ) -> Audio: ...

    @overload
    def resampled(
        self,
        target_sr: int,
        fmt: str | None = None,
        layout: str | None = None,
        *,
        chunk_size: None = None,
        output: Literal["frame"],
    ) -> list[AudioFrame]: ...

    @overload
    def resampled(
        self,
        target_sr: int,
        fmt: str | None = None,
        layout: str | None = None,
        *,
        chunk_size: int,
        output: Literal["audio"] = "audio",
    ) -> list[Audio]: ...

    @overload
    def resampled(
        self,
        target_sr: int,
        fmt: str | None = None,
        layout: str | None = None,
        *,
        chunk_size: int,
        output: Literal["frame"],
    ) -> list[AudioFrame]: ...

    def resampled(
        self,
        target_sr: int,
        fmt: str | None = None,
        layout: str | None = None,
        *,
        chunk_size: int | None = None,
        output: Literal["audio", "frame"] = "audio",
    ) -> Audio | list[Audio] | list[AudioFrame]:
        target_fmt = fmt or self._fmt
        target_layout = layout or self._layout

        if (
            self._sample_rate == target_sr
            and target_fmt == self._fmt
            and target_layout == self._layout
            and chunk_size is None
            and output == "audio"
        ):
            return self

        resampler = AudioResampler(
            format=target_fmt,
            layout=target_layout,
            rate=target_sr,
            frame_size=chunk_size,
        )
        frames = resampler.resample(self.to_av_frame())
        frames += resampler.resample(None)  # flush

        if not frames:
            raise RuntimeError("Resampler returned no frames")

        # --- no chunking ---
        if chunk_size is None:
            if output == "frame":
                return frames

            return Audio.from_list([Audio.from_av_frame(f) for f in frames])

        # --- chunking already handled by resampler ---
        if output == "frame":
            return frames

        return [Audio.from_av_frame(f) for f in frames]

    @cached_property
    def as_int16(self) -> np.ndarray:
        """Returns (samples, channels) int16 array — for VAD or STT."""
        if self._fmt == "s16":
            return self._buffer
        return self._float32_to_int16(self._as_float32())

    def to_bytes(self) -> bytes:
        return self.as_int16.tobytes()

    def to_base64(self) -> str:
        return base64.b64encode(self.to_bytes()).decode()

    @cached_property
    def as_float32(self) -> np.ndarray:
        return self._as_float32()

    @cached_property
    def as_torch(self) -> torch.Tensor:
        return torch.from_numpy(self.as_float32)

    def _as_float32(self) -> np.ndarray:
        if self._fmt in ("flt", "fltp"):
            return self._buffer.astype(np.float32, copy=False)
        if self._fmt == "s16":
            return self._buffer.astype(np.float32) / 32768.0
        raise ValueError(f"Unsupported fmt: {self._fmt}")

    @staticmethod
    def _float32_to_int16(arr: np.ndarray) -> np.ndarray:
        return (arr * 32767.0).clip(-32768, 32767).astype(np.int16)

    def save_wav(self, name: str):
        if not name.endswith(".wav"):
            name = name + ".wav"

        sf.write(name, self.as_float32, samplerate=self._sample_rate)
