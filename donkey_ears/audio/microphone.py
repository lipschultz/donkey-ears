import io
from typing import List, Mapping, Optional

import pyaudio
from pydub import AudioSegment

from donkey_ears.audio.base import AudioSample, BaseAudioSource


class Microphone(BaseAudioSource):
    def __init__(self, device_index: Optional[int] = None):
        """
        The ``device_index`` is used to tell PyAudio which audio device to listen on.
        """
        pa_instance = pyaudio.PyAudio()
        device_count = pa_instance.get_device_count()
        pa_instance.terminate()

        if not (device_index is None or (isinstance(device_index, int) and 0 <= device_index < device_count)):
            raise ValueError(
                f"device_index must be None or positive integer between 0 and {device_count}, got: {device_index!r}"
            )

        self._device_index = device_index
        self.DEFAULT_READ_DURATION_SECONDS = 5  # pylint: disable=invalid-name

    @classmethod
    def get_device_names(cls) -> List[str]:
        pa_instance = pyaudio.PyAudio()
        try:
            return [
                str(pa_instance.get_device_info_by_index(i).get("name")) for i in range(pa_instance.get_device_count())
            ]
        finally:
            pa_instance.terminate()

    @property
    def device_index(self) -> Optional[int]:
        return self._device_index

    @property
    def audio_device_information(self) -> Mapping:
        pa_instance = pyaudio.PyAudio()
        try:
            if self.device_index is None:
                return pa_instance.get_default_input_device_info()

            return pa_instance.get_device_info_by_index(self.device_index)
        finally:
            pa_instance.terminate()

    @property
    def frame_rate(self) -> int:
        return int(self.audio_device_information["defaultSampleRate"])

    @property
    def n_channels(self) -> int:
        return 1

    @property
    def _pyaudio_format(self) -> int:
        return pyaudio.paInt16

    @property
    def bit_depth(self) -> int:
        return 16

    @property
    def frame_width(self) -> int:
        return pyaudio.get_sample_size(self._pyaudio_format)

    @property
    def DEFAULT_READ_DURATION_FRAMES(self) -> int:  # pylint: disable=invalid-name
        return self.seconds_to_frame(self.DEFAULT_READ_DURATION_SECONDS)

    def read_bytes(self, n_frames: int) -> bytes:
        if not isinstance(n_frames, int):
            raise TypeError(
                f"`n_frames` must be an integer greater than zero, received {n_frames!r} (type={type(n_frames)})"
            )
        if n_frames == 0:
            raise ValueError(f"`n_frames` must be an integer greater than zero, received {n_frames!r}")

        pa_instance = pyaudio.PyAudio()
        try:
            audio_source = pa_instance.open(
                input_device_index=self.device_index,
                channels=self.n_channels,
                format=self._pyaudio_format,
                rate=self.frame_rate,
                input=True,
            )
            return audio_source.read(n_frames, exception_on_overflow=False)
        finally:
            pa_instance.terminate()

    def read_pydub(self, n_frames: int) -> AudioSegment:
        """
        Read n_frames from the microphone and return a pydub.AudioSegment object.
        """
        with io.BytesIO(self.read_bytes(n_frames)) as fp:
            return AudioSegment.from_raw(
                fp,
                sample_width=self.frame_width,
                frame_rate=self.frame_rate,
                channels=self.n_channels,
            )

    def read(self, n_frames: Optional[int]) -> AudioSample:
        if n_frames is None:
            n_frames = self.DEFAULT_READ_DURATION_FRAMES
        return AudioSample(self.read_pydub(n_frames))
