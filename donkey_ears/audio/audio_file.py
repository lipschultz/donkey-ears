from pathlib import Path
from typing import Optional, Union

from pydub import AudioSegment

from donkey_ears.audio.base import AudioSample, BaseAudioSource


class AudioFile(BaseAudioSource):
    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self._audio_data = AudioSegment.from_file(filepath)
        self.frame_index = 0

    def __max_audio_frames(self) -> int:
        return int(self._audio_data.frame_count())

    def jump_to_frame(self, frame_number: int):
        if not isinstance(frame_number, int):
            raise TypeError(f"`frame_number` must be an integer, received {frame_number!r} (type={type(frame_number)})")
        if frame_number < 0 or frame_number >= self.__max_audio_frames():
            raise ValueError(
                f"`frame_number` must be an integer between 0 and {self.__max_audio_frames()} (the number of frames in the file), received {frame_number!r}"
            )

        self.frame_index = frame_number

    def reset(self):
        self.jump_to_frame(0)

    @property
    def frame_rate(self) -> int:
        return self._audio_data.frame_rate

    def read_pydub(self, n_frames: Optional[int]) -> AudioSegment:
        if self.frame_index >= self.__max_audio_frames():
            raise EOFError("Attempted to read past the end of the audio file.")

        if n_frames is None:
            n_frames = self.__max_audio_frames() - self.frame_index
        if not isinstance(n_frames, int):
            raise TypeError(
                f"`n_frames` must be an integer greater than zero, received {n_frames!r} (type={type(n_frames)})"
            )
        if n_frames == 0:
            raise ValueError(f"`n_frames` must be an integer greater than zero, received {n_frames!r}")

        read_data = self._audio_data.get_sample_slice(self.frame_index, self.frame_index + n_frames)
        self.frame_index += n_frames
        return read_data

    def read(self, n_frames: Optional[int]) -> AudioSample:
        """
        Read frames of audio from the file and return them in an ``AudioSample``.

        If ``n_frames` is ``None``, read all remaining frames in the file.

        If reading after the end of the file, EOFError will be raised.
        """
        return AudioSample(self.read_pydub(n_frames))
