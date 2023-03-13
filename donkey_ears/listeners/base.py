import itertools

from donkey_ears.audio.base import AudioSample, BaseAudioSource


class BaseListener:
    def __init__(self, source: BaseAudioSource):
        self.source = source

    def read(self, n_frames: int = 2**14) -> AudioSample:
        """
        Read and return an audio sample from the source.

        ``n_frames`` is the number of frames to read from the source.
        """
        return self.source.read(n_frames)

    def __iter__(self):
        for _ in itertools.count():
            try:
                yield self.read()
            except EOFError:
                return None
