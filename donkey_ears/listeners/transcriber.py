import itertools
from typing import Union

from donkey_ears.listeners.audio import ContinuousListener, Listener, NoAudioAvailable
from donkey_ears.speech_to_text.base import BaseSpeechToText, DetailedTranscripts


class Transcriber:
    def __init__(self, listener: Union[Listener, ContinuousListener], speech_to_text: BaseSpeechToText):
        self.listener = listener
        self.speech_to_text = speech_to_text

    def read(self) -> str:
        sample = self.listener.read()
        return self.speech_to_text.transcribe_audio(sample)

    def read_detailed(self, *, n_transcriptions: int = 3, segment_timestamps: bool = True) -> DetailedTranscripts:
        sample = self.listener.read()
        return self.speech_to_text.transcribe_audio_detailed(
            sample, n_transcriptions=n_transcriptions, segment_timestamps=segment_timestamps
        )

    def __iter__(self):
        for _ in itertools.count():
            try:
                yield self.read()
            except NoAudioAvailable:
                return None
        return None

    def iter_detailed(self, *, n_transcriptions: int = 3, segment_timestamps: bool = True):
        for _ in itertools.count():
            try:
                yield self.read_detailed(n_transcriptions=n_transcriptions, segment_timestamps=segment_timestamps)
            except NoAudioAvailable:
                return None
        return None
