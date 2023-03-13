from unittest.mock import MagicMock

import pytest

from donkey_ears.audio.base import AudioSample
from donkey_ears.listeners.base import BaseListener


class TestBaseListener:
    @staticmethod
    def test_read_sample():
        audio_source = MagicMock()
        audio_source.read = MagicMock(return_value=AudioSample.generate_silence(1, 44100))
        subject = BaseListener(audio_source)

        actual = subject.read()

        audio_source.read.assert_called_once()
        assert actual == AudioSample.generate_silence(1, 44100)

    @staticmethod
    def test_chunk_size_passed_to_source_read_method():
        audio_source = MagicMock()
        audio_source.read = MagicMock(return_value=AudioSample.generate_silence(1, 44100))
        subject = BaseListener(audio_source)

        subject.read(22050)

        audio_source.read.assert_called_once_with(22050)

    @staticmethod
    def test_no_audio_available():
        audio_source = MagicMock()
        audio_source.read = MagicMock(side_effect=EOFError)
        subject = BaseListener(audio_source)

        with pytest.raises(EOFError):
            subject.read()

    @staticmethod
    def test_iterating_over_listener_returns_samples_recorded():
        audio_source = MagicMock()
        audio_source.read = MagicMock(
            side_effect=[AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100), EOFError]
        )
        subject = BaseListener(audio_source)

        results = list(subject)

        assert results == [AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100)]
