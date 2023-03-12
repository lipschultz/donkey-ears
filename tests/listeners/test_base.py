from unittest.mock import MagicMock

import pytest

from donkey_ears.audio.base import AudioSample
from donkey_ears.listeners.base import BaseContinuousListener, BaseListener, NoAudioAvailable


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


class TestContinuousBaseListener:
    @staticmethod
    def test_is_listening_returns_false_after_creation():
        subject = BaseContinuousListener(MagicMock())

        assert subject.is_listening is False

    @staticmethod
    def test_is_listening_returns_true_after_starting():
        subject = BaseContinuousListener(MagicMock())
        subject.start()

        assert subject.is_listening is True

    @staticmethod
    def test_is_listening_returns_false_after_starting_then_stopping():
        subject = BaseContinuousListener(MagicMock())
        subject.start()
        subject.stop()

        assert subject.is_listening is False

    @staticmethod
    def test_source_read_called_when_listener_started():
        audio_source = MagicMock()
        audio_source.read = MagicMock()
        subject = BaseContinuousListener(audio_source)
        subject.start()
        subject.stop()

        audio_source.read.assert_called()

    @staticmethod
    def test_source_audio_added_to_queue():
        audio_source = MagicMock()
        audio_source.read = MagicMock(side_effect=[AudioSample.generate_silence(1, 44100), EOFError])
        subject = BaseContinuousListener(audio_source)
        subject.start()
        subject.stop()

        result = subject.queue.get()
        assert result == AudioSample.generate_silence(1, 44100)
        assert subject.empty()

    @staticmethod
    def test_read():
        audio_source = MagicMock()
        audio_source.read = MagicMock(side_effect=[AudioSample.generate_silence(1, 44100), EOFError])
        subject = BaseContinuousListener(audio_source)
        subject.start()
        subject.stop()

        result = subject.read()
        assert result == AudioSample.generate_silence(1, 44100)
        assert subject.empty()

    @staticmethod
    def test_read_raises_exception_when_not_listening_and_queue_empty():
        subject = BaseContinuousListener(MagicMock())

        with pytest.raises(NoAudioAvailable):
            subject.read()

    @staticmethod
    @pytest.mark.parametrize(
        "wait, expected_blocking, expected_timeout",
        [
            (False, False, None),
            (0, False, None),
            (0.5, True, 0.5),
            (1, True, 1),
            (True, True, None),
        ],
    )
    def test_read_passes_arguments_to_queue_get(wait, expected_blocking, expected_timeout):
        audio_source = MagicMock()
        audio_source.read = MagicMock(side_effect=[AudioSample.generate_silence(1, 44100), EOFError])
        subject = BaseContinuousListener(audio_source)
        subject.queue = MagicMock()
        subject.queue.get = MagicMock()
        subject.queue.empty = MagicMock(return_value=False)
        subject.start()
        subject.stop()

        subject.read(wait)

        subject.queue.get.assert_called_once()
        subject.queue.get.assert_called_once_with(expected_blocking, expected_timeout)

    @staticmethod
    def test_empty_returns_true_when_not_started():
        subject = BaseContinuousListener(MagicMock())

        assert subject.empty() is True

    @staticmethod
    def test_empty_returns_false_when_audio_recorded():
        subject = BaseContinuousListener(MagicMock())
        subject.start()
        subject.stop()

        assert subject.empty() is False

    @staticmethod
    def test_empty_returns_true_when_audio_recorded_but_queue_has_been_emptied():
        audio_source = MagicMock()
        audio_source.read = MagicMock(side_effect=[AudioSample.generate_silence(1, 44100), EOFError])
        subject = BaseContinuousListener(audio_source)
        subject.start()
        subject.stop()

        subject.read(False)

        assert subject.empty() is True
