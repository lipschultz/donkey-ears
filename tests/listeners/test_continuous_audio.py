from unittest.mock import MagicMock

import pytest

from donkey_ears.audio.base import AudioSample
from donkey_ears.listeners.continuous_audio import ContinuousListener, NoAudioAvailable


class TestContinuousListener:
    @staticmethod
    def test_is_listening_returns_false_after_creation():
        subject = ContinuousListener(MagicMock())

        assert subject.is_listening is False

    @staticmethod
    def test_is_listening_returns_true_after_starting():
        subject = ContinuousListener(MagicMock())
        subject.start()

        assert subject.is_listening is True

    @staticmethod
    def test_is_listening_returns_false_after_starting_then_stopping():
        subject = ContinuousListener(MagicMock())
        subject.start()
        subject.stop()

        assert subject.is_listening is False

    @staticmethod
    def test_source_read_called_when_listener_started():
        listener = MagicMock()
        listener.read = MagicMock()
        subject = ContinuousListener(listener)
        subject.start()
        subject.stop()

        listener.read.assert_called()

    @staticmethod
    def test_read():
        listener = MagicMock()
        listener.read = MagicMock(side_effect=[AudioSample.generate_silence(1, 44100), EOFError])
        subject = ContinuousListener(listener)
        subject.start()
        subject.stop()

        result = subject.read()

        assert result == AudioSample.generate_silence(1, 44100)
        with pytest.raises(NoAudioAvailable):
            subject.read()

    @staticmethod
    def test_read_while_started():
        listener = MagicMock()
        listener.read = MagicMock(side_effect=[AudioSample.generate_silence(1, 44100), EOFError])
        subject = ContinuousListener(listener)
        subject.start()

        result = subject.read()
        subject.stop()

        assert result == AudioSample.generate_silence(1, 44100)
        with pytest.raises(NoAudioAvailable):
            subject.read()

    @staticmethod
    def test_read_raises_exception_when_listener_immediately_raises_eoferror():
        listener = MagicMock()
        listener.read = MagicMock(side_effect=[EOFError])
        subject = ContinuousListener(listener)
        subject.start()
        subject.stop()

        with pytest.raises(NoAudioAvailable):
            subject.read()

    @staticmethod
    def test_read_raises_exception_when_not_listening_and_queue_empty():
        subject = ContinuousListener(MagicMock())

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
        listener = MagicMock()
        listener.read = MagicMock(side_effect=[AudioSample.generate_silence(1, 44100), EOFError])
        subject = ContinuousListener(listener)
        subject._recordings = MagicMock()
        subject._recordings.get = MagicMock()
        subject._recordings.empty = MagicMock(return_value=False)
        subject.start()
        subject.stop()

        subject.read(wait)

        subject._recordings.get.assert_called_once()
        subject._recordings.get.assert_called_once_with(expected_blocking, expected_timeout)

    @staticmethod
    def test_iterating_over_listener_returns_samples_recorded():
        listener = MagicMock()
        listener.read = MagicMock(
            side_effect=[AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100), EOFError]
        )
        subject = ContinuousListener(listener)
        subject.start()
        subject.stop()

        results = list(subject)

        assert results == [AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100)]
