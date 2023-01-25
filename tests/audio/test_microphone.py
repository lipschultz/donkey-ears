import io
from unittest.mock import MagicMock, call

import pyaudio
import pytest
from pydub import AudioSegment

from donkey_ears.audio.base import AudioSample
from donkey_ears.audio.microphone import Microphone


class TestMicrophone:
    @staticmethod
    def test_device_index_is_validated_against_available_devices():
        # Arrange
        mock_pa_instance = MagicMock()
        mock_pa_instance.get_device_count = MagicMock(return_value=3)
        mock_pa_instance.terminate = MagicMock()
        pyaudio.PyAudio = MagicMock(return_value=mock_pa_instance)

        # Act
        Microphone(2)

        # Assert
        mock_pa_instance.get_device_count.assert_called_once_with()
        assert mock_pa_instance.terminate.call_count == pyaudio.PyAudio.call_count

    @staticmethod
    def test_device_index_out_of_range_is_invalid():
        # Arrange
        mock_pa_instance = MagicMock()
        mock_pa_instance.get_device_count = MagicMock(return_value=3)
        mock_pa_instance.terminate = MagicMock()
        pyaudio.PyAudio = MagicMock(return_value=mock_pa_instance)

        # Act and Assert
        with pytest.raises(ValueError):
            Microphone(5)

        assert mock_pa_instance.terminate.call_count == pyaudio.PyAudio.call_count

    @staticmethod
    def test_default_device_used_when_device_index_is_none():
        # Arrange
        mock_pa_instance = MagicMock()
        mock_pa_instance.get_default_input_device_info = MagicMock(return_value={"device": "information"})
        mock_pa_instance.terminate = MagicMock()
        pyaudio.PyAudio = MagicMock(return_value=mock_pa_instance)
        subject = Microphone()

        # Act
        actual_device_info = subject.audio_device_information

        # Assert
        assert actual_device_info == {"device": "information"}
        mock_pa_instance.get_default_input_device_info.assert_called_once_with()
        assert mock_pa_instance.terminate.call_count == pyaudio.PyAudio.call_count

    @staticmethod
    def test_device_used_when_device_index_is_given():
        # Arrange
        mock_pa_instance = MagicMock()
        mock_pa_instance.get_device_info_by_index = MagicMock(return_value={"device": "information"})
        mock_pa_instance.get_device_count = MagicMock(return_value=5)
        mock_pa_instance.terminate = MagicMock()
        pyaudio.PyAudio = MagicMock(return_value=mock_pa_instance)
        subject = Microphone(3)

        # Act
        actual_device_info = subject.audio_device_information

        # Assert
        assert actual_device_info == {"device": "information"}
        mock_pa_instance.get_device_info_by_index.assert_called_once_with(3)
        assert mock_pa_instance.terminate.call_count == pyaudio.PyAudio.call_count

    @staticmethod
    def test_read_bytes():
        # Arrange
        mock_input_source = MagicMock()
        mock_input_source.read = MagicMock(return_value=b"0")
        mock_pa_instance = MagicMock()
        mock_pa_instance.open = MagicMock(return_value=mock_input_source)
        mock_pa_instance.terminate = MagicMock()
        pyaudio.PyAudio = MagicMock(return_value=mock_pa_instance)
        subject = Microphone()
        pyaudio.PyAudio.reset_mock()
        mock_pa_instance.terminate.reset_mock()

        # Act
        bytes_read = subject.read_bytes(1)

        # Assert
        pyaudio.PyAudio.assert_has_calls([call(), call()])
        assert pyaudio.PyAudio.call_count == 2
        mock_pa_instance.open.assert_called_once_with(
            input_device_index=None,
            channels=subject.n_channels,
            format=pyaudio.paInt16,
            rate=subject.frame_rate,
            input=True,
        )
        assert mock_pa_instance.terminate.call_count == pyaudio.PyAudio.call_count
        mock_input_source.read.assert_called_once_with(1, exception_on_overflow=False)
        assert bytes_read == b"0"

    @staticmethod
    def test_read_pydub():
        # Arrange
        mock_pa_instance = MagicMock()
        mock_pa_instance.terminate = MagicMock()
        pyaudio.PyAudio = MagicMock(return_value=mock_pa_instance)
        subject = Microphone()
        subject.read_bytes = MagicMock(return_value=b"0")

        with io.BytesIO(b"0") as fp:
            expected_audio_segment = AudioSegment.from_raw(
                fp,
                sample_width=subject.frame_width,
                frame_rate=subject.frame_rate,
                channels=subject.n_channels,
            )

        # Act
        actual_audio_segment = subject.read_pydub(2)

        # Assert
        assert mock_pa_instance.terminate.call_count == pyaudio.PyAudio.call_count
        subject.read_bytes.assert_called_once_with(2)
        assert actual_audio_segment == expected_audio_segment

    @staticmethod
    def test_read_with_frames_given():
        # Arrange
        mock_pa_instance = MagicMock()
        mock_pa_instance.terminate = MagicMock()
        pyaudio.PyAudio = MagicMock(return_value=mock_pa_instance)
        subject = Microphone()
        subject.read_bytes = MagicMock(return_value=b"0")

        with io.BytesIO(b"0") as fp:
            expected_audio = AudioSample(
                AudioSegment.from_raw(
                    fp,
                    sample_width=subject.frame_width,
                    frame_rate=subject.frame_rate,
                    channels=subject.n_channels,
                )
            )

        # Act
        actual_audio = subject.read(2)

        # Assert
        assert mock_pa_instance.terminate.call_count == pyaudio.PyAudio.call_count
        subject.read_bytes.assert_called_once_with(2)
        assert actual_audio == expected_audio

    @staticmethod
    def test_default_frame_count_used_when_read_called_with_no_frame_count_given():
        # Arrange
        mock_pa_instance = MagicMock()
        mock_pa_instance.terminate = MagicMock()
        pyaudio.PyAudio = MagicMock(return_value=mock_pa_instance)
        subject = Microphone()
        subject.read_bytes = MagicMock(return_value=b"0")

        with io.BytesIO(b"0") as fp:
            expected_audio = AudioSample(
                AudioSegment.from_raw(
                    fp,
                    sample_width=subject.frame_width,
                    frame_rate=subject.frame_rate,
                    channels=subject.n_channels,
                )
            )

        # Act
        actual_audio = subject.read(None)

        # Assert
        assert mock_pa_instance.terminate.call_count == pyaudio.PyAudio.call_count
        subject.read_bytes.assert_called_once_with(subject.DEFAULT_READ_DURATION_SECONDS)
        assert actual_audio == expected_audio
