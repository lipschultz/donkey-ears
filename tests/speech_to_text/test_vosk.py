import json
from unittest.mock import MagicMock, call

from donkey_ears.speech_to_text import vosk
from donkey_ears.speech_to_text.base import DetailedTranscript, DetailedTranscripts, TranscriptSegment


def test_creating_instance_passes_correct_values_to_vosk():
    # Arrange
    vosk.Model = MagicMock(return_value="any_model")
    vosk.KaldiRecognizer = MagicMock()

    # Act
    vosk.VoskSpeechToText("/path/to/model", 16_000, 16, 1)

    # Assert
    vosk.Model.assert_called_once_with("/path/to/model")
    vosk.KaldiRecognizer.assert_called_once_with("any_model", 16_000)


def test_changing_model_path_creates_new_model():
    """Changing a field in the copy should not have an effect on the original"""
    # Arrange
    vosk.Model = MagicMock(side_effect=["first model", "second model"])
    vosk.KaldiRecognizer = MagicMock(side_effect=["first recognizer", "second recognizer"])
    subject = vosk.VoskSpeechToText("/path/to/model", 16_000, 16, 1)

    # Act
    subject.model_path = "/path/to/different/model"

    # Assert
    vosk.Model.assert_has_calls([call("/path/to/model"), call("/path/to/different/model")])
    assert vosk.Model.call_count == 2
    vosk.KaldiRecognizer.assert_has_calls([call("first model", 16_000), call("second model", 16_000)])
    assert vosk.KaldiRecognizer.call_count == 2
    assert subject._model == "second model"
    assert subject._recognizer == "second recognizer"


def test_changing_field_in_copy_does_not_affect_original():
    """Changing a field in the copy should not have an effect on the original"""
    # Arrange
    vosk.Model = MagicMock()
    vosk.KaldiRecognizer = MagicMock()
    subject = vosk.VoskSpeechToText("/path/to/model", 16_000, 16, 1)
    copy = subject.copy()

    # Act
    copy.model_path = "/path/to/different/model"
    copy.frame_rate = 32_000
    copy.bit_depth = 8
    copy.n_channels = 2

    # Assert
    assert subject.model_path == "/path/to/model"
    assert subject.frame_rate == 16_000
    assert subject.bit_depth == 16
    assert subject.n_channels == 1


def test_changing_field_in_original_does_not_affect_copy():
    """Changing a field in the copy should not have an effect on the original"""
    # Arrange
    vosk.Model = MagicMock()
    vosk.KaldiRecognizer = MagicMock()
    subject = vosk.VoskSpeechToText("/path/to/model", 16_000, 16, 1)
    copy = subject.copy()

    # Act
    subject.model_path = "/path/to/different/model"
    subject.frame_rate = 32_000
    subject.bit_depth = 8
    subject.n_channels = 2

    # Assert
    assert copy.model_path == "/path/to/model"
    assert copy.frame_rate == 16_000
    assert copy.bit_depth == 16
    assert copy.n_channels == 1


def test_detailed_transcription_with_segments():
    # Arrange
    vosk.Model = MagicMock(return_value="first model")
    mock_recognizer_instance = MagicMock()
    vosk.KaldiRecognizer = MagicMock(return_value=mock_recognizer_instance)
    mock_recognizer_instance.SetMaxAlternatives = MagicMock()
    mock_recognizer_instance.SetWords = MagicMock()
    mock_recognizer_instance.AcceptWaveform = MagicMock()
    raw_final_result = {
        "alternatives": [
            {
                "text": "first transcription",
                "confidence": 0.99,
                "result": [
                    {"word": "first", "start": 0, "end": 0.8},
                    {"word": "transcription", "start": 0.9, "end": 1.7},
                ],
            },
            {
                "text": "second transcription",
                "confidence": 0.88,
                "result": [
                    {"word": "second", "start": 0.1, "end": 1.1},
                    {"word": "transcription", "start": 1.3, "end": 2.2},
                ],
            },
        ]
    }
    mock_recognizer_instance.FinalResult = MagicMock(return_value=json.dumps(raw_final_result))
    subject = vosk.VoskSpeechToText("/path/to/model", 16_000, 16, 1)

    converted_audio = MagicMock()
    converted_audio.to_bytes = MagicMock(return_value=b"audio in bytes")
    audio = MagicMock()
    audio.convert = MagicMock(return_value=converted_audio)

    expected_transcriptions = DetailedTranscripts(
        [
            DetailedTranscript(
                "first transcription",
                0.99,
                [TranscriptSegment("first", 0, 0.8), TranscriptSegment("transcription", 0.9, 1.7)],
            ),
            DetailedTranscript(
                "second transcription",
                0.88,
                [TranscriptSegment("second", 0.1, 1.1), TranscriptSegment("transcription", 1.3, 2.2)],
            ),
        ],
        raw_final_result,
    )

    # Act
    actual_transcriptions = subject.transcribe_audio_detailed(audio, n_transcriptions=2, segment_timestamps=True)

    # Assert
    mock_recognizer_instance.SetMaxAlternatives.assert_called_once_with(2)
    mock_recognizer_instance.SetWords.assert_called_once_with(True)
    mock_recognizer_instance.AcceptWaveform.assert_called_once_with(b"audio in bytes")
    audio.convert.assert_called_once_with(
        sample_width=subject.sample_width,
        frame_rate=subject.frame_rate,
        n_channels=subject.n_channels,
    )
    assert expected_transcriptions == actual_transcriptions


def test_detailed_transcription_with_no_segments():
    # Arrange
    vosk.Model = MagicMock(return_value="first model")
    mock_recognizer_instance = MagicMock()
    vosk.KaldiRecognizer = MagicMock(return_value=mock_recognizer_instance)
    mock_recognizer_instance.SetMaxAlternatives = MagicMock()
    mock_recognizer_instance.SetWords = MagicMock()
    mock_recognizer_instance.AcceptWaveform = MagicMock()
    raw_final_result = {
        "alternatives": [
            {"text": "first transcription", "confidence": 0.99},
            {"text": "second transcription", "confidence": 0.88},
        ]
    }
    mock_recognizer_instance.FinalResult = MagicMock(return_value=json.dumps(raw_final_result))
    subject = vosk.VoskSpeechToText("/path/to/model", 16_000, 16, 1)

    converted_audio = MagicMock()
    converted_audio.to_bytes = MagicMock(return_value=b"audio in bytes")
    audio = MagicMock()
    audio.convert = MagicMock(return_value=converted_audio)

    expected_transcriptions = DetailedTranscripts(
        [
            DetailedTranscript("first transcription", 0.99, None),
            DetailedTranscript("second transcription", 0.88, None),
        ],
        raw_final_result,
    )

    # Act
    actual_transcriptions = subject.transcribe_audio_detailed(audio, n_transcriptions=2, segment_timestamps=False)

    # Assert
    mock_recognizer_instance.SetMaxAlternatives.assert_called_once_with(2)
    mock_recognizer_instance.SetWords.assert_called_once_with(False)
    mock_recognizer_instance.AcceptWaveform.assert_called_once_with(b"audio in bytes")
    audio.convert.assert_called_once_with(
        sample_width=subject.sample_width,
        frame_rate=subject.frame_rate,
        n_channels=subject.n_channels,
    )
    assert expected_transcriptions == actual_transcriptions


def test_detailed_transcription_when_no_transcription_generated():
    # TODO
    pass


def test_restricting_vocabulary_and_excluding_unknown_token():
    # Arrange
    vosk.Model = MagicMock()
    mock_recognizer_instance = MagicMock()
    vosk.KaldiRecognizer = MagicMock(return_value=mock_recognizer_instance)
    mock_recognizer_instance.SetGrammar = MagicMock()
    subject = vosk.VoskSpeechToText("/path/to/model", 16_000, 16, 1)

    # Act
    subject.restrict_vocabulary_to(["words", "to", "restrict", "to"], False)

    # Assert
    mock_recognizer_instance.SetGrammar.assert_called_once()
    assert set(json.loads(mock_recognizer_instance.SetGrammar.call_args[0][0])) == {"restrict", "to", "words"}


def test_restricting_vocabulary_and_including_unknown_token():
    # Arrange
    vosk.Model = MagicMock()
    mock_recognizer_instance = MagicMock()
    vosk.KaldiRecognizer = MagicMock(return_value=mock_recognizer_instance)
    mock_recognizer_instance.SetGrammar = MagicMock()
    subject = vosk.VoskSpeechToText("/path/to/model", 16_000, 16, 1)

    # Act
    subject.restrict_vocabulary_to(["words", "to", "restrict", "to"], True)

    # Assert
    mock_recognizer_instance.SetGrammar.assert_called_once()
    assert set(json.loads(mock_recognizer_instance.SetGrammar.call_args[0][0])) == {"restrict", "to", "words", "[unk]"}
