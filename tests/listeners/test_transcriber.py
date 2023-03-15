from unittest.mock import MagicMock, call

from donkey_ears.audio.base import AudioSample
from donkey_ears.listeners.audio import ContinuousListener, Listener
from donkey_ears.listeners.transcriber import Transcriber
from donkey_ears.speech_to_text.base import DetailedTranscript, DetailedTranscripts


class TestTranscriber:
    @staticmethod
    def test_read_gets_sample_from_listener_and_sends_to_speech_to_text():
        listener = MagicMock()
        audio_sample = AudioSample.generate_silence(1, 44100)
        listener.read = MagicMock(return_value=audio_sample)

        speech_to_text = MagicMock()
        speech_to_text.transcribe_audio = MagicMock(return_value="any text")

        subject = Transcriber(listener, speech_to_text)

        actual = subject.read()

        assert actual == "any text"
        listener.read.assert_called_once_with()
        speech_to_text.transcribe_audio.assert_called_once_with(audio_sample)

    @staticmethod
    def test_read_detailed_gets_sample_from_listener_and_sends_to_speech_to_text():
        listener = MagicMock()
        audio_sample = AudioSample.generate_silence(1, 44100)
        listener.read = MagicMock(return_value=audio_sample)

        speech_to_text = MagicMock()
        detailed_transcription = DetailedTranscripts(
            [
                DetailedTranscript("any transcript 1", 0.99, None),
                DetailedTranscript("any transcript 2", 0.89, None),
            ],
            None,
        )
        speech_to_text.transcribe_audio_detailed = MagicMock(return_value=detailed_transcription)

        subject = Transcriber(listener, speech_to_text)

        actual = subject.read_detailed(n_transcriptions=7, segment_timestamps=False)

        assert actual is detailed_transcription
        listener.read.assert_called_once_with()
        speech_to_text.transcribe_audio_detailed.assert_called_once_with(
            audio_sample, n_transcriptions=7, segment_timestamps=False
        )

    @staticmethod
    def test_listener_works_as_audio_source():
        audio_source = MagicMock()
        audio_samples = [AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100)]
        audio_source.read = MagicMock(side_effect=audio_samples + [EOFError])
        listener = Listener(audio_source)

        speech_to_text = MagicMock()
        speech_to_text.transcribe_audio = MagicMock(return_value="any text")

        subject = Transcriber(listener, speech_to_text)

        actual = subject.read()

        assert actual == "any text"
        speech_to_text.transcribe_audio.assert_called_once_with(audio_samples[0])

    @staticmethod
    def test_continuous_listener_works_as_audio_source():
        audio_source = MagicMock()
        audio_samples = [AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100)]
        audio_source.read = MagicMock(side_effect=audio_samples + [EOFError])
        raw_listener = Listener(audio_source)
        listener = ContinuousListener(raw_listener)

        speech_to_text = MagicMock()
        speech_to_text.transcribe_audio = MagicMock(return_value="any text")

        subject = Transcriber(listener, speech_to_text)

        with listener.listen():
            actual = subject.read()

        assert actual == "any text"
        speech_to_text.transcribe_audio.assert_called_once_with(audio_samples[0])

    @staticmethod
    def test_iterating_over_transcriber_returns_text_of_audio_recorded():
        audio_source = MagicMock()
        audio_samples = [AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100)]
        audio_source.read = MagicMock(side_effect=audio_samples + [EOFError])
        listener = Listener(audio_source)

        speech_to_text = MagicMock()
        speech_to_text.transcribe_audio = MagicMock(side_effect=["any text", "more text"])

        subject = Transcriber(listener, speech_to_text)

        actual = list(subject)

        assert speech_to_text.transcribe_audio.call_count == 2
        speech_to_text.transcribe_audio.assert_has_calls([call(audio_samples[0]), call(audio_samples[1])])
        assert actual == ["any text", "more text"]

    @staticmethod
    def test_empty_iterable_when_iterating_over_transcriber_with_no_audio():
        audio_source = MagicMock()
        audio_source.read = MagicMock(side_effect=[EOFError])
        listener = Listener(audio_source)

        speech_to_text = MagicMock()
        speech_to_text.transcribe_audio = MagicMock(side_effect=["any text", "more text"])

        subject = Transcriber(listener, speech_to_text)

        actual = list(subject)

        assert len(actual) == 0

    @staticmethod
    def test_iterating_over_transcriber_detailed_iter_returns_details_of_transcriptions():
        audio_source = MagicMock()
        audio_samples = [AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100)]
        audio_source.read = MagicMock(side_effect=audio_samples + [EOFError])
        listener = Listener(audio_source)

        speech_to_text = MagicMock()
        detailed_transcripts = [
            DetailedTranscripts(
                [
                    DetailedTranscript("any transcript 1", 0.99, None),
                    DetailedTranscript("any transcript 2", 0.89, None),
                ],
                None,
            ),
            DetailedTranscripts(
                [
                    DetailedTranscript("another transcript 1", 0.97, None),
                    DetailedTranscript("another transcript 2", 0.87, None),
                ],
                None,
            ),
        ]
        speech_to_text.transcribe_audio_detailed = MagicMock(side_effect=detailed_transcripts)

        subject = Transcriber(listener, speech_to_text)

        actual = list(subject.iter_detailed(n_transcriptions=2, segment_timestamps=False))

        assert speech_to_text.transcribe_audio_detailed.call_count == 2
        speech_to_text.transcribe_audio_detailed.assert_has_calls(
            [
                call(audio_samples[0], n_transcriptions=2, segment_timestamps=False),
                call(audio_samples[1], n_transcriptions=2, segment_timestamps=False),
            ]
        )
        assert actual == detailed_transcripts

    @staticmethod
    def test_empty_iterable_when_iterating_over_detailed_iter_with_no_audio():
        audio_source = MagicMock()
        audio_source.read = MagicMock(side_effect=[EOFError])
        listener = Listener(audio_source)

        speech_to_text = MagicMock()
        speech_to_text.transcribe_audio_detailed = MagicMock()

        subject = Transcriber(listener, speech_to_text)

        actual = list(subject)

        speech_to_text.transcribe_audio_detailed.assert_not_called()
        assert len(actual) == 0
