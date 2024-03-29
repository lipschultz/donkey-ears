import json
from pathlib import Path
from typing import Iterable, Union

from vosk import KaldiRecognizer, Model

from donkey_ears.audio.base import AudioSample
from donkey_ears.speech_to_text.base import BaseSpeechToText, DetailedTranscript, DetailedTranscripts, TranscriptSegment


class VoskSpeechToText(BaseSpeechToText):
    FRAME_RATE = 16_000
    BIT_DEPTH = 16
    N_CHANNELS = 1

    UNRECOGNIZED_TOKEN = "[unk]"  # nosec hardcoded_password_string

    def __init__(
        self,
        model_path: Union[str, Path],
        frame_rate: int = FRAME_RATE,
        bit_depth: int = BIT_DEPTH,
        n_channels: int = N_CHANNELS,
    ):
        super().__init__()
        self._model = None  # type: Model
        self._recognizer = None  # type: KaldiRecognizer
        self.frame_rate = frame_rate
        self.bit_depth = bit_depth
        self.n_channels = n_channels

        self.model_path = model_path

    @property
    def model_path(self) -> Union[str, Path]:
        return self._model_path

    @model_path.setter
    def model_path(self, model_path: Union[str, Path]):
        self._model_path = model_path
        self._model = Model(self.model_path)
        self._recognizer = KaldiRecognizer(self._model, self.frame_rate)

    def copy(self) -> "VoskSpeechToText":
        """
        Return a copy of the object.  Changes to the original will not affect the copy, nor will changes to the copy
        affect the original.
        """
        return VoskSpeechToText(self.model_path, self.frame_rate, self.bit_depth, self.n_channels)

    @property
    def sample_width(self) -> int:
        return self.bit_depth // 8

    def transcribe_audio_detailed(
        self,
        audio: AudioSample,
        *,
        n_transcriptions: int = 3,
        segment_timestamps: bool = True,
    ) -> DetailedTranscripts:
        """
        Transcribe an audio sample, returning a transcript with extra details about the transcription, such as
        timestamps and confidence.

        ``n_transcriptions`` indicates the maximum number of transcripts to generate.

        ``segment_timestamps``, if True, will provide start and end timestamps for each word in the transcript.
        """
        self._recognizer.SetMaxAlternatives(n_transcriptions)
        self._recognizer.SetWords(segment_timestamps)
        self._recognizer.AcceptWaveform(
            audio.convert(
                sample_width=self.sample_width,
                frame_rate=self.frame_rate,
                n_channels=self.n_channels,
            ).to_bytes()
        )
        result = json.loads(self._recognizer.FinalResult())

        return DetailedTranscripts(
            [
                DetailedTranscript(
                    transcript["text"],
                    transcript["confidence"],
                    [
                        TranscriptSegment(segment["word"], segment["start"], segment["end"])
                        for segment in transcript["result"]
                    ]
                    if "result" in transcript
                    else None,
                )
                for transcript in result["alternatives"]
            ],
            result,
        )

    def restrict_vocabulary_to(self, vocabulary: Iterable[str], include_unrecognized_token: bool = True) -> None:
        """
        Restrict the model to only use the provided vocabulary.

        Duplicate terms in the vocabulary are removed.

        If `include_unrecognized_token` is True (default), then include the "unrecognized" token (`[unk]')
        """
        vocabulary = set(vocabulary)
        if include_unrecognized_token:
            vocabulary |= {"[unk]"}
        self._recognizer.SetGrammar(json.dumps(list(vocabulary)))
