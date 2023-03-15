# Donkey Ears

A package that provides a common interface to various speech recognition libraries and tools for streaming audio from a
microphone and transcribing it.

## Installation

### Required Non-Python Dependencies

Install [PortAudio 19](http://www.portaudio.com/).
You may need to install the development libraries for PortAudio.

Follow the instruction for installing:

* [Pydub dependencies](https://github.com/jiaaro/pydub#dependencies)

### Optional Dependencies

#### Pocketsphinx

To install for Pocketsphinx, you will need to follow their [installation instructions](https://github.com/bambocher/pocketsphinx-python#installation) for installing the non-Python dependencies.

#### Whisper

Whisper depends on PyTorch, but adding pytorch as a dependency through poetry is difficult (see [PyTorch issue](https://github.com/pytorch/pytorch/issues/26340), [Poetry issue](https://github.com/python-poetry/poetry/issues/4231)).
While PyTorch is listed as a dependency for whisper, you may need to manually install PyTorch by following their installation instructions: https://pytorch.org/get-started/locally/

### Installing Donkey Ears

Once you have installed the non-Python dependencies and any optional dependencies, install Donkey Ears with `poetry install`.

## Speech to Text Models

Donkey Ears provides support for four speech to text engines:
* [Coqui STT](https://stt.readthedocs.io/en/latest/)
* [PocketSphinx](https://github.com/cmusphinx/pocketsphinx)
* [Vosk](https://alphacephei.com/vosk/)
* [Whisper](https://github.com/openai/whisper)

Many engines require a model for the language you want to transcribe.
You must find and download the model before the engine can be used (it will be required in the class's initializer).

## Quick Start

### Transcribing a File

Open a file and transcribe all of it using Whisper:

```python
from donkey_ears.audio.audio_file import AudioFile
from donkey_ears.speech_to_text.whisper import WhisperSpeechToText

a_file = AudioFile("tests/resources/english-one_two_three.wav")
sample = a_file.read(None)
stt_whisper = WhisperSpeechToText("base", download_root="/path/to/models/whisper")

print(stt_whisper.transcribe_audio(sample))
```

will print:

```
 1, 2, 3
```

Open a file and get detailed transcript information using Vosk:

```python
from donkey_ears.audio.audio_file import AudioFile
from donkey_ears.speech_to_text.vosk import VoskSpeechToText

a_file = AudioFile('tests/resources/english-one_two_three.wav')
sample = a_file.read(None)
stt_vosk = VoskSpeechToText("/path/to/models/vosk/vosk-model-small-en-us-0.15")

transcript = stt_vosk.transcribe_audio_detailed(sample, n_transcriptions=3, segment_timestamps=True)
print(transcript)
```

will print:

```
DetailedTranscripts(
    transcripts=[
        DetailedTranscript(
            text="one two three",
            confidence=246.324112,
            segments=[
                TranscriptSegment(text="one", start_time=11.03975, end_time=11.51975),
                TranscriptSegment(text="two", start_time=11.99975, end_time=12.53975),
                TranscriptSegment(text="three", start_time=12.95975, end_time=13.58975),
            ],
        )
    ],
    raw_model_response={
        "alternatives": [
            {
                "confidence": 246.324112,
                "result": [
                    {"start": 11.03975, "end": 11.51975, "word": "one"},
                    {"start": 11.99975, "end": 12.53975, "word": "two"},
                    {"start": 12.95975, "end": 13.58975, "word": "three"},
                ],
                "text": "one two three",
            }
        ]
    },
)
```

### Transcribing from the Microphone

Record segments of audio (separated by silence) and transcribe each segment:

```python
from donkey_ears.audio.microphone import Microphone
from donkey_ears.listeners.audio import SilenceBasedListener
from donkey_ears.listeners.transcriber import Transcriber
from donkey_ears.speech_to_text.whisper import WhisperSpeechToText

mic = Microphone()
silence_rms = mic.read_seconds(3).rms  # Get the rms for silence / background noise
silence_listener = SilenceBasedListener(mic, silence_threshold_rms=silence_rms)

stt_whisper = WhisperSpeechToText("base", download_root="/path/to/models/whisper")

with silence_listener.continuous_listener() as listener:
    transcriber = Transcriber(listener, stt_whisper)
    for text in transcriber:
        print(f"Heard: {text!r}")
```

To get detailed transcriptions instead of just text, use `transcriber.iter_detailed(...)` for the for-loop instead of
just `transcriber`.

## Documentation

### Getting Audio

Transcription requires an audio sample (`donkey_ears.audio.base.AudioSample`), which can come from an audio file:

```python
from donkey_ears.audio.audio_file import AudioFile

a_file = AudioFile("tests/resources/english-one_two_three.wav")
sample = a_file.read(None)
```

or be recorded from the microphone:

```python
from donkey_ears.audio.microphone import Microphone

mic = Microphone()
sample = mic.read_seconds(3)
```

### Transcription

Create an instance of the speech-to-text class you want to use.
The classes share a core set of methods for transcribing audio, described below.
They primarily differ in the initializer and potentially some extra methods for engine-specific features.
To demonstrate the overall similarity, the Vosk and Whisper classes will be used.

```python
from donkey_ears.speech_to_text.vosk import VoskSpeechToText
stt_vosk = VoskSpeechToText("/path/to/models/vosk/vosk-model-small-en-us-0.15")

from donkey_ears.speech_to_text.whisper import WhisperSpeechToText
stt_whisper = WhisperSpeechToText("base", download_root="/path/to/models/whisper")
```

#### Get Simple Transcription

Use the `transcribe_audio` method to get a string of the highest-confidence transcription of the audio sample

```python
from donkey_ears.audio.audio_file import AudioFile
from donkey_ears.speech_to_text.vosk import VoskSpeechToText
from donkey_ears.speech_to_text.whisper import WhisperSpeechToText

sample = AudioFile('tests/resources/english-one_two_three.wav').read(None)
stt_vosk = VoskSpeechToText("/path/to/models/vosk/vosk-model-small-en-us-0.15")
stt_whisper = WhisperSpeechToText("base", download_root="/path/to/models/whisper")
print(f"Vosk: {stt_vosk.transcribe_audio(sample)!r}")
print(f"Whisper: {stt_whisper.transcribe_audio(sample)!r}")
```

will print to standard out:
```
Vosk: 'one two three'
Whisper: ' 1, 2, 3'
```

You may get a warning printed to standard error for Whisper.
They should be safe to ignore.

As you can see, each speech-to-text library may transcribe text differently.

#### Get Detailed Transcription

More detailed transcription information is available through the `transcribe_audio_detailed` method, which can return
multiple transcriptions, their confidences, and timestamps for segments in the audio:

```python
from donkey_ears.audio.audio_file import AudioFile
from donkey_ears.speech_to_text.vosk import VoskSpeechToText
from donkey_ears.speech_to_text.whisper import WhisperSpeechToText

sample = AudioFile('tests/resources/english-one_two_three.wav').read(None)
stt_vosk = VoskSpeechToText("/path/to/models/vosk/vosk-model-small-en-us-0.15")
stt_whisper = WhisperSpeechToText("base", download_root="/path/to/models/whisper")

transcript = stt_vosk.transcribe_audio_detailed(sample, n_transcriptions=3, segment_timestamps=True)
print(transcript)
```

Will print:

```
DetailedTranscripts(
    transcripts=[
        DetailedTranscript(
            text="one two three",
            confidence=246.324112,
            segments=[
                TranscriptSegment(text="one", start_time=11.03975, end_time=11.51975),
                TranscriptSegment(text="two", start_time=11.99975, end_time=12.53975),
                TranscriptSegment(text="three", start_time=12.95975, end_time=13.58975),
            ],
        )
    ],
    raw_model_response={
        "alternatives": [
            {
                "confidence": 246.324112,
                "result": [
                    {"start": 11.03975, "end": 11.51975, "word": "one"},
                    {"start": 11.99975, "end": 12.53975, "word": "two"},
                    {"start": 12.95975, "end": 13.58975, "word": "three"},
                ],
                "text": "one two three",
            }
        ]
    },
)
```

The `DetailedTranscripts` dataclass has two attributes:
* `transcripts`: a list of all transcripts (as `DetailedTranscript` objects), sorted with the highest confidence first
  * The list's length may be less than the number of transcripts requested in cases where the model is not confident
    in any other potential transcription.
* `raw_model_response`: the raw response from the speech-to-text model

Each `DetailedTranscript` dataclass instance has three attributes:
* `text`: the complete transcript
* `confidence`: the model's confidence in the transcript
  * Higher is always more confident
  * These confidences cannot be compared across models
* `segments`: A list of start and end segments of the audio corresponding to parts of the transcript
  * Different models will handle this differently.  In some cases each word will be its own segment whereas in other
    cases multiple words will be in one segment


## Listeners (Streaming Transcription)

The listeners module (`donkey_ears.listeners`) contains classes for working with streaming audio, such as from the
microphone.
A listener class will listen to the audio until a certain condition is met, then return the audio recorded as an
`AudioSample` instance.
The exact condition to wait for depends on the listener.
For example, the `donkey_ears.listeners.audio.TimeBasedListener` class will record audio for a specified duration
whereas the `SilenceBasedListener` will wait until a chunk of audio is below a specified rms.
Other listeners can be constructed by inheriting from the `donkey_ears.listeners.base.BaseListener` class or
`BaseStateListener`.

The example below will record audio until talking stops (i.e. the rms drops to the level of non-talking).
```python
from donkey_ears.audio.microphone import Microphone
from donkey_ears.listeners.audio import SilenceBasedListener

mic = Microphone()

# Determine what the rms when there isn't any speaking
non_speaking_audio = mic.read_seconds(2)
non_speaking_threshold = non_speaking_audio.rms

# Create the listener
listener = SilenceBasedListener(mic, non_speaking_threshold)
speech = listener.read()
# `speech` can then be given to a speech-to-text instance for transcription
```

Note that any audio between calls to `read` will not be recorded.
To constantly listen, create a continuous listener, which will record audio samples using the listener it was
created from.
The samples are stored and can be retrieved using the continuous listener's `read` method.

```python
from donkey_ears.audio.microphone import Microphone
from donkey_ears.listeners.audio import SilenceBasedListener
from donkey_ears.listeners.transcriber import Transcriber
from donkey_ears.speech_to_text.whisper import WhisperSpeechToText

mic = Microphone()
stt_whisper = WhisperSpeechToText("base", download_root="/path/to/models/whisper")

# Determine the rms when there isn't any speaking
non_speaking_audio = mic.read_seconds(2)
non_speaking_threshold = non_speaking_audio.rms

# Create the listener and background listener
listener = SilenceBasedListener(mic, non_speaking_threshold)
with listener.continuous_listener() as clistener:
    transcriber = Transcriber(clistener, stt_whisper)

    first_transcript = transcriber.read()
    # process `first_transcript`, which will be a string transcription of the audio

    second_transcript = transcriber.read()
    # ...

    # Or process as a stream:
    for transcript in transcriber:
        # process `transcript`
        pass
```

Instances of the `Transcriber` class can act as an iterable.
It will wait until audio is available from the provided `listener`, send it to the provided speech-to-text object, then
yield the resulting text transcription.
