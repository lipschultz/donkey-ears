try:
    from donkey_ears.speech_to_text.coqui_stt import CoquiSpeechToText
except ModuleNotFoundError:
    pass

try:
    from donkey_ears.speech_to_text.sphinx import SphinxSpeechToText
except ModuleNotFoundError:
    pass

try:
    from donkey_ears.speech_to_text.vosk import VoskSpeechToText
except ModuleNotFoundError:
    pass

try:
    from donkey_ears.speech_to_text.whisper import WhisperSpeechToText
except ModuleNotFoundError:
    pass
