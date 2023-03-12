import queue
import threading
from dataclasses import dataclass
from enum import Enum
from queue import SimpleQueue
from typing import Optional, Union

from loguru import logger

from donkey_ears.audio.base import AudioSample, BaseAudioSource


class FrameStateEnum(Enum):
    LISTEN = "LISTEN"
    PAUSE = "PAUSE"
    STOP = "STOP"


@dataclass
class AnnotatedFrame:
    frame: AudioSample
    state: FrameStateEnum


class ListenerRunningError(Exception):
    pass


class NoAudioAvailable(Exception):
    pass


class ContinuousBaseListener:
    def __init__(self, source: BaseAudioSource):
        self.source = source
        self._thread = None
        self.queue = SimpleQueue()
        self._please_shutdown_thread = False

    def start(self) -> None:
        """
        Start listening in a background thread.  Audio samples will be recorded and available through the ``read``
        method.
        """
        if self._thread is not None:
            if self._thread.is_alive():
                raise ListenerRunningError(f"{self} is already running")
            self.stop()
        self._please_shutdown_thread = False
        self._thread = threading.Thread(target=self._listen)
        self._thread.daemon = True
        self._thread.start()

    def stop(self, timeout: Optional[int] = None) -> bool:
        """
        Stop listening in the background.

        ``timeout`` indicates how long to wait when joining the background listener thread.  If ``None`` (default),
        then don't wait.

        Returns ``True`` if the listening thread stopped or ``timeout`` was ``None``, ``False`` otherwise.

        Audio already recorded will still be available through the ``read`` method.
        """
        self._please_shutdown_thread = True
        self._thread.join(timeout)
        return_value = timeout is None or not self._thread.is_alive()
        self._thread = None
        return return_value

    @property
    def is_listening(self) -> bool:
        """
        Return whether the listener is running.
        """
        return self._thread is not None and self._thread.is_alive()

    def _get_audio(self) -> AudioSample:
        """
        Record audio from the source and return it.
        """
        return self.source.read(None)

    def _listen(self):
        """
        The method that does the actual listening and adding the audio to a queue.
        """
        while not self._please_shutdown_thread and self.is_listening:
            try:
                audio = self._get_audio()
                # logger.debug(f"Received audio: {audio}")
                self.queue.put(audio)
            except EOFError:
                logger.info("Background listener reached end of file")
                break
            except StopIteration:
                logger.info("Background listener received StopIteration")
                break
            except Exception as exc:
                logger.exception(f"Background listener received exception: {type(exc)}")
                break

    def read(self, wait: Union[int, float, bool] = True) -> AudioSample:
        """
        Return the next audio sample recorded.

        If the listener is stopped and there's no audio available, then ``NoAudioAvailable`` will be raised.

        If no audio sample is currently available and the listener is still running, then ``wait`` will be used:
        * When ``False`` or ``0``, raise a ``NoAudioAvailable`` exception immediately.
        * When a positive number, wait up to that number of seconds for audio.  Return an audio segment if one is
          received by then, otherwise raise a ``NoAudioAvailable`` exception.
        * When ``True`` (default), wait indefinitely for an audio sample.

        """
        if not self.is_listening and self.empty():
            raise NoAudioAvailable()

        blocking = not (wait is False or wait == 0)
        timeout = wait if isinstance(wait, (int, float)) and not isinstance(wait, bool) and wait > 0 else None
        try:
            return self.queue.get(blocking, timeout)
        except queue.Empty:
            raise NoAudioAvailable()

    def empty(self) -> bool:
        """
        Returns True if there is nothing currently available to read, False otherwise.
        """
        return self.queue.empty()
