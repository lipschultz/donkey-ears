import itertools
import math
import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from loguru import logger

from donkey_ears.audio.base import AudioSample, BaseAudioSource, TimeType


class NoAudioAvailable(Exception):
    pass


class ListenerRunningError(Exception):
    pass


class Listener:
    def __init__(self, source: BaseAudioSource):
        self.source = source

    def read(self, n_frames: int = 2**14) -> AudioSample:
        """
        Read and return an audio sample from the source.

        ``n_frames`` is the number of frames to read from the source.

        Raises ``NoAudioAvailable`` if there is no audio available from the source (e.g. end of file was reached).
        """
        try:
            return self.source.read(n_frames)
        except EOFError as exc:
            raise NoAudioAvailable() from exc

    def __iter__(self):
        for _ in itertools.count():
            try:
                yield self.read()
            except NoAudioAvailable:
                return None
        return None

    def continuous_listener(self) -> "ContinuousListener":
        return ContinuousListener(self)


class _EndOfListener:
    """Used to mark when the listener has stopped listening and no more audio should be expected."""


class ContinuousListener:
    """
    Always listen to the source, storing audio segments to be accessed later (via ``read``).
    """

    END_OF_LISTENER = _EndOfListener()

    def __init__(self, listener: Listener):
        self.listener = listener

        self._thread = None  # type: Optional[threading.Thread]
        self._recordings = queue.SimpleQueue()  # type: queue.SimpleQueue[Union[AudioSample, _EndOfListener]]
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
        if self._thread is not None:
            self._thread.join(timeout)
            return_value = timeout is None or not self._thread.is_alive()
        else:
            return_value = True
        self._thread = None
        return return_value

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def is_listening(self) -> bool:
        """
        Return whether the listener is running.
        """
        return self._thread is not None and self._thread.is_alive()

    def _get_audio(self) -> AudioSample:
        """
        Record audio from the listener and return it.
        """
        return self.listener.read()

    def _store_recording(self, audio: AudioSample) -> None:
        self._recordings.put(audio)

    def _store_recording_stopped(self) -> None:
        self._recordings.put(self.END_OF_LISTENER)

    def _listen(self):
        """
        The method that does the actual listening and adding the audio to a queue.
        """
        while not self._please_shutdown_thread and self.is_listening:
            try:
                audio = self._get_audio()
                # logger.debug(f"Received audio: {audio}")
                self._store_recording(audio)
            except (EOFError, NoAudioAvailable):
                logger.info("No audio available")
                break
            except StopIteration:
                break
            except Exception as exc:
                logger.exception(f"Continuous listener received exception: {type(exc)}")
                break
        self._store_recording_stopped()

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
        if not self.is_listening and self._recordings.empty():
            raise NoAudioAvailable()

        blocking = not (wait is False or wait == 0)
        timeout = wait if isinstance(wait, (int, float)) and not isinstance(wait, bool) and wait > 0 else None
        try:
            result = self._recordings.get(blocking, timeout)
            if isinstance(result, _EndOfListener):
                raise NoAudioAvailable()
            return result
        except queue.Empty as exc:
            raise NoAudioAvailable() from exc

    def __iter__(self):
        for _ in itertools.count():
            try:
                frame = self.read()
                yield frame
            except NoAudioAvailable:
                return None
        return None

    @contextmanager
    def listen(self):
        """
        Context manager to start listening, then stop listening at the end of the ``with`` block.  It provides an
        object with ``read`` and ``empty`` methods, and the object is iterable.
        """
        self.start()
        try:
            yield self
        finally:
            self.stop()


class FrameStateEnum(Enum):
    LISTEN = "LISTEN"
    PAUSE = "PAUSE"
    STOP = "STOP"


@dataclass
class AnnotatedFrame:
    frame: AudioSample
    state: FrameStateEnum


class BaseStateListener(Listener):
    """
    Listen to an audio source, collecting samples until a certain state is met.  Once the state is met, combine the
    appropriate samples as a single sample.

    This is an abstract base class providing default implementations for much of the class's behavior.  The
    ``_determine_frame_state` method is the only method that must be implemented by any child classes.
    """

    def _determine_frame_state(self, latest_frame: AudioSample, all_frames: List[AnnotatedFrame]) -> FrameStateEnum:
        """
        Determine the state of the listener from the ``latest_frame`` that's recorded and ``all_frames`` that have been
        recorded (it does not include ``latest_frame``).
        """
        raise NotImplementedError  # pragma: no cover

    def _pre_process_individual_audio_sample(self, audio: AudioSample) -> AudioSample:
        """
        Apply any kind of filtering or modification of an audio sample before determining the frame state.  By default,
        it just returns the audio sample unchanged.
        """
        return audio

    def _listen_frames(self, n_frames: int) -> List[AnnotatedFrame]:
        """
        Record and collect audio frames until the ``STOP`` frame state has been reached.  Return a list of all frames
        recorded (including the ``STOP`` frame).

        If the end of the source is reached, then listening will end regardless of what ``_determine_frame_state`` will
        return.
        """
        all_frames = []  # type: List[AnnotatedFrame]

        frame = None  # type: Optional[AudioSample]
        frame = self.source.read(n_frames)
        frame = self._pre_process_individual_audio_sample(frame)
        frame_state = self._determine_frame_state(frame, all_frames)

        while frame_state != FrameStateEnum.STOP:
            all_frames.append(AnnotatedFrame(frame, frame_state))
            try:
                frame = self.source.read(n_frames)
            except EOFError:
                frame = None
                frame_state = FrameStateEnum.STOP
                break
            frame = self._pre_process_individual_audio_sample(frame)
            frame_state = self._determine_frame_state(frame, all_frames)

        if frame is not None:
            all_frames.append(AnnotatedFrame(frame, frame_state))
        return all_frames

    def _filter_audio_samples(self, all_frames: List[AnnotatedFrame]) -> List[AnnotatedFrame]:
        """
        Modify/remove audio samples before they are merged into a single audio sample.

        This method keeps all frames where the frame state was ``LISTEN`` or where the previous frame's state was
        ``LISTEN``.
        """
        saved_frames = [
            frame
            for frame, previous_frame in zip(
                all_frames, [AnnotatedFrame(AudioSample.generate_silence(0, 44100), FrameStateEnum.PAUSE)] + all_frames
            )
            if frame.state == FrameStateEnum.LISTEN or previous_frame.state == FrameStateEnum.LISTEN
        ]
        return saved_frames

    def _join_audio_samples(self, all_frames: List[AnnotatedFrame]) -> AudioSample:
        """
        Combine audio frames into a single audio sample.
        """
        sample = AudioSample.from_iterable(frame.frame for frame in all_frames)
        if sample is None:
            return AudioSample.generate_silence(0, self.source.frame_rate)
        return sample

    def _post_process_final_audio_sample(self, audio: AudioSample) -> AudioSample:
        """
        Apply any kind of filtering or modification of the final audio sample.  By default, it just returns the audio
        sample unchanged.
        """
        return audio

    def read(self, n_frames: int = 2**14) -> AudioSample:
        """
        Read and return an audio sample from the source.

        This may make multiple calls to the source's ``read`` method so that enough audio is collected to return.

        ``n_frames`` is the number of frames to read from the source on each call.
        """
        all_frames = self._listen_frames(n_frames)
        all_frames = self._filter_audio_samples(all_frames)
        audio_sample = self._join_audio_samples(all_frames)
        audio_sample = self._post_process_final_audio_sample(audio_sample)

        return audio_sample


class SilenceBasedListener(BaseStateListener):
    """
    Listen to the audio source and collect a sample that exceeds a silence threshold.

    Do not start recording until the sample's RMS exceeds the threshold given (``silence_threshold_rms``).  Once
    recording, continue recording until the new sample drops below the threshold.
    """

    def __init__(self, source: BaseAudioSource, silence_threshold_rms: int = 500):
        super().__init__(source)
        self.silence_threshold_rms = silence_threshold_rms

    def _determine_frame_state(self, latest_frame: AudioSample, all_frames: List[AnnotatedFrame]) -> FrameStateEnum:
        if latest_frame.rms > self.silence_threshold_rms:
            return FrameStateEnum.LISTEN
        if len(all_frames) == 0 or all_frames[-1].state == FrameStateEnum.PAUSE:
            # Haven't started listening yet
            return FrameStateEnum.PAUSE
        return FrameStateEnum.STOP


class TimeBasedListener(BaseStateListener):
    """
    Listen to the audio source and return an audio sample that is ``total_duration`` seconds long.
    """

    def __init__(self, source: BaseAudioSource, total_duration: TimeType):
        super().__init__(source)
        self.total_duration = total_duration

    def _determine_frame_state(self, latest_frame: AudioSample, all_frames: List[AnnotatedFrame]) -> FrameStateEnum:
        duration = sum(frame.frame.n_seconds for frame in all_frames) + latest_frame.n_seconds
        if len(latest_frame) == 0:
            return FrameStateEnum.PAUSE
        if duration <= self.total_duration or len(all_frames) == 0:
            return FrameStateEnum.LISTEN
        return FrameStateEnum.STOP

    def read(self, n_frames: int = 2**14) -> AudioSample:
        """
        Read samples from the source until the total duration is reached.  ``n_frames`` is the suggested size for each
        sample.  The number will be adjusted so that an integer number of samples is required to get the total duration.
        """
        total_duration_frames = self.total_duration * self.source.frame_rate
        n_recordings = round(total_duration_frames / n_frames)
        adjusted_n_frames = math.ceil(total_duration_frames / n_recordings)
        return super().read(adjusted_n_frames)
