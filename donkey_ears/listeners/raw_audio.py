import itertools
import math
from dataclasses import dataclass
from enum import Enum
from typing import List

from donkey_ears.audio.base import AudioSample, BaseAudioSource, TimeType


class NoAudioAvailable(Exception):
    pass


class BaseListener:
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


class FrameStateEnum(Enum):
    LISTEN = "LISTEN"
    PAUSE = "PAUSE"
    STOP = "STOP"


@dataclass
class AnnotatedFrame:
    frame: AudioSample
    state: FrameStateEnum


class BaseStateListener(BaseListener):
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
        all_frames = []

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
            for frame, previous_frame in zip(all_frames, [AnnotatedFrame(None, None)] + all_frames)
            if frame.state == FrameStateEnum.LISTEN or previous_frame.state == FrameStateEnum.LISTEN
        ]
        return saved_frames

    def _join_audio_samples(self, all_frames: List[AnnotatedFrame]) -> AudioSample:
        """
        Combine audio frames into a single audio sample.
        """
        return AudioSample.from_iterable(frame.frame for frame in all_frames)

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
