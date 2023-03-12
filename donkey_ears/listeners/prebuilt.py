from typing import List

from donkey_ears.audio.base import AudioSample, BaseAudioSource, TimeType
from donkey_ears.listeners.base import AnnotatedFrame, BaseStateContinuousListener, FrameStateEnum


class TimeBasedListener(BaseStateContinuousListener):
    def __init__(self, source: BaseAudioSource, total_duration: TimeType):
        super().__init__(source)
        self.total_duration = total_duration

    def _determine_frame_state(self, latest_frame: AudioSample, all_frames: List[AnnotatedFrame]) -> FrameStateEnum:
        duration = sum(frame.frame.n_seconds for frame in all_frames) + latest_frame.n_seconds
        if duration < self.total_duration and len(latest_frame) > 0:
            return FrameStateEnum.LISTEN
        return FrameStateEnum.STOP


class SilenceBasedListener(BaseStateContinuousListener):
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
