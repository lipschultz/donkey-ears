from unittest.mock import MagicMock

from donkey_ears.audio.base import AudioSample
from donkey_ears.listeners.state_based import (
    AnnotatedFrame,
    BaseStateListener,
    FrameStateEnum,
    SilenceBasedListener,
    TimeBasedListener,
)


class TestBaseStateListener:
    @staticmethod
    def test_listening_to_frames_when_stop_frame_eventually_countered():
        class StateListener(BaseStateListener):  # pylint: disable=too-few-public-methods
            def _determine_frame_state(self, latest_frame, all_frames):  # pylint: disable=unused-argument
                if len(all_frames) == 3:
                    return FrameStateEnum.STOP
                return FrameStateEnum.LISTEN

        audio_source = MagicMock()
        audio_source.read = MagicMock(return_value=AudioSample.generate_silence(1, 44100))
        subject = StateListener(audio_source)

        actual = subject._listen_frames(2**14)

        assert audio_source.read.call_count == 4
        assert actual == [
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.LISTEN),
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.LISTEN),
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.LISTEN),
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.STOP),
        ]

    @staticmethod
    def test_listening_to_frames_when_stop_frame_encountered_immediately():
        class StateListener(BaseStateListener):  # pylint: disable=too-few-public-methods
            def _determine_frame_state(self, latest_frame, all_frames):  # pylint: disable=unused-argument
                return FrameStateEnum.STOP

        audio_source = MagicMock()
        audio_source.read = MagicMock(return_value=AudioSample.generate_silence(1, 44100))
        subject = StateListener(audio_source)

        actual = subject._listen_frames(2**14)

        assert audio_source.read.call_count == 1
        assert actual == [AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.STOP)]

    @staticmethod
    def test_listening_to_frames_when_end_of_source_eventually_encountered():
        class StateListener(BaseStateListener):  # pylint: disable=too-few-public-methods
            def _determine_frame_state(self, latest_frame, all_frames):  # pylint: disable=unused-argument
                if len(all_frames) == 3:
                    return FrameStateEnum.STOP
                return FrameStateEnum.LISTEN

        audio_source = MagicMock()
        audio_source.read = MagicMock(
            side_effect=[AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(1, 44100), EOFError()]
        )
        subject = StateListener(audio_source)

        actual = subject._listen_frames(2**14)

        assert audio_source.read.call_count == 3
        assert actual == [
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.LISTEN),
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.LISTEN),
        ]

    @staticmethod
    def test_filtering_samples_with_no_trailing_stop():
        subject = BaseStateListener(MagicMock())
        input_frames = [
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.LISTEN),
            AnnotatedFrame(AudioSample.generate_silence(2, 44100), FrameStateEnum.LISTEN),
        ]

        actual = subject._filter_audio_samples(input_frames)

        assert actual == input_frames

    @staticmethod
    def test_filtering_samples_with_trailing_stop():
        subject = BaseStateListener(MagicMock())
        input_frames = [
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.LISTEN),
            AnnotatedFrame(AudioSample.generate_silence(2, 44100), FrameStateEnum.LISTEN),
            AnnotatedFrame(AudioSample.generate_silence(3, 44100), FrameStateEnum.STOP),
        ]

        actual = subject._filter_audio_samples(input_frames)

        assert actual == input_frames

    @staticmethod
    def test_filtering_empty_sample_list():
        subject = BaseStateListener(MagicMock())
        input_frames = []

        actual = subject._filter_audio_samples(input_frames)

        assert actual == input_frames

    @staticmethod
    def test_filtering_samples_with_single_pause_in_middle():
        subject = BaseStateListener(MagicMock())
        input_frames = [
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.LISTEN),
            AnnotatedFrame(AudioSample.generate_silence(2, 44100), FrameStateEnum.PAUSE),
            AnnotatedFrame(AudioSample.generate_silence(3, 44100), FrameStateEnum.LISTEN),
        ]

        actual = subject._filter_audio_samples(input_frames)

        assert actual == input_frames

    @staticmethod
    def test_filtering_samples_with_multiple_pauses_in_middle():
        subject = BaseStateListener(MagicMock())
        input_frames = [
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.LISTEN),
            AnnotatedFrame(AudioSample.generate_silence(2, 44100), FrameStateEnum.PAUSE),
            AnnotatedFrame(AudioSample.generate_silence(3, 44100), FrameStateEnum.PAUSE),
            AnnotatedFrame(AudioSample.generate_silence(4, 44100), FrameStateEnum.LISTEN),
        ]

        actual = subject._filter_audio_samples(input_frames)

        assert actual == [
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.LISTEN),
            AnnotatedFrame(AudioSample.generate_silence(2, 44100), FrameStateEnum.PAUSE),
            AnnotatedFrame(AudioSample.generate_silence(4, 44100), FrameStateEnum.LISTEN),
        ]

    @staticmethod
    def test_filtering_samples_with_stop_as_only_sample():
        subject = BaseStateListener(MagicMock())
        input_frames = [
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.STOP),
        ]

        actual = subject._filter_audio_samples(input_frames)

        assert actual == []

    @staticmethod
    def test_filtering_samples_with_pause_at_start():
        subject = BaseStateListener(MagicMock())
        input_frames = [
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.PAUSE),
            AnnotatedFrame(AudioSample.generate_silence(2, 44100), FrameStateEnum.LISTEN),
            AnnotatedFrame(AudioSample.generate_silence(3, 44100), FrameStateEnum.LISTEN),
        ]

        actual = subject._filter_audio_samples(input_frames)

        assert actual == [
            AnnotatedFrame(AudioSample.generate_silence(2, 44100), FrameStateEnum.LISTEN),
            AnnotatedFrame(AudioSample.generate_silence(3, 44100), FrameStateEnum.LISTEN),
        ]

    @staticmethod
    def test_filtering_samples_with_multiple_pauses_at_start():
        subject = BaseStateListener(MagicMock())
        input_frames = [
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.PAUSE),
            AnnotatedFrame(AudioSample.generate_silence(2, 44100), FrameStateEnum.PAUSE),
            AnnotatedFrame(AudioSample.generate_silence(3, 44100), FrameStateEnum.LISTEN),
            AnnotatedFrame(AudioSample.generate_silence(4, 44100), FrameStateEnum.LISTEN),
        ]

        actual = subject._filter_audio_samples(input_frames)

        assert actual == [
            AnnotatedFrame(AudioSample.generate_silence(3, 44100), FrameStateEnum.LISTEN),
            AnnotatedFrame(AudioSample.generate_silence(4, 44100), FrameStateEnum.LISTEN),
        ]

    @staticmethod
    def test_joining_audio_frames():
        frames = [
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.LISTEN),
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.LISTEN),
        ]
        subject = BaseStateListener(MagicMock())

        actual = subject._join_audio_samples(frames)

        assert actual == AudioSample.generate_silence(2, 44100)

    @staticmethod
    def test_joining_audio_frames_when_no_frames_exist():
        frames = []
        subject = BaseStateListener(MagicMock())

        actual = subject._join_audio_samples(frames)

        assert actual is None

    @staticmethod
    def test_read_calls_other_methods():
        frames_heard = [
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.LISTEN),
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.PAUSE),
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.LISTEN),
            AnnotatedFrame(AudioSample.generate_silence(1, 44100), FrameStateEnum.STOP),
        ]
        filtered_frames_heard = [frame for frame in frames_heard if frame.state != FrameStateEnum.PAUSE]
        joined_sample = AudioSample.generate_silence(2, 44100)
        post_processed_sample = AudioSample.generate_silence(1.5, 44100)
        subject = BaseStateListener(MagicMock())
        subject._listen_frames = MagicMock(return_value=frames_heard)
        subject._filter_audio_samples = MagicMock(return_value=filtered_frames_heard)
        subject._join_audio_samples = MagicMock(return_value=joined_sample)
        subject._post_process_final_audio_sample = MagicMock(return_value=post_processed_sample)

        actual = subject.read(44100)

        subject._listen_frames.assert_called_once_with(44100)
        subject._filter_audio_samples.assert_called_once_with(frames_heard)
        subject._join_audio_samples.assert_called_once_with(filtered_frames_heard)
        subject._post_process_final_audio_sample.assert_called_once_with(joined_sample)
        assert actual == post_processed_sample


class TestSilenceBasedListener:
    @staticmethod
    def test_frame_above_threshold_is_labeled_listen():
        subject = SilenceBasedListener(MagicMock(), 10)
        latest_frame = MagicMock()
        latest_frame.rms = 200

        actual = subject._determine_frame_state(latest_frame, [])

        assert actual == FrameStateEnum.LISTEN

    @staticmethod
    def test_frame_below_threshold_with_no_previous_frames_is_labeled_pause():
        subject = SilenceBasedListener(MagicMock(), 10)
        latest_frame = MagicMock()
        latest_frame.rms = 0

        actual = subject._determine_frame_state(latest_frame, [])

        assert actual == FrameStateEnum.PAUSE

    @staticmethod
    def test_frame_below_threshold_with_previous_frame_labeled_pause_is_labeled_pause():
        subject = SilenceBasedListener(MagicMock(), 10)
        latest_frame = MagicMock()
        latest_frame.rms = 0

        actual = subject._determine_frame_state(latest_frame, [AnnotatedFrame(MagicMock(), FrameStateEnum.PAUSE)])

        assert actual == FrameStateEnum.PAUSE

    @staticmethod
    def test_frame_below_threshold_with_previous_frame_labeled_listen_is_labeled_stop():
        subject = SilenceBasedListener(MagicMock(), 10)
        latest_frame = MagicMock()
        latest_frame.rms = 0

        actual = subject._determine_frame_state(latest_frame, [AnnotatedFrame(MagicMock(), FrameStateEnum.LISTEN)])

        assert actual == FrameStateEnum.STOP


class TestTimeBasedListener:
    @staticmethod
    def test_first_frame_labeled_listen_if_shorter_than_total_duration():
        subject = TimeBasedListener(MagicMock(), 10)
        latest_frame = MagicMock()
        latest_frame.n_seconds = 1
        latest_frame.__len__ = MagicMock(return_value=44100)

        actual = subject._determine_frame_state(latest_frame, [])

        assert actual == FrameStateEnum.LISTEN

    @staticmethod
    def test_first_frame_labeled_listen_if_equal_to_total_duration():
        subject = TimeBasedListener(MagicMock(), 1)
        latest_frame = MagicMock()
        latest_frame.n_seconds = 1
        latest_frame.__len__ = MagicMock(return_value=44100)

        actual = subject._determine_frame_state(latest_frame, [])

        assert actual == FrameStateEnum.LISTEN

    @staticmethod
    def test_first_frame_labeled_listen_if_longer_than_total_duration():
        subject = TimeBasedListener(MagicMock(), 1)
        latest_frame = MagicMock()
        latest_frame.n_seconds = 2
        latest_frame.__len__ = MagicMock(return_value=2 * 44100)

        actual = subject._determine_frame_state(latest_frame, [])

        assert actual == FrameStateEnum.LISTEN

    @staticmethod
    def test_first_frame_labeled_pause_if_frame_has_no_duration():
        subject = TimeBasedListener(MagicMock(), 1)
        latest_frame = MagicMock()
        latest_frame.n_seconds = 0
        latest_frame.__len__ = MagicMock(return_value=0)

        actual = subject._determine_frame_state(latest_frame, [])

        assert actual == FrameStateEnum.PAUSE

    @staticmethod
    def test_frame_labeled_pause_if_frame_has_no_duration():
        subject = TimeBasedListener(MagicMock(), 10)
        latest_frame = MagicMock()
        latest_frame.n_seconds = 0
        latest_frame.__len__ = MagicMock(return_value=0)
        existing_frame = MagicMock()
        existing_frame.n_seconds = 1

        actual = subject._determine_frame_state(latest_frame, [AnnotatedFrame(existing_frame, FrameStateEnum.LISTEN)])

        assert actual == FrameStateEnum.PAUSE

    @staticmethod
    def test_frame_labeled_stop_if_frame_pushes_duration_over_total():
        subject = TimeBasedListener(MagicMock(), 10)
        latest_frame = MagicMock()
        latest_frame.n_seconds = 1
        latest_frame.__len__ = MagicMock(return_value=44100)
        existing_frame = MagicMock()
        existing_frame.n_seconds = 10

        actual = subject._determine_frame_state(latest_frame, [AnnotatedFrame(existing_frame, FrameStateEnum.LISTEN)])

        assert actual == FrameStateEnum.STOP
