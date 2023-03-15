# pylint: disable=protected-access

from unittest.mock import MagicMock

import pytest

from donkey_ears.audio.base import AudioSample
from donkey_ears.listeners.audio import (
    AnnotatedFrame,
    BaseStateListener,
    ContinuousListener,
    FrameStateEnum,
    Listener,
    ListenerRunningError,
    NoAudioAvailable,
    SilenceBasedListener,
    TimeBasedListener,
)


class TestListener:
    @staticmethod
    def test_read_sample():
        audio_source = MagicMock()
        audio_source.read = MagicMock(return_value=AudioSample.generate_silence(1, 44100))
        subject = Listener(audio_source)

        actual = subject.read()

        audio_source.read.assert_called_once()
        assert actual == AudioSample.generate_silence(1, 44100)

    @staticmethod
    def test_chunk_size_passed_to_source_read_method():
        audio_source = MagicMock()
        audio_source.read = MagicMock(return_value=AudioSample.generate_silence(1, 44100))
        subject = Listener(audio_source)

        subject.read(22050)

        audio_source.read.assert_called_once_with(22050)

    @staticmethod
    def test_no_audio_available():
        audio_source = MagicMock()
        audio_source.read = MagicMock(side_effect=EOFError)
        subject = Listener(audio_source)

        with pytest.raises(NoAudioAvailable):
            subject.read()

    @staticmethod
    def test_iterating_over_listener_returns_samples_recorded():
        audio_source = MagicMock()
        audio_source.read = MagicMock(
            side_effect=[AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100), EOFError]
        )
        subject = Listener(audio_source)

        results = list(subject)

        assert results == [AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100)]

    @staticmethod
    def test_getting_continuous_listener():
        audio_source = MagicMock()
        audio_source.read = MagicMock(
            side_effect=[AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100), EOFError]
        )
        listener = Listener(audio_source)

        subject = listener.continuous_listener()

        assert isinstance(subject, ContinuousListener)
        assert subject.listener is listener
        assert not subject.is_listening
        with pytest.raises(NoAudioAvailable):
            subject.read()

    @staticmethod
    def test_getting_continuous_listener_as_part_of_context_manager():
        audio_source = MagicMock()
        audio_source.read = MagicMock(
            side_effect=[AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100), EOFError]
        )
        subject = Listener(audio_source)

        with subject.continuous_listener() as clistener:
            assert isinstance(clistener, ContinuousListener)
            assert clistener.listener is subject
            assert list(clistener) == [AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100)]

        assert not clistener.is_listening


class TestContinuousListener:
    @staticmethod
    def test_is_listening_returns_false_after_creation():
        subject = ContinuousListener(MagicMock())

        assert subject.is_listening is False

    @staticmethod
    def test_is_listening_returns_true_after_starting():
        subject = ContinuousListener(MagicMock())
        subject.start()

        assert subject.is_listening is True

    @staticmethod
    def test_is_listening_returns_false_after_starting_then_stopping():
        subject = ContinuousListener(MagicMock())
        subject.start()
        subject.stop()

        assert subject.is_listening is False

    @staticmethod
    def test_double_starting_listener_raises_exception():
        subject = ContinuousListener(MagicMock())
        subject.start()
        with pytest.raises(ListenerRunningError):
            subject.start()

    @staticmethod
    def test_source_read_called_when_listener_started():
        listener = MagicMock()
        listener.read = MagicMock()
        subject = ContinuousListener(listener)
        subject.start()
        subject.stop()

        listener.read.assert_called()

    @staticmethod
    def test_read():
        listener = MagicMock()
        listener.read = MagicMock(side_effect=[AudioSample.generate_silence(1, 44100), EOFError])
        subject = ContinuousListener(listener)
        subject.start()
        subject.stop()

        result = subject.read()

        assert result == AudioSample.generate_silence(1, 44100)
        with pytest.raises(NoAudioAvailable):
            subject.read()

    @staticmethod
    def test_read_while_started():
        listener = MagicMock()
        listener.read = MagicMock(side_effect=[AudioSample.generate_silence(1, 44100), EOFError])
        subject = ContinuousListener(listener)
        subject.start()

        result = subject.read()
        subject.stop()

        assert result == AudioSample.generate_silence(1, 44100)
        with pytest.raises(NoAudioAvailable):
            subject.read()

    @staticmethod
    def test_read_raises_exception_when_listener_immediately_raises_eoferror():
        listener = MagicMock()
        listener.read = MagicMock(side_effect=[EOFError])
        subject = ContinuousListener(listener)
        subject.start()
        subject.stop()

        with pytest.raises(NoAudioAvailable):
            subject.read()

    @staticmethod
    def test_read_raises_exception_when_not_listening_and_queue_empty():
        subject = ContinuousListener(MagicMock())

        with pytest.raises(NoAudioAvailable):
            subject.read()

    @staticmethod
    @pytest.mark.parametrize(
        "wait, expected_blocking, expected_timeout",
        [
            (False, False, None),
            (0, False, None),
            (0.5, True, 0.5),
            (1, True, 1),
            (True, True, None),
        ],
    )
    def test_read_passes_arguments_to_queue_get(wait, expected_blocking, expected_timeout):
        listener = MagicMock()
        listener.read = MagicMock(side_effect=[AudioSample.generate_silence(1, 44100), EOFError])
        subject = ContinuousListener(listener)
        subject._recordings = MagicMock()  # pylint: disable=protected-access
        subject._recordings.get = MagicMock()  # pylint: disable=protected-access
        subject._recordings.empty = MagicMock(return_value=False)  # pylint: disable=protected-access
        subject.start()
        subject.stop()

        subject.read(wait)

        subject._recordings.get.assert_called_once()  # pylint: disable=protected-access
        subject._recordings.get.assert_called_once_with(  # pylint: disable=protected-access
            expected_blocking, expected_timeout
        )

    @staticmethod
    def test_iterating_over_listener_returns_samples_recorded():
        listener = MagicMock()
        listener.read = MagicMock(
            side_effect=[AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100), EOFError]
        )
        subject = ContinuousListener(listener)
        subject.start()
        subject.stop()

        results = list(subject)

        assert results == [AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100)]

    @staticmethod
    def test_context_manager():
        class SubjectContinuousListen(ContinuousListener):
            def __init__(self, listen):
                super().__init__(listen)
                self.start_call_count = 0
                self.stop_call_count = 0

            def start(self):
                self.start_call_count += 1
                super().start()

            def stop(self, timeout=None):
                self.stop_call_count += 1
                super().stop(timeout)

        listener = MagicMock()
        listener.read = MagicMock(
            side_effect=[AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100), EOFError]
        )
        subject = SubjectContinuousListen(listener)

        with subject.listen() as continuous_listen:
            results = list(continuous_listen)

        assert results == [AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100)]
        assert subject.start_call_count == 1
        assert subject.stop_call_count == 1

    @staticmethod
    def test_using_in_context_manager():
        audio_source = MagicMock()
        audio_source.read = MagicMock(
            side_effect=[AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100), EOFError]
        )
        listener = Listener(audio_source)
        subject = ContinuousListener(listener)

        with subject.listen() as clistener:
            assert clistener is subject
            assert clistener.listener is listener
            assert list(clistener) == [AudioSample.generate_silence(1, 44100), AudioSample.generate_silence(2, 44100)]

        assert not clistener.is_listening


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
