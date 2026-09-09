from ..model import Result, Log
from .model import LoggerModel
import cv2
from collections import deque


class PoseTracker():
    def __init__(self, result: Result, start_ts: int, max_seq_size: int):
        self.result = result
        self.start_ts = start_ts
        self.focus_count = 0
        self.blur_count = 0
        self.pitches: deque = deque([], maxlen=max_seq_size)
        self.rolls: deque = deque([], maxlen=max_seq_size)
        self.yaws: deque = deque([], maxlen=max_seq_size)
        self.ages: deque = deque([], maxlen=max_seq_size)
        self.genders: deque = deque([], maxlen=max_seq_size)
        self.on_focus(result)

    def on_focus(self, result: Result):
        self.focus_count += 1
        self.blur_count = 0
        self.pitches.append(result.pitch)
        self.rolls.append(result.roll)
        self.yaws.append(result.yaw)
        if result.age is not None:
            self.ages.append(result.age)
        if result.gender is not None:
            self.genders.append(result.gender)

    def on_blur(self):
        self.blur_count += 1


class PoseLogger(LoggerModel):
    def __init__(self, min_focus_count: int, pitch_range: list[float], roll_range: list[float], yaw_range: list[float]):
        super().__init__()
        """
        min_focus_count must >= max_disappear
        """
        self.min_focus_count = min_focus_count
        self.max_blur_count = min_focus_count
        self.pitch_range = pitch_range
        self.roll_range = roll_range
        self.yaw_range = yaw_range
        self.pose_info: dict[str, PoseTracker] = {}

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]', ts: int):
        logs: list[Log] = []
        blur_set = set(self.pose_info.keys())
        for result in results:
            assert result.id is not None
            assert result.pitch is not None
            assert result.roll is not None
            assert result.yaw is not None
            is_focus_before = result.id in self.pose_info
            is_focus_now = result.disappear_count == 0
            is_focus_now &= result.pitch >= self.pitch_range[0]
            is_focus_now &= result.pitch <= self.pitch_range[1]
            is_focus_now &= result.roll >= self.roll_range[0]
            is_focus_now &= result.roll <= self.roll_range[1]
            is_focus_now &= result.yaw >= self.yaw_range[0]
            is_focus_now &= result.yaw <= self.yaw_range[1]

            if is_focus_before and is_focus_now:
                self.pose_info[result.id].on_focus(result)
                blur_set.remove(result.id)
            elif not is_focus_before and is_focus_now:
                self.pose_info[result.id] = PoseTracker(result, ts, self.MAX_SEQ_SIZE)
            elif is_focus_before and (result.is_lost or not is_focus_now):
                self.pose_info[result.id].on_blur()

        for blur_id in blur_set:
            if self.pose_info[blur_id].blur_count > self.max_blur_count:
                pose = self.pose_info.pop(blur_id)
                if pose.focus_count > self.min_focus_count:
                    logs.append(Log().set(
                        source=self.__class__.__name__,
                        id=blur_id,
                        start_date=pose.start_ts,
                        end_date=ts,
                        pitch=self.get_mean(pose.pitches),
                        roll=self.get_mean(pose.rolls),
                        yaw=self.get_mean(pose.yaws),
                        age=self.get_mean(pose.ages),
                        gender=self.get_mean(pose.genders, 0.5),
                    ))
        return logs
