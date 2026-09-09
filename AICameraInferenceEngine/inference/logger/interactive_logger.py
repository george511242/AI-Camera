from ..model import Result, Log
from .model import LoggerModel
import cv2


class InteractiveAgeGenderGroup():
    def __init__(
        self,
        group_code: str,
        group_dest: str,
        age_range: list[float],
        gender: int,
    ):
        self.group_code = group_code
        self.group_dest = group_dest
        self.age_range = age_range
        self.gender = gender


class InteractivePoseGroup():
    def __init__(
        self,
        group_code: str,
        group_dest: str,
        pitch_range: list[float],
        roll_range: list[float],
        yaw_range: list[float],
    ):
        self.group_code = group_code
        self.group_dest = group_dest
        self.pitch_range = pitch_range
        self.roll_range = roll_range
        self.yaw_range = yaw_range


class InteractiveTracker():
    def __init__(self, enter_ts: int, id: str, age_gender_group_idx: int | None, pose_group_idx: int | None):
        self.enter_ts = enter_ts
        self.id = id
        self.age_gender_group_idx = age_gender_group_idx
        self.pose_group_idx = pose_group_idx
        self.first_notify_sent = False

    def __eq__(self, value: tuple[str, int, int]):
        return all([s == o for s, o in zip([
            self.id,
            self.age_gender_group_idx,
            self.pose_group_idx,
        ], value)])


class InteractiveLogger(LoggerModel):
    def __init__(
        self,
        notify_age_gender_group: list[InteractiveAgeGenderGroup] = [],
        notify_pose_group: list[InteractivePoseGroup] = [],
        notify_in_region_sec: int | None = None,
        notify_in_region_count: int | None = None,
    ) -> None:
        """
        notify_in_region_sec, notify_in_region_count should choose one to use\n
        if use notify_in_region_count, must > max_disappear
        """
        assert notify_in_region_sec is not None and notify_in_region_count is not None, ValueError('notify_in_region_sec or notify_in_region_count should be set')
        self.notify_age_gender_group = notify_age_gender_group
        self.notify_pose_group = notify_pose_group
        self.notify_threshold: int = notify_in_region_sec or notify_in_region_count
        self.interactive_tracker: InteractiveTracker | None = None
        # notify data
        self.first_notify_sent = False

        assert self.notify_threshold is int

    def find_age_gender_group_idx(self, age: float | None, gender: int | float | None):
        if age is None or gender is None:
            return None
        return next((
            idx for idx, group in enumerate(self.notify_age_gender_group)
            if group.age_range[0] <= age <= group.age_range[1] and group.gender == gender
        ), None)

    def find_pose_group_idx(self, pitch: float, roll: float, yaw: float):
        return next((
            idx for idx, group in enumerate(self.notify_pose_group)
            if group.pitch_range[0] <= pitch <= group.pitch_range[1]
            and group.roll_range[0] <= roll <= group.roll_range[1]
            and group.yaw_range[0] <= yaw <= group.yaw_range[1]
        ), None)

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]', ts: int):
        logs: list[Log] = []
        has_result = bool(len(results))
        has_tracker = self.interactive_tracker is not None

        if has_result:
            result = results[0]
            age = self.get_mean(result.get_attr_from_history("age"))
            gender = self.get_mean(result.get_attr_from_history("gender"), 0.5)

            age_gender_group_idx = self.find_age_gender_group_idx(age, gender)
            pose_group_idx = self.find_pose_group_idx(result.pitch, result.roll, result.yaw)

            if age_gender_group_idx is None and pose_group_idx is None:
                return logs
            assert age_gender_group_idx is not None or pose_group_idx is not None

            if has_tracker and self.interactive_tracker == (result.id, age_gender_group_idx, pose_group_idx):
                assert self.interactive_tracker is not None
                duration = ts - self.interactive_tracker.enter_ts
                is_reach_threshold = duration >= self.notify_threshold
                if is_reach_threshold and not self.interactive_tracker.first_notify_sent:
                    self.interactive_tracker.first_notify_sent = True
                    if age_gender_group_idx is not None:
                        group = self.notify_age_gender_group[age_gender_group_idx]
                    elif pose_group_idx is not None:
                        group = self.notify_pose_group[pose_group_idx]
                    return [
                        Log().set(
                            source=self.__class__.__name__,
                            id=self.interactive_tracker.id,
                            enter_date=self.interactive_tracker.enter_ts,
                            group_code=group.group_code,
                            group_dest=group.group_dest,
                        )
                    ]
            else:
                self.interactive_tracker = InteractiveTracker(ts, result.id, age_gender_group_idx, pose_group_idx)
        elif not has_result and has_tracker:
            self.interactive_tracker = None
        return logs
