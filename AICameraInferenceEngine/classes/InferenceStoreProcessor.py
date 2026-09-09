from typing import Any
from inference.rule_based.object_tracker import ObjectTracker
from inference.rule_based.region_checker import RegionChecker
from inference.rule_based.pose_checker import PoseChecker
from typing import TypeVar, Generic
from classes.InferenceStore import InferenceStore
from classes.InferenceConfig import inference_config
from loguru import logger
T = TypeVar('T')

# 處理 Inference Store 資料的 Processor


class StoreProcessor(Generic[T]):
    store: T

    def execute():
        pass

    def set_store(self, store: T):
        # if not isinstance(self, StoreProcessor[T]):
        #   raise Exception('Store is not support the processor')
        self.store = store


class TrackingStoreProcessor(StoreProcessor[InferenceStore]):
    def __init__(self) -> None:
        super().__init__()
        self.object_tracker = ObjectTracker(
            metric_threshold=inference_config.config['maxBBoxDistance'],
            max_disappear_count=inference_config.config['maxDisappearCount'],
        )

    def execute(self):
        self.store.clear_records()
        self.store.inference_result = self.object_tracker(self.store.inference_result)


class RegionStoreProcessor(StoreProcessor[InferenceStore]):
    def __init__(self) -> None:
        super().__init__()
        self.region_checker = RegionChecker(
            regions=inference_config.config['focusRegion'],
            min_exist_count=inference_config.config['maxDisappearCount']
        )

    def execute(self):
        self.store.records += self.region_checker(self.store.inference_result)


class PoseStoreProcessor(StoreProcessor[InferenceStore]):
    def __init__(self) -> None:
        super().__init__()
        self.load_post_checker(inference_config.config)
        # TODO: handle this event listener, need remove it when the factory is destoryed
        inference_config.on_config_loaded(self.load_post_checker)

    def load_post_checker(self, config: Any = None):
        self.pose_checker = PoseChecker(
            min_gaze_count=config['minGazeCount'],
            min_exist_count=config['maxDisappearCount'],
            pitch_range=[config['gazeToCameraRange']['pitchMin'], config['gazeToCameraRange']['pitchMax']],
            yaw_range=[config['gazeToCameraRange']['yawMin'], config['gazeToCameraRange']['yawMax']]
        )

    def execute(self):
        self.store.records += self.pose_checker(self.store.inference_result)
