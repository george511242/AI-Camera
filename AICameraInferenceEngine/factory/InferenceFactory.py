import cv2
from classes.Inference import Inference
from loguru import logger
from dataModel.modelOutput import InferenceMode, S4MRecordSchema, S4MInteractiveSchema, PeopleCountSchema
from classes.InferenceStore import InferenceStore
from classes.InferenceConfig import inference_config
from settings import settings
from factory.AgeModelPath import resolve_age_model
from factory.GenderModelPath import resolve_gender_model_path
from factory.HeadPoseModelPath import resolve_head_pose_model_path
from factory.YuNetModelPath import resolve_yunet_model_path
from inference.model import Workflow, Log
from inference.algorithm import (
    Sort,
    MaxIouFilter,
    RegionsFilter,
    HeadSnapshotTaker,
    HandsSnapshotTaker,
    Debugger,
)
from inference.logger import (
    RegionLogger,
    PoseLogger,
    InteractiveLogger,
    InteractiveAgeGenderGroup,
    InteractivePoseGroup,
    PhoneScrollingLogger,
    IdCountLogger,
)
from inference.utils import draw_utils
from inference.rknn.yolo11 import Yolo11
from inference.rknn.mobilenet import MobileNet
from inference.rknn.yolov8_pose import YoloV8Pose
from inference.rknn.yunet import YuNet
from inference.rknn.ssrnet import SSRNet
from inference.rknn.lightweight_head_pose_estimation import LightweightHeadPoseEstimation
from inference.rknn.mobilenetv3_age import MobileNetV3Age
model_weight_path = {
    'yolo11': '/model/rknn/yolo11n.rknn',
    'yolov8_pose': '/model/rknn/yolov8_pose.rknn',
    'mobilenet': '/model/rknn/mobilenet_v2.rknn',
}



def fps_calculate(frame, inferencer: Inference):
    return cv2.putText(frame, f"FPS: {inferencer.fps:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)


def draw_canny(frame, inferencer: Inference):
    return cv2.putText(inferencer.results[0].canny, f'Count: {inferencer.results[0].edges_count}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)


def draw_people_count(frame, inferencer: Inference):
    count = len([result for result in inferencer.results if not result.disappear_count])
    pos_x = frame.shape[1] - 240
    return cv2.putText(frame, f'Count: {count}', (pos_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)


def draw_region_wrapper(frame, inferencer: Inference):
    drawed = frame.copy()
    for region in inference_config.config['focusRegion']:
        drawed = draw_utils.draw_region(drawed, region['name'], region['bbox'])
    return drawed


def draw_result(frame, inferencer: Inference):
    drawed = frame.copy()
    for result in inferencer.results:
        drawed = draw_utils.draw_result(
            drawed,
            result,
            show_score=False,
            show_crop_info=True,
            show_imagenet_class=True,
            show_coco_class=True,
            show_id=True,
            show_tracking_info=False,
            show_age_gender=False,
            show_coco_kps=True,
            show_pose=True,
            show_pose_info=False,
            show_landmarks=False,
            show_focus_count=True,
            show_debug=True,
            ignore_disappear=True,
            text_color=(255, 0, 0),
        )
    return drawed


class InferenceFactory():
    def load(self, mode: InferenceMode):
        logger.info(f'Loading inference for mode: {mode}')
        logger.info(f'Current config: {inference_config.to_dict()}')
        if mode == InferenceMode.LeaveZone.value:
            return self.face_detection_handler()
        if mode == InferenceMode.LeaveZoneWithAgeGender.value:
            return self.face_with_age_gender_handler()
        if mode == InferenceMode.EyeTracking.value:
            return self.eye_tracking_handler()
        if mode == InferenceMode.LeaveZoneWithFacialFeatures.value:
            return self.age_gender_eye_tracking_handler()
        if mode == InferenceMode.PhonePoseDetection.value:
            return self.phone_post_detection_handler()
        return self.face_detection_handler()

    def _load_yolo_11_model(self):
        logger.info('Loading YOLO11 model')
        params = inference_config.config["custom"].get("yolo11", dict(
            score_threshold=.25,
            nms_thresh=.45,
            phone_score_threshold=0.005,
        ))
        params["cls_score_threshold"] = {67: params.pop("phone_score_threshold", 0.005)}
        return Yolo11(model_weight_path['yolo11'], **params)

    def _load_yolo_v8_pose_model(self):
        logger.info('Loading YOLOv8_pose model')
        params = inference_config.config["custom"].get("yolo_v8_pose", dict(
            score_threshold=.5,
            kp_threshold=.25,
        ))
        return YoloV8Pose(model_weight_path['yolov8_pose'], **params)

    def _load_face_model(self):
        platform = settings.rknpu_platform.strip().lower()
        model_path = resolve_yunet_model_path(platform)
        logger.info(f'Selected RKNPU platform: {platform}')
        logger.info(f'Selected YuNet model path: {model_path}')
        return YuNet(model_path)

    def _load_age_model(self):
        platform = settings.rknpu_platform.strip().lower()
        version = settings.age_model_version.strip().lower()
        model = resolve_age_model(version=version, platform=platform)
        logger.info(f'Selected RKNPU platform: {platform}')
        logger.info(f'Selected Age model version: {version}')
        logger.info(f'Selected Age model path: {model["path"]}')
        if model["implementation"] == "ssrnet":
            return SSRNet(model["path"], SSRNet.SSRNET_TARGET.AGE, margin=.45)
        if model["implementation"] == "mobilenetv3":
            return MobileNetV3Age(model["path"], margin=.45)
        raise ValueError(
            f'Unsupported Age model implementation: {model["implementation"]}'
        )

    def _load_gender_model(self):
        platform = settings.rknpu_platform.strip().lower()
        model_path = resolve_gender_model_path(platform)
        logger.info(f'Selected RKNPU platform: {platform}')
        logger.info(f'Selected Gender model path: {model_path}')
        return SSRNet(model_path, SSRNet.SSRNET_TARGET.GENDER, margin=.45)

    def _load_head_pose_model(self):
        platform = settings.rknpu_platform.strip().lower()
        model_path = resolve_head_pose_model_path(platform)
        logger.info(f'Selected RKNPU platform: {platform}')
        logger.info(f'Selected Head Pose model path: {model_path}')
        return LightweightHeadPoseEstimation(model_path, margin=.6)

    def _load_mobilenet_model(self):
        logger.info('Loading MobileNet model')
        params = inference_config.config["custom"].get("mobilenet", dict(
            focus_cls_idx_list=[487, 528, 707, 620],
        ))
        return MobileNet(model_weight_path['mobilenet'], **params)

    def _load_object_tracking_model(self, creator_filter=None):
        logger.info('Loading SORT model')
        return Sort(
            max_disappear=inference_config.config['maxDisappearCount'],
            max_distance=inference_config.config['maxBBoxDistance'],
            creator_filter=creator_filter,
        )

    def _load_max_iou_filter(self, region: list[float]):
        return MaxIouFilter(region)

    def _load_regions_filter(self, regions: list[list[float]]):
        return RegionsFilter(regions)

    def _load_phone_scrolling_logger(self):
        logger.info('Loading phone scrolling logger')
        params = inference_config.config["custom"].get("phone_scrolling_logger", dict(
            scrolling_queue_size=10,
            min_scrolling_count=3,
            head_down_ratio=0.5,
            wrist_to_shoulder_upper=-0.05,
            wrist_to_shoulder_lower=0.6,
            elbow_angle=80,
            bbox_move_threshold=0.1,
            bbox_size_change_threshold=0.1,
            ignore_count_after_send=60 * 60 * 4,  # seconds * minutes * avg FPS
        ))
        return PhoneScrollingLogger(**params)

    def _load_id_count_logger(self):
        logger.info('Loading id count logger')
        params = inference_config.config["custom"].get("id_count_logger", dict(
            sync_midnight=True,
        ))
        return IdCountLogger(**params)

    def _load_region_logger(self, name: str, region: list[float]):
        logger.info('Loading region logger')
        return RegionLogger(name=name, region=region)

    def _load_pose_logger(
        self,
        min_focus_count: int,
        pitch_range: list[float] = [-180, 180],
        roll_range: list[float] = [-180, 180],
        yaw_range: list[float] = [-180, 180],
    ):
        logger.info('Loading pose logger')
        return PoseLogger(
            min_focus_count=min_focus_count,
            pitch_range=pitch_range,
            roll_range=roll_range,
            yaw_range=yaw_range,
        )

    def _load_age_gender_interactive_logger(self):
        logger.info('Loading age gender interactive logger')
        notify_age_gender_group = [
            InteractiveAgeGenderGroup(
                group['rangeCode'],
                f"{group['serverIP']}:{group['serverPort']}",
                [group['ageMin'], group['ageMax']],
                group['gender']
            )
            for group in inference_config.config['interactiveAgeGenderRanges']
        ]
        return InteractiveLogger(
            notify_in_region_sec=inference_config.config['interactiveTriggerTime'],
            notify_age_gender_group=notify_age_gender_group,
        )

    def _load_pose_interactive_logger(self):
        logger.info('Loading pose interactive logger')
        notify_pose_group = [
            InteractivePoseGroup(
                group['rangeCode'],
                f"{group['serverIP']}:{group['serverPort']}",
                [group['pitchMin'], group['pitchMax']],
                [group['rollMin'], group['rollMax']],
                [group['yawMin'], group['yawMax']]
            )
            for group in inference_config.config['interactiveAgeGenderRanges']
        ]
        return InteractiveLogger(
            notify_in_region_sec=inference_config.config['interactiveTriggerTime'],
            notify_pose_group=notify_pose_group,
        )

    def face_detection_handler(self) -> Inference[InferenceStore]:
        def formatter(logs: list[Log]):
            return list(map(lambda log: S4MRecordSchema(mode=InferenceMode.LeaveZone.value, **log.to_dict()).model_dump(), logs))
        return Inference[InferenceStore](
            name=InferenceMode.LeaveZone.value,
            workflow=Workflow(
                models=[
                    # Canny(9, 100, 200),
                    self._load_face_model(),
                    self._load_object_tracking_model(),
                ],
                loggers=[
                    self._load_region_logger(name=region['name'], region=region['bbox'])
                    for region in inference_config.config['focusRegion']
                ],
            ),
            # interceptor after inference
            post_interceptor=[

            ],
            draw_functions=[
                # draw_region_wrapper,
                # draw_result,
                # draw_people_count,
                # fps_calculate
            ],
            output_formatter=formatter
        )

    def face_with_age_gender_handler(self) -> Inference[InferenceStore]:
        def formatter(logs: list[Log]):
            return list(map(lambda log: S4MRecordSchema(mode=InferenceMode.LeaveZoneWithAgeGender.value, **log.to_dict()).model_dump(), logs))

        return Inference[InferenceStore](
            name=InferenceMode.LeaveZoneWithAgeGender.value,
            workflow=Workflow(
                models=[
                    self._load_face_model(),
                    self._load_age_model(),
                    self._load_gender_model(),
                    self._load_object_tracking_model(),
                ],
                loggers=[
                    self._load_region_logger(name=region['name'], region=region['bbox'])
                    for region in inference_config.config['focusRegion']
                ],
            ),
            # interceptor after inference
            post_interceptor=[
                # detect_age_gender_abnormal_data
            ],
            draw_functions=[
                # draw_region_wrapper,
                # draw_result,
                # fps_calculate
            ],
            output_formatter=formatter
        )

    def eye_tracking_handler(self) -> Inference[InferenceStore]:
        def formatter(logs: list[Log]):
            return list(map(lambda log: S4MRecordSchema(mode=InferenceMode.EyeTracking.value, **log.to_dict()).model_dump(), logs))

        return Inference[InferenceStore](
            name=InferenceMode.EyeTracking.value,
            workflow=Workflow(
                models=[
                    self._load_face_model(),
                    self._load_head_pose_model(),
                    self._load_object_tracking_model(),
                ],
                loggers=[
                    *[
                        self._load_region_logger(name=region['name'], region=region['bbox'])
                        for region in inference_config.config['focusRegion']
                    ],
                    self._load_pose_logger(
                        min_focus_count=inference_config.config['minGazeCount'],
                        pitch_range=[inference_config.config['gazeToCameraRange']['pitchMin'], inference_config.config['gazeToCameraRange']['pitchMax']],
                        yaw_range=[inference_config.config['gazeToCameraRange']['yawMin'], inference_config.config['gazeToCameraRange']['yawMax']],
                    ),
                ],
            ),
            # interceptor after inference
            post_interceptor=[
                # detect_eye_tracking_abnormal_data
            ],
            draw_functions=[
                # draw_region_wrapper,
                # draw_result,
                # fps_calculate
            ],
            output_formatter=formatter
        )

    def age_gender_eye_tracking_handler(self) -> Inference[InferenceStore]:
        def formatter(logs: list[Log]):
            return list(map(lambda log: S4MRecordSchema(mode=InferenceMode.LeaveZoneWithFacialFeatures.value, **log.to_dict()).model_dump(), logs))

        return Inference[InferenceStore](
            name=InferenceMode.LeaveZoneWithFacialFeatures.value,
            workflow=Workflow(
                models=[
                    self._load_face_model(),
                    self._load_age_model(),
                    self._load_gender_model(),
                    self._load_head_pose_model(),
                    self._load_object_tracking_model(),
                ],
                loggers=[
                    *[
                        self._load_region_logger(name=region['name'], region=region['bbox'])
                        for region in inference_config.config['focusRegion']
                    ],
                    self._load_pose_logger(
                        min_focus_count=inference_config.config['minGazeCount'],
                        pitch_range=[inference_config.config['gazeToCameraRange']['pitchMin'], inference_config.config['gazeToCameraRange']['pitchMax']],
                        yaw_range=[inference_config.config['gazeToCameraRange']['yawMin'], inference_config.config['gazeToCameraRange']['yawMax']],
                    ),
                ],
            ),
            # interceptor after inference
            post_interceptor=[
                # detect_eye_tracking_abnormal_data
            ],
            draw_functions=[
                # draw_region_wrapper,
                # draw_result,
                # fps_calculate
            ],
            output_formatter=formatter
        )

    def phone_post_detection_handler(self) -> Inference[InferenceStore]:
        def formatter(logs: list[Log]):
            submit_logs = []
            for log in logs:
                if log.source == 'IdCountLogger':
                    submit_logs.append(PeopleCountSchema(mode=InferenceMode.PeopleCounting.value, **log.to_dict()).model_dump())
                else:
                    submit_logs.append(S4MRecordSchema(mode=InferenceMode.PhonePoseDetection.value, **log.to_dict()).model_dump())
            return submit_logs

        return Inference[InferenceStore](
            name=InferenceMode.PhonePoseDetection.value,
            workflow=Workflow(
                models=[
                    [
                        self._load_yolo_v8_pose_model(),
                        self._load_yolo_11_model(),
                    ],
                    Debugger(),
                    self._load_object_tracking_model("YoloV8Pose"),

                    # HeadSnapshotTaker(),
                    # self._load_head_pose_model(),
                    # self._load_face_model(),

                    # HandsSnapshotTaker(),
                    # self._load_mobilenet_model(),
                ],
                loggers=[
                    self._load_phone_scrolling_logger(),
                    self._load_id_count_logger(),
                ],
            ),
            # interceptor after inference
            post_interceptor=[

            ],
            draw_functions=[
                # draw_region_wrapper,
                # draw_result,
                # draw_people_count,
                # fps_calculate
            ],
            output_formatter=formatter
        )

    def interactive_age_gender_handler(self) -> Inference[InferenceStore]:
        def formatter(logs: list[Log]):
            return list(map(lambda log: S4MInteractiveSchema(**log.to_dict()).model_dump(), logs))

        return Inference[InferenceStore](
            name=InferenceMode.InteractiveAgeGender.value,
            workflow=Workflow(
                models=[
                    self._load_face_model(),
                    self._load_max_iou_filter(
                        inference_config.config['focusRegion'][0]['bbox']
                        if len(inference_config.config['focusRegion']) else [0, 0, 1, 1]
                    ),
                    self._load_age_model(),
                    self._load_gender_model(),
                    self._load_object_tracking_model(),
                ],
                loggers=[
                    self._load_age_gender_interactive_logger(),
                ],
            ),
            # interceptor after inference
            post_interceptor=[
                # detect_eye_tracking_abnormal_data
            ],
            draw_functions=[
                draw_region_wrapper,
                draw_result,
                draw_people_count,
                fps_calculate
            ],
            output_formatter=formatter
        )

    def interactive_pose_handler(self) -> Inference[InferenceStore]:
        def formatter(logs: list[Log]):
            return list(map(lambda log: S4MInteractiveSchema(**log.to_dict()).model_dump(), logs))

        return Inference[InferenceStore](
            name=InferenceMode.InteractivePose.value,
            workflow=Workflow(
                models=[
                    self._load_face_model(),
                    self._load_max_iou_filter(
                        inference_config.config['focusRegion'][0]['bbox']
                        if len(inference_config.config['focusRegion']) else [0, 0, 1, 1]
                    ),
                    self._load_head_pose_model(),
                    self._load_object_tracking_model(),
                ],
                loggers=[
                    self._load_pose_interactive_logger(),
                ],
            ),
            # interceptor after inference
            post_interceptor=[
                # detect_eye_tracking_abnormal_data
            ],
            draw_functions=[
                draw_region_wrapper,
                draw_result,
                draw_people_count,
                fps_calculate
            ],
            output_formatter=formatter
        )
