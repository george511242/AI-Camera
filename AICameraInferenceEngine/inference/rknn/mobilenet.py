from ..model import Model, Result
from ..utils import img_utils
from .model import RKNNModel
import cv2
import numpy as np


class MobileNet(RKNNModel):
    def __init__(self, weight_path: str, focus_cls_idx_list: list[int] | None = [487, 528, 707, 620]) -> None:
        super().__init__(weight_path)
        self.input_size = (224, 224)
        self.focus_cls_idx_list = focus_cls_idx_list

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]') -> 'list[Result]':
        for result in results:
            if result.crop_info is not None:
                cls_idx, scores, cls = self.inference_from_crop(img, result.crop_info)
                result.set(
                    imagenet_class_idx=cls_idx,
                    imagenet_class_score=scores,
                    imagenet_class=cls,
                )
            if result.crop_infos is not None:
                imagenet_class_idxes = list()
                imagenet_class_scores = list()
                imagenet_classes = list()
                for crop_info in result.crop_infos:
                    cls_idx, scores, cls = self.inference_from_crop(img, crop_info)
                    imagenet_class_idxes.append(cls_idx)
                    imagenet_class_scores.append(scores)
                    imagenet_classes.append(cls)
                result.set(
                    imagenet_class_idxes=imagenet_class_idxes,
                    imagenet_class_scores=imagenet_class_scores,
                    imagenet_classes=imagenet_classes,
                )
        return results

    def inference_from_crop(self, img, crop_info):
        # preprocess
        crop = img_utils.crop(img, crop_info)
        crop = img_utils.pad_zero(crop, 1)
        crop = img_utils.resize(crop, self.input_size)
        crop = img_utils.switch_br(crop)
        crop = img_utils.to_batch([crop])

        # inference
        predict = self.session.inference([crop])

        # postprocess
        scores = img_utils.softmax(predict[0], -1)
        scores = np.squeeze(scores)
        cls_idx_sorted = np.argsort(scores)[::-1]

        if self.focus_cls_idx_list is not None:
            for cls_idx in cls_idx_sorted:
                if cls_idx in self.focus_cls_idx_list:
                    return cls_idx, scores[cls_idx], Model.imagenet_classes[cls_idx]

        return cls_idx_sorted[0], scores[cls_idx_sorted[0]], Model.imagenet_classes[cls_idx_sorted[0]]
