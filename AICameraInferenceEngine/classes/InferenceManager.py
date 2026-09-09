from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from frame_process import FrameProcessorWithThread

from classes.Inference import Inference

class InferenceManager:
    instance: Inference = None
    frame_processor: Optional["FrameProcessorWithThread"] = None

    def set_frame_processor(self, frame_processor):
        self.frame_processor = frame_processor

    def get_frame_processor(self) -> "FrameProcessorWithThread":
        return self.frame_processor

    def set_inference(self, inference):
        self.instance = inference

    def get_inference(self):
        return self.instance
