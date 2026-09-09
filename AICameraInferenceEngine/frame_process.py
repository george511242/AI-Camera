from typing import Tuple
import cv2, subprocess, time
from loguru import logger
from m01_frame_processor import FrameProcessor
from classes.InferenceManager import InferenceManager
from settings import settings
from factory.InferenceFactory import draw_region_wrapper

class FrameProcessorWithThread:
    def __init__(self, inference_manager: InferenceManager, rotate: int = 0, resolution: Tuple[int, int] = (640, 480), camera_source: str = None, mjpeg_port: int = 8080):
        self.resolution = resolution
        self.inference_manager = inference_manager
        self.rotate = rotate

        source = camera_source if camera_source else "0"
        logger.info(f'[Frame processor] Creating m01 FrameProcessor - Source: {source}, Resolution: {resolution[0]}x{resolution[1]}')

        self.processor = FrameProcessor(
            source=source,
            width=resolution[0],
            height=resolution[1],
            mjpeg_port=mjpeg_port,
        )

        # Apply rotation
        if rotate > 0:
            self.processor.set_rotation(rotate)
        self.processor.set_retry_policy("forever", interval_secs=2.0)
        self.processor.set_source_fallbacks([
            "/dev/video0",
            "/dev/video1",
            "/dev/video2",
            "/dev/video3",
        ])
        self.processor.set_target_fps(15)
        self.processor.set_target_stream_fps(30)

        # Start capture and MJPEG server
        self.processor.start()
        logger.info(f'[Frame processor] m01 FrameProcessor started - Source: {source}, MJPEG port: {mjpeg_port}')

    def set_rotate(self, rotate: int):
        logger.info('[Frame processor] rotate updated')
        self.rotate = rotate
        self.processor.set_rotation(rotate)

    def set_resolution(self, resolution: Tuple[int, int] = (640, 480)):
        logger.info(f'[Frame processor] Resolution updated to {resolution[0]}x{resolution[1]}')
        self.resolution = resolution
        self.processor.set_resolution(resolution[0], resolution[1])

    def update_source(self, source: str):
        logger.info(f'[Frame processor] Updating source to: {source}')
        self.processor.update_source(source)
        logger.info(f'[Frame processor] Successfully switched to new source: {source}')

    def read(self):
        """Get latest frame in RGB format (same as original)."""
        frame = self.processor.get_latest_frame()
        if frame is not None:
            # m01 FrameProcessor returns RGB, inference expects RGB
            return frame
        return None

    def read_jpg(self):
        frame = self.read()
        if frame is not None:
            frame = draw_region_wrapper(frame, self.inference_manager.get_inference())
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                return jpeg
        no_frame = cv2.imread('noImage.jpg')
        no_frame = cv2.resize(no_frame, (self.resolution[0], self.resolution[1]))
        ret, jpeg = cv2.imencode('.jpg', no_frame)
        return jpeg

    def stop(self):
        self.processor.stop()

    def release(self):
        self.processor.stop()
