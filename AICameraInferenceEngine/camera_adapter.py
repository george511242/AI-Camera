from abc import ABC, abstractmethod
from typing import Tuple, Optional
import cv2
import numpy as np
import os
from loguru import logger


class CameraAdapter(ABC):
    def __init__(self, source: str):
        self.source = source
        self.capture: Optional[cv2.VideoCapture] = None

    @abstractmethod
    def open(self) -> bool:
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        pass

    @abstractmethod
    def grab(self) -> bool:
        pass

    @abstractmethod
    def retrieve(self) -> Tuple[bool, Optional[np.ndarray]]:
        pass

    @abstractmethod
    def reconnect(self) -> bool:
        pass

    @abstractmethod
    def release(self):
        pass

    @abstractmethod
    def get_type(self) -> str:
        pass


class LocalCameraAdapter(CameraAdapter):

    def __init__(self, source: str, resolution: Tuple[int, int] = (640, 480)):
        super().__init__(source)
        self.resolution = resolution

    def open(self) -> bool:
        try:
            # 判斷是否為數字索引
            if self.source.isdigit():
                camera_index = int(self.source)
                logger.info(f'[LocalCamera] Opening camera with index: {camera_index}')
                self.capture = cv2.VideoCapture(camera_index)
            else:
                logger.info(f'[LocalCamera] Opening camera: {self.source}')
                self.capture = cv2.VideoCapture(self.source)

            if not self.capture.isOpened():
                logger.error(f'[LocalCamera] Failed to open camera: {self.source}')
                return False

            # 設定解析度
            logger.debug(f'[LocalCamera] Setting resolution: {self.resolution[0]}x{self.resolution[1]}')
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # 設定 buffer size 為 1，確保取得最新幀
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # 設定 MJPEG 編碼（如果支援）
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

            logger.info(f'[LocalCamera] Camera initialized: {self.source}')
            return True

        except Exception as e:
            logger.error(f'[LocalCamera] Error opening camera: {str(e)}')
            return False

    def is_opened(self) -> bool:
        return self.capture is not None and self.capture.isOpened()

    def grab(self) -> bool:
        if self.capture is None:
            return False
        return self.capture.grab()

    def retrieve(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.capture is None:
            return False, None
        return self.capture.retrieve()

    def reconnect(self) -> bool:
        logger.warning('[LocalCamera] Reconnecting to camera')
        self.release()
        return self.open()

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            logger.info('[LocalCamera] Camera released')

    def get_type(self) -> str:
        return "LocalCamera"


class RTSPCameraAdapter(CameraAdapter):

    def __init__(self, source: str):
        super().__init__(source)

    def open(self) -> bool:
        try:
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;tcp|'
                'buffer_size;256|'
                'max_delay;100000|'
                'fflags;nobuffer|'
                'flags;low_delay'
            )

            logger.info(f'[RTSP] Opening RTSP stream with aggressive low-latency settings: {self.source}')
            self.capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

            if not self.capture.isOpened():
                logger.error(f'[RTSP] Failed to open RTSP stream: {self.source}')
                return False

            logger.info('[RTSP] RTSP stream opened with TCP transport and aggressive low-latency mode')

            # 設定 buffer size 為 1（對本地相機有效，對 RTSP 效果有限）
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            logger.debug('[RTSP] Flushing initial buffer (30 frames)')
            for _ in range(30):
                self.capture.grab()

            logger.info(f'[RTSP] RTSP stream initialized with aggressive low latency: {self.source}')
            return True

        except Exception as e:
            logger.error(f'[RTSP] Error opening stream: {str(e)}')
            return False

    def is_opened(self) -> bool:
        return self.capture is not None and self.capture.isOpened()

    def grab(self) -> bool:
        if self.capture is None:
            return False
        return self.capture.grab()

    def retrieve(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.capture is None:
            return False, None
        return self.capture.retrieve()

    def reconnect(self) -> bool:
        logger.warning('[RTSP] Reconnecting to RTSP stream')
        self.release()
        return self.open()

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            logger.info('[RTSP] RTSP stream released')

    def get_type(self) -> str:
        return "RTSP Stream"


def create_camera_adapter(source: Optional[str] = None, resolution: Tuple[int, int] = (640, 480)) -> CameraAdapter:
    """
    工廠函數：根據 source 自動創建對應的 CameraAdapter

    Args:
        source: 相機來源，可以是：
            - None: 從環境變數 CAMERA_SOURCE 讀取，預設為 "0"
            - "rtsp://...": RTSP 串流 URL
            - "/dev/videoN": Linux 設備路徑
            - "0", "1", ...: 相機索引數字
        resolution: 解析度 (width, height)，只對本地相機有效

    Returns:
        CameraAdapter 實例
    """
    if source is None:
        source = os.getenv('CAMERA_SOURCE', '0')
        logger.info(f'[CameraFactory] Using camera source from env: {source}')

    # 判斷是否為 RTSP
    if source.startswith('rtsp://') or source.startswith('rtmp://'):
        logger.info(f'[CameraFactory] Creating RTSP adapter for: {source}')
        return RTSPCameraAdapter(source)
    else:
        logger.info(f'[CameraFactory] Creating local camera adapter for: {source}')
        return LocalCameraAdapter(source, resolution)
