import datetime
import threading
import requests
import time
import cv2
from dataModel.StreamSchema import StreamRecordRequestSchema
from settings import settings
from main import inference_manager
from loguru import logger
from utils.readConfig import config, get_config
import time
import redis

POOL = redis.ConnectionPool(host=settings.redis_host,
                            port=settings.redis_port, decode_responses=True)
REDIS = redis.Redis(connection_pool=POOL)

class StreamRemoteRecord():
    _instance_lock = threading.Lock()

    @staticmethod
    def get_instance(payload: StreamRecordRequestSchema):
        with StreamRemoteRecord._instance_lock:
            if not hasattr(StreamRemoteRecord, '_instance'):
                StreamRemoteRecord._instance = StreamRemoteRecord(payload)
                return StreamRemoteRecord._instance

            StreamRemoteRecord._instance.cancel()
            if hasattr(StreamRemoteRecord._instance, 'recording_thread') and StreamRemoteRecord._instance.recording_thread:
                StreamRemoteRecord._instance.recording_thread.join(timeout=5)

            StreamRemoteRecord._instance = StreamRemoteRecord(payload)
            return StreamRemoteRecord._instance

    @staticmethod
    def cancel_recording():
        with StreamRemoteRecord._instance_lock:
            if hasattr(StreamRemoteRecord, '_instance'):
                StreamRemoteRecord._instance.cancel()

    def __init__(self, payload: StreamRecordRequestSchema):
        postUrl = ''
        device_id = get_config('SYSTEM', 'device_id', False)
        if payload.post_url is None or payload.post_url == '':
            postUrl = f"{get_config('API', 'wms_base', False)}/camera/record/streaming/{device_id}"
        else:
            postUrl = payload.post_url
        if device_id is None or device_id == '':
            raise Exception('No device ID found')
        self.url = postUrl
        self.split_time = payload.split_time
        self.start_time = datetime.datetime.fromtimestamp(payload.range.start_time)
        self.end_time = datetime.datetime.fromtimestamp(payload.range.end_time)

        self._stop_event = threading.Event()
        self.recording_thread = None
        self._session = requests.Session()


    def start(self):
        if self.recording_thread is None or not self.recording_thread.is_alive():
            self.recording_thread = threading.Thread(target=self.streamimg_post, daemon=True)
            self.recording_thread.start()
            logger.info("Recording thread started")
        else:
            logger.warning("Recording thread is already running")

    def expired(self):
        return self._stop_event.is_set() or not (self.start_time <= datetime.datetime.now() <= self.end_time)

    def get_frame_data(self):
        start_time = datetime.datetime.now()
        while not self._stop_event.is_set():
            time.sleep(1/10)
            if (datetime.datetime.now() - start_time).total_seconds() > self.split_time or self.expired():
                logger.warning('Stop recording - time limit reached')
                break

            try:
                frame = cv2.cvtColor(inference_manager.get_frame_processor().read(), cv2.COLOR_BGR2RGB)
                ret, frameBytes = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n' + frameBytes.tobytes() + b'\r\n')
                else:
                    logger.error('Failed to encode frame')
                    break
            except Exception as e:
                logger.error(f'Error processing frame: {e}')
                break

        logger.info('Frame data generation stopped')
            
    def get_headers(self):
        token = REDIS.get("accessTokenForOTA") or get_config('WMS', 'access_token', '')
        return {
            "Authorization": f"Bearer {token}",
            'Content-Type': 'application/octet-stream'
        }
    
    def streamimg_post(self):
        logger.warning(f'Triggered schedule recording from start time ({self.start_time}) to end time ({self.end_time})')

        while datetime.datetime.now() < self.start_time and not self._stop_event.is_set():
            time.sleep(1)

        if self._stop_event.is_set():
            logger.info('Recording cancelled before start time')
            return

        try:
            while not self.expired():
                logger.warning('Start recording segment')
                data_generator = (data for data in self.get_frame_data())

                try:
                    response = self._session.post(
                        self.url,
                        data=data_generator,
                        stream=True,
                        headers=self.get_headers(),
                        timeout=(10, 30)  # (connect_timeout, read_timeout)
                    )
                    response.raise_for_status()
                    logger.info(f'Recording segment uploaded successfully. Status: {response.status_code}')
                except requests.exceptions.RequestException as e:
                    logger.error(f'Network error during recording upload: {e}')
                    if not self._stop_event.is_set():
                        time.sleep(5)  # 等待重試

                if self._stop_event.is_set():
                    logger.info('Recording cancelled during upload')
                    break

        except Exception as e:
            logger.error(f'Unexpected error in recording: {e}')
        finally:
            self._session.close()
            logger.warning('Finished recording')
    
    def cancel(self):
        logger.warning('Cancel recording')
        self._stop_event.set()
        self.end_time = datetime.datetime.now()

        if hasattr(self, '_session'):
            try:
                self._session.close()
            except Exception as e:
                logger.error(f'Error closing session: {e}')

        logger.info('Recording cancellation completed')
