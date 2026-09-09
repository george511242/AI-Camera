import json, threading, time, resource, traceback, asyncio
from websocket.Message import WebSocketMessage
import upload_to_wms
from classes.QueueConfig import QueueConfig
from fastapi import FastAPI
from collections import deque
from loguru import logger
from utils.log import truncate_long_strings
from contextlib import asynccontextmanager
from factory.InferenceFactory import InferenceFactory
from dataModel.modelOutput import InferenceMode
from frame_process import FrameProcessorWithThread
from settings import settings
from classes.InferenceManager import InferenceManager
from classes.RedisQueue import RedisQueue
from classes.InferenceConfig import inference_config
from utils.readConfig import config, get_config
from utils.publisher import publisher, PublishEvent
from classes.SqliteQueue import SqliteQueue
from websocket.ConnectionManager import connection_manager
from factory.InferenceWebSocketPayloadFactory import create_inference_websocket_payload
from factory.InferenceResultPayloadFactory import wrapper_payload
from utils.audio import restore_usb_audio_volume

background_loop = None

def setup_background_loop():
    global background_loop
    background_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(background_loop)
    background_loop.run_forever()

background_thread = threading.Thread(target=setup_background_loop, daemon=True)
background_thread.start()
time.sleep(0.1)

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

queueConfig = QueueConfig()
sqliteQueue = SqliteQueue(db_path=settings.temp_queue_sqlite_path, maxlen=queueConfig.max_size, max_days=queueConfig.max_days)
upload_queue = RedisQueue('sync_to_wms', 'sync_to_wms_temp', settings.redis_host, settings.redis_port, maxlen=queueConfig.max_size, sqliteQueue=sqliteQueue)

inference_manager = InferenceManager()

# get camera stream
camera_width = int(get_config("CAMERA", "camera_width", 640))
camera_height = int(get_config("CAMERA", "camera_height", 480))
camera_rotate = int(get_config("CAMERA", "rotate_90_degree", 0))
logger.debug('Load camera config: {extra}', extra={ "camera_width": camera_width, "camera_height": camera_height, "camera_rotate": camera_rotate })


def updateConfig(config):
    inferencer = inference_manager.get_inference()
    if 'mode' in config:
        inference_manager.set_inference(InferenceFactory().load(config['mode']))
    else:
        inference_manager.set_inference(InferenceFactory().load(InferenceMode.LeaveZoneWithAgeGender.value))

inference_config.on_config_loaded(updateConfig)

def sync_to_wms():
    logger.debug('WMS sync service running')
    delay = 20
    batch_size = 30

    while True:
        upload_queue.check_redis_connection()
        payload = upload_queue.pop_from_queue(batch_size)
        if len(payload) > 0:
            logger.debug(f'Got upload payload {len(payload)}')
            try:
                response = upload_to_wms.upload(payload)
                logger.info(f'Synced {len(payload)} records to WMS')
                delay = int(response.get("periodTimeSec", 20))
                batch_size = int(response.get("sizeLimit", 30))
            except Exception as e:
                logger.error(f'Error during WMS sync: {e}')
                upload_queue.clear_temp_queue(False, payload)

        time.sleep(delay)

def main():
    poll_interval = 0.015
    while (True):
        time.sleep(poll_interval)
        try:
            inferencer = inference_manager.get_inference()
            frame_processor = inference_manager.get_frame_processor()
            if not inferencer or not frame_processor:
                continue
            frame = frame_processor.read()

            if (frame is None):
                time.sleep(2)
                logger.warning('Waiting for camera frame')
                continue

            inference_result = inferencer.infer(frame)

            # Broadcast all inference result when there is at least one client connected
            if connection_manager.get_connection_count() > 0:
                try:
                    data = {
                        "fps": inferencer.fps,
                        "results": create_inference_websocket_payload(frame, inference_result.results, inference_config)
                    }
                    asyncio.run_coroutine_threadsafe(
                        connection_manager.broadcast(WebSocketMessage('inference', data=data)),
                        background_loop
                    )
                except Exception as ws_error:
                    logger.warning(f'WebSocket inference broadcast error: {str(ws_error)}')

                try:
                    websocket_data = wrapper_payload(inference_result.output(), inferencer.logs, inference_config, 'websocket')
                    if len(websocket_data):
                        asyncio.run_coroutine_threadsafe(
                            connection_manager.broadcast(WebSocketMessage('result', data=websocket_data)),
                            background_loop
                        )
                except Exception as ws_error:
                    logger.warning(f'WebSocket result broadcast error: {str(ws_error)}')

            result_data = wrapper_payload(inference_result.output(), inferencer.logs, inference_config, 'record')
            if len(result_data):
                logger.debug('Detect actions: {extra}', extra=truncate_long_strings(result_data))
                for record in result_data:
                    upload_queue.push_to_queue(json.dumps(record))

        except Exception as e:
            logger.error(f'[person tracking:main.py]: {traceback.format_exc()}')
            time.sleep(10)
            continue

sync_to_wms_thread = threading.Thread(target=sync_to_wms, daemon=True)
inference_thread = threading.Thread(target=main, daemon=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        restore_usb_audio_volume()
    except Exception as e:
        logger.error(f'Failed to restore USB audio volume: {str(e)}')
    inference_config.update_from_remote()
    sync_to_wms_thread.start()
    streaming_source = get_config("STREAMING", "SOURCE", None)
    inference_manager.set_frame_processor(FrameProcessorWithThread(inference_manager, camera_rotate, (camera_width, camera_height), camera_source=streaming_source))
    inference_thread.start()
    yield

def update_config(data):
    rotate = int(get_config("CAMERA", "rotate_90_degree", 0))
    width = int(get_config("CAMERA", "camera_width", 640))
    height = int(get_config("CAMERA", "camera_height", 480))
    logger.debug('API Request [Camera Config]: {extra}', extra={ "camera_width": width, "camera_height": height, "camera_rotate": rotate })
    inference_manager.get_frame_processor().set_rotate(rotate)
    inference_manager.get_frame_processor().set_resolution((width, height))
    upload_queue.maxlen = queueConfig.max_size
    sqliteQueue.max_days = queueConfig.max_days
    sqliteQueue.maxlen = queueConfig.max_size

publisher.subscribe(PublishEvent.UPDATED_CONFIG, update_config)