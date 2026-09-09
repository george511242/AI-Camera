from fastapi import APIRouter, status, HTTPException, Depends
from dataModel.StreamSchema import StreamingRequestSchema
from repository.ConfigRepository import ConfigRepository
from utils.audio import get_device_usb_audio, set_device_usb_audio_volume, get_device_usb_audio_volume
from utils.camera import get_video_device
from main import inference_manager

DeviceRouter = APIRouter(tags=['App'], prefix='/api/v1/device')


@DeviceRouter.get('/audio')
def get_usb_audio():
    return get_device_usb_audio()

@DeviceRouter.get('/audio/{index}')
def get_usb_audio_volume(index: int):
    try:
        volume = get_device_usb_audio_volume(index)
        return {'index': index, 'volume': volume}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@DeviceRouter.put('/audio/{index}/{volume}')
def set_usb_audio(index: int, volume: int):
    set_device_usb_audio_volume(index, volume)
    return {'status': 'ok'}

@DeviceRouter.get('/webcams')
def get_webcams():
    return get_video_device()

@DeviceRouter.put('/stream')
def set_streaming(payload: StreamingRequestSchema, config_repository: ConfigRepository = Depends()):
    config_repository.set_config('STREAMING', 'SOURCE', str(payload.source))
    config_repository.save()
    inference_manager.get_frame_processor().update_source(str(payload.source))
