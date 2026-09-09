from fastapi import APIRouter, status, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse
from main import inference_manager
from dataModel.StreamSchema import StreamRecordRequestSchema
from classes.StreamRemoteRecord import StreamRemoteRecord
from dataModel.AppSchema import HealthSchema
from utils.readConfig import reload_config
from classes.InferenceConfig import inference_config
import io
import asyncio
from loguru import logger
from utils.publisher import publisher, PublishEvent
import wave
from pathlib import Path

AppRouter = APIRouter(tags=['App'])


@AppRouter.get('/updateConfig')
def update_config():
    inference_config.update_from_remote()
    
@AppRouter.get('/stream/capture')
async def capture():
    try:
        # 在執行緒池中執行可能阻塞的操作
        image_bytes = await asyncio.get_event_loop().run_in_executor(
            None, inference_manager.get_frame_processor().read_jpg
        )
        if image_bytes is not None:
            return StreamingResponse(io.BytesIO(image_bytes.tobytes()), media_type="image/png")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Unable to capture camera frame')
    except asyncio.TimeoutError:
        raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail='Capture timeout')
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Capture error: {str(e)}')
    
@AppRouter.post('/fetch-settings')
def fetch_settings():
    reload_config()
    
@AppRouter.post('/fetch-signature')
def fetch_signature():
    publisher.publish(PublishEvent.UPDATED_SIGNATURE)
    
def send_frame(payload: StreamRecordRequestSchema):
    streamRecord = StreamRemoteRecord.get_instance(payload)
    streamRecord.start()

@AppRouter.post('/stream/recording')
async def start_recording(payload: StreamRecordRequestSchema, background_tasks: BackgroundTasks):
    try:
        if hasattr(StreamRemoteRecord, '_instance'):
            existing_instance = StreamRemoteRecord._instance
            if (hasattr(existing_instance, 'recording_thread') and
                existing_instance.recording_thread and
                existing_instance.recording_thread.is_alive()):
                logger.warning("Recording is already in progress, cancelling existing recording first")
                StreamRemoteRecord.cancel_recording()

        background_tasks.add_task(send_frame, payload)
        logger.info("Recording task scheduled successfully")
        return {'status': 'ok', 'message': 'Recording started'}
    except Exception as e:
        logger.error(f'Failed to start recording: {str(e)}')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Failed to start recording: {str(e)}')

@AppRouter.delete('/stream/recording')
def stop_recording():
    try:
        if hasattr(StreamRemoteRecord, '_instance'):
            instance = StreamRemoteRecord._instance
            if (hasattr(instance, 'recording_thread') and
                instance.recording_thread and
                instance.recording_thread.is_alive()):
                StreamRemoteRecord.cancel_recording()
                logger.info("Recording cancelled successfully")
                return {'status': 'ok', 'message': 'Recording stopped'}
            else:
                logger.warning("No active recording found")
                return {'status': 'warning', 'message': 'No active recording to stop'}
        else:
            logger.warning("No recording instance found")
            return {'status': 'warning', 'message': 'No recording instance found'}
    except Exception as e:
        logger.error(f'Failed to stop recording: {str(e)}')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Failed to stop recording: {str(e)}')

@AppRouter.get('/stream/recording/status')
def get_recording_status():
    try:
        if hasattr(StreamRemoteRecord, '_instance'):
            instance = StreamRemoteRecord._instance
            if (hasattr(instance, 'recording_thread') and
                instance.recording_thread and
                instance.recording_thread.is_alive()):
                return {
                    'status': 'recording',
                    'start_time': instance.start_time.isoformat(),
                    'end_time': instance.end_time.isoformat(),
                    'url': instance.url
                }
            else:
                return {'status': 'idle', 'message': 'No active recording'}
        else:
            return {'status': 'idle', 'message': 'No recording instance'}
    except Exception as e:
        logger.error(f'Failed to get recording status: {str(e)}')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Failed to get recording status: {str(e)}')

@AppRouter.get('/health')
def health():
    try:
        frame = inference_manager.get_frame_processor().read()
        if frame is None:
            return HealthSchema(health=False)
        return HealthSchema(health=True)
    except Exception as e:
        return HealthSchema(health=False, error=str(e))

@AppRouter.post('/audio/upload')
async def upload_audio_file(file: UploadFile = File(...)):
    """
    Upload a .wav audio file to /app/audio/no-phone.wav
    Validates that the file is a valid WAV format before saving.
    """
    try:
        # Check file extension
        if not file.filename.lower().endswith('.wav'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Only .wav files are allowed'
            )

        # Read file content
        content = await file.read()

        # Validate WAV format by attempting to read it with wave module
        try:
            with io.BytesIO(content) as wav_buffer:
                with wave.open(wav_buffer, 'rb') as wav_file:
                    # Verify it's a valid WAV file by reading parameters
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    frame_rate = wav_file.getframerate()
                    frames = wav_file.getnframes()

                    logger.info(
                        f'Valid WAV file: {channels} channels, '
                        f'{sample_width} bytes/sample, '
                        f'{frame_rate} Hz, {frames} frames'
                    )
        except wave.Error as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Invalid WAV file format: {str(e)}'
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Unable to validate WAV file: {str(e)}'
            )

        # Create audio directory if it doesn't exist
        audio_dir = Path('/app/audio')
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Save file to /app/audio/no-phone.wav
        file_path = audio_dir / 'no-phone.wav'
        with open(file_path, 'wb') as f:
            f.write(content)

        logger.info(f'Audio file uploaded successfully to {file_path}')

        return {
            'status': 'ok',
            'message': 'Audio file uploaded successfully',
            'path': str(file_path),
            'size': len(content),
            'channels': channels,
            'sample_rate': frame_rate
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to upload audio file: {str(e)}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to upload audio file: {str(e)}'
        )
