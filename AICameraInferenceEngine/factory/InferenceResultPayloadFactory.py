import cv2
import base64
from classes import InferenceConfig
from inference.model.result import Result

def find_logs_by_id (logs, id):
    for log in logs:
      if log.id == id:
        return log
    return None

def convert_frame_to_base64(frame):
    # convert color
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def wrapper_payload(results: list[Result], logs, config: InferenceConfig, record_type = 'record') -> list[dict]:
    """
    Wraps inference results with additional snapshot data based on configuration and record type.

    Args:
        results (list[Result]): List of inference result dictionaries.
        logs: Log data containing snapshots and related information.
        config (InferenceConfig): Configuration object for inference settings.
        record_type (str, optional): Type of record to determine which snapshot to use ('record' or 'websocket'). Defaults to 'record'.

    Returns:
        list[dict]: List of result dictionaries, each potentially augmented with a 'snapshot' field if in 'phone-pose-detection' mode; otherwise, returns the original results.
    """
    if config.get('mode') == 'phone-pose-detection':
        wrapped_results = []
        for result in results:
            if result['mode'] == 'phone-pose-detection':
                person_id = result['personId']
                log = find_logs_by_id(logs, person_id)
                if record_type == 'record':
                    result['snapshot'] = convert_frame_to_base64(log.snapshot) if log.snapshot is not None else None
                else:
                    result['snapshot'] = convert_frame_to_base64(log.snapshot_mosaic) if log.snapshot_mosaic is not None else None
            wrapped_results.append(result)
        return wrapped_results
    return results
