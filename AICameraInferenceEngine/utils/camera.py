import cv2, os


def get_video_device():
    device_list: list[str] = os.listdir('/dev')
    video_list = [ device for device in device_list if device.startswith('video') ]
    return video_list

def get_max_video_device_index():
    video_list = get_video_device()
    video_list.sort()
    if len(video_list) == 0:
        return -1
    return int(video_list.pop().replace('video', ''))

def list_cameras():
    index = 0
    camera_idx = []
    failed = 0
    max_gaps = 3

    while failed < max_gaps:
        vcap = cv2.VideoCapture(index)
        if not vcap.read()[0]:
            failed += 1
        else:
            camera_idx.append(index)
        vcap.release()
        index += 1
    return camera_idx

def get_first_available_camera():
    index = 0
    failed = 0
    max_tries = get_max_video_device_index()

    while failed < max_tries:
        capture = cv2.VideoCapture(index)
        if not capture.read()[0]:
            failed += 1
        else:
            capture.release()
            return index
        capture.release()
        index += 1
    return -1