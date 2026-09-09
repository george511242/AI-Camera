import json
import requests
import time
from utils.readConfig import config
from typing import Callable, List
from loguru import logger
from settings import settings


class InferenceConfig:
    config: dict
    listener: List = []

    def __init__(self):
        self.config = dict()

    def on_config_loaded(self, callback: Callable):
        self.listener.append(callback)

    def remove_listener(self, callback: Callable):
        self.listener.remove(callback)

    def load_config(self):
        with open('config.json', 'r') as f:
            self.config.clear()
            self.config.update(json.load(f))

    def get(self, key: str, default = None):
        cfg = self.config
        if key in cfg:
            return cfg[key]
        return default

    def update_from_remote(self):
        self.load_config()
        while True:
            try:
                configFromRemote = requests.get(f'{settings.backend_url}/get_box_info').json()
                logger.debug('Got box info from web: {data}', data=configFromRemote)
                self.config['mode'] = configFromRemote.get('mode')
                self.config['maxBBoxDistance'] = (configFromRemote.get('maxDistance', 20) or 20) / 100
                self.config['maxDisappearCount'] = configFromRemote.get('maxIntevalTimes', 5) or 5
                self.config['minGazeCount'] = configFromRemote.get('minGazeCount', min(5, self.config['maxDisappearCount']) or 5)
                self.config['scoreThreshold'] = configFromRemote.get('scoreThreshold', 0.75)
                self.config['gazeToCameraRange'] = {
                    'pitchMax': configFromRemote.get('gazeToCameraRange', {}).get('pitchMax', 30),
                    'pitchMin': configFromRemote.get('gazeToCameraRange', {}).get('pitchMin', -10),
                    'yawMax': configFromRemote.get('gazeToCameraRange', {}).get('yawMax', 15),
                    'yawMin': configFromRemote.get('gazeToCameraRange', {}).get('yawMin', -15)
                }
                self.config['custom'] = configFromRemote.get('custom', {}) or {}
                camera_width = int(config.get_config("CAMERA", "camera_width", 640))
                camera_height = int(config.get_config("CAMERA", "camera_height", 480))
                hotZones = configFromRemote.get('hotZones', []) or []
                self.config['focusRegion'] = []
                for region in hotZones:
                    if 'start' in region and 'end' in region:
                        # Format with start/end points
                        self.config['focusRegion'].append({
                            'name': region.get('name', ''),
                            'bbox': [
                                region['start']['x'],
                                region['start']['y'],
                                region['end']['x'],
                                region['end']['y'],
                            ]
                        })
                    else:
                        # Original format with x, y, w, h
                        self.config['focusRegion'].append({
                            'name': region.get('name', ''),
                            'bbox': [
                                region['x'] / camera_width,
                                region['y'] / camera_height,
                                (region['x'] + region['w']) / camera_width,
                                (region['y'] + region['h']) / camera_height,
                            ]
                        })
                print(f"Updated focusRegion: {self.config['focusRegion']}")
                # Serializing config & write
                configObj = json.dumps(self.config, indent=2)
                with open('config.json', 'w') as f:
                    f.write(configObj)
                break
            except Exception as e:
                logger.warning(f'Failed to fetch box info, try within 5 seconds {e}')
                time.sleep(5)
                continue
        for callback in self.listener:
            callback(self.config)
        self.load_config()

    def to_dict(self):
        return self.config

inference_config = InferenceConfig()
