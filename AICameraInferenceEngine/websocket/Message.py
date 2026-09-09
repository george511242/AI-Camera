import json
from datetime import datetime
from typing import Any
import numpy as np

class WebSocketMessage:
    def __init__(self, event: str, data):
        self.event = event
        self.payload = data
    
    def _serialize_payload(self, obj: Any) -> Any:
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_payload(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(key): self._serialize_payload(value) for key, value in obj.items()}
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            # 對於有 __dict__ 的物件，轉換為字典
            return self._serialize_payload(obj.__dict__)
        elif hasattr(obj, 'to_dict'):
            # 如果物件有 to_dict 方法
            return self._serialize_payload(obj.to_dict())
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            print(type(obj))
            return str(obj)
    
    def dump(self):
        return {
            "event": self.event,
            "payload": self._serialize_payload(self.payload)
        }