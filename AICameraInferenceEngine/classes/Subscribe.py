from typing import Callable, Union, Any

class Publisher:
    def __init__(self):
        self._subscribers = {}

    def subscribe(self, event_type: str, callback: Callable):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(callback)

    def publish(self, event_type: str, data: Union[Any, None] = None):
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                callback(data)