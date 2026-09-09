from ..model import Model, Result, Log
import cv2


class LoggerModel(Model):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]', ts: int = 0) -> 'list[Log]':
        raise NotImplementedError('__call__ method not implemented')
