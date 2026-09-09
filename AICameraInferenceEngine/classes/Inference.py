import time
from typing import Any, Generic, List, TypeVar, Union, Callable
from inference.model import Result, Log, Workflow
from classes.InferenceHandler import InferenceHandler
from classes.InferenceStore import BaseInferenceStore
from classes.tools.FPSCalculator import FPSCalculator

fpsCalculator = FPSCalculator()

T = TypeVar('T')


class Inference(Generic[T]):
    name: str
    fps: int = 0
    # 不同模型組合可動態指定不同的 Data store
    store: Union[T, BaseInferenceStore] = None
    # 執行與處理 Inference
    handler: InferenceHandler = None
    # 儲存最後的 camera frame
    last_frame: Union[Any, None] = None
    # 儲存最後一張經過 draw function 繪製的 frame
    last_inference_frame: Union[Any, None] = None
    # 指定客製化輸出格式的處理函式
    output_formatter: Callable = None
    # 根據 store 內部資料繪製 frame 的函式
    draw_functions: List[Callable] = []
    post_interceptor: List[Callable] = []

    def __init__(
        self,
        name: str,
        workflow: Workflow,
        output_formatter: Callable = None,
        draw_functions: List[Callable] = [],
        post_interceptor: List[Callable] = [],
    ):
        self.name = name
        self.workflow = workflow
        self.output_formatter = output_formatter
        self.draw_functions = draw_functions
        self.fpsCalculator = fpsCalculator
        self.post_interceptor = post_interceptor
        self.results: list[Result] = list()
        self.logs: list[Log] = list()

    def set_last_frame(self, frame):
        self.last_frame = frame

    def get_last_frame(self):
        return self.last_frame
    
    def set_last_inference_frame(self, frame):
        self.last_inference_frame = frame

    def get_last_inference_frame(self):
        return self.last_inference_frame

    def clear_frame(self):
        self.set_last_frame(None)
        self.set_last_inference_frame(None)

    def get_store(self):
        return self.store

    def store_process(self):
        for process in self.store_processors:
            process.set_store(self.store)
            process.execute()

    def post_interceptor_process(self, frame):
        for process in self.post_interceptor:
            process(self, frame)

    def infer(self, frame):
        fpsCalculator.set_start_time(time.time())
        self.results = self.workflow.run(frame, list())
        self.logs = self.workflow.get_logs(frame, self.results)
        self.post_interceptor_process(frame)
        self.fps = fpsCalculator.calculate()
        self.set_last_inference_frame(self.draw(frame))
        return self

    def output(self):
        """
        This method is responsible for formatting the output of the inference process.
        If an output formatter is defined, it uses that to format the store's inference result.
        If no output formatter is defined, it simply returns the raw inference result from the store.
        """
        if self.output_formatter is None:
            return self.logs
        return self.output_formatter(self.logs)

    def draw(self, frame):
        for draw_fn in self.draw_functions:
            frame = draw_fn(frame, self)
        return frame

    def unload(self):
        pass
