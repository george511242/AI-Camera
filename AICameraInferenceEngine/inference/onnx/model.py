import onnxruntime as ort
from ..model import Model


ort.preload_dlls()


class ONNXModel(Model):
    def __init__(self, weight_path: str) -> None:
        super().__init__()
        self.weight_path = weight_path
        self.session = ort.InferenceSession(
            self.weight_path,
            sess_options=ort.SessionOptions(),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            provider_options=[dict(), dict()],
        )
        self.input_name = [i.name for i in self.session.get_inputs()]
        self.output_name = [i.name for i in self.session.get_outputs()]
