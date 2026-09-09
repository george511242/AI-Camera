from ..model import Model
from rknnlite.api import RKNNLite


class RKNNModel(Model):
    def __init__(self, weight_path: str) -> None:
        super().__init__()
        self.weight_path = weight_path
        self.session = RKNNLite()

        self.is_fail = self.session.load_rknn(weight_path)
        if self.is_fail != 0:
            print('Load RKNN model failed')

        self.is_fail = self.session.init_runtime()
        if self.is_fail != 0:
            print('Init runtime environment failed')

    def __del__(self):
        self.session.release()
