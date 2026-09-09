class BaseInferenceStore:
    inference_result = None

    def set_inference_result(self, result):
        self.inference_result = result


class InferenceStore(BaseInferenceStore):
    records: list

    def __init__(self):
        self.records = list()

    def clear(self):
        self.clear_records()

    def clear_records(self):
        self.records.clear()
