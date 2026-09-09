
from typing import List, Callable
from inference.model import Model

class InferenceHandler():
    base_model: Model
    sub_models: List[Model or Callable]
    pipeline_data = None
    post_processors: List[Callable] = []
    
    def __init__(self, base_model: Model, sub_models: List[Model] or Callable = [], post_processors = []) -> None:
        self.base_model = base_model
        self.sub_models = sub_models
        self.post_processors = post_processors
    
    def post_process(self, image):
        for processor in self.post_processors:
            self.pipeline_data = processor(self.pipeline_data, image)
    
    def inference(self, image):
        self.pipeline_data = self.base_model.fit(image)
        
        for model in self.sub_models:
            if isinstance(model, Model):
                self.pipeline_data = model.fit(image, self.pipeline_data)
            else:
                self.pipeline_data = model(image, self.pipeline_data)
        self.post_process(image)
        return self.pipeline_data