from ..model import Model
import torch


class TorchModel(Model):
    def __init__(self, model, weight_path: str | None) -> None:
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model
        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path, weights_only=False, map_location=self.device))
        self.model.to(device=self.device)
        self.model.eval()
