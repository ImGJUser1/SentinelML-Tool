import torch
import numpy as np

class TorchAdapter:
    def __init__(self, model, device="cpu"):
        self.model = model.eval().to(device)
        self.device = device

    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.to(self.device)

        with torch.no_grad():
            out = self.model(x.unsqueeze(0))

        return out.cpu().numpy().squeeze()