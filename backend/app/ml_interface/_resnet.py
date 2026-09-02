"""Holds the loaded ResNet-50 model singleton. Populated at startup by ml_interface.load_models()."""

import torch

model = None
device = torch.device("cpu")
