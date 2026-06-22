import torch

@torch.no_grad()
def evaluate(model, val_data, args):
    model.eval()
    losses = []
    