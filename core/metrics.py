from torch import nn

class AccuracyMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        predictions = outputs.argmax(dim=1)
        correct_predictions = (predictions == targets).sum().float()
        return correct_predictions / len(targets)
