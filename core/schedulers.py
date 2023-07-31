# import exponential scheduler

from torch.optim.lr_scheduler import ExponentialLR

class ExponentialLR(ExponentialLR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
